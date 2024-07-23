//
// Created by Stan Wang on 2022/12/23.
//
#include "include/smith/RegionGeneration.h"
#include "include/smith/TypeGeneration.h"
#include "include/smith/generators/OpGeneration.h"

OpGenerator tensorCastGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(operandCandidates, "tensor.cast");
    auto srcTy = src.type.dyn_cast<RankedTensorType>();
    SmallVector<int64_t> newShape;
    if (srcTy.hasStaticShape()) {
      for (int i = 0; i < srcTy.getRank(); ++i) {
        newShape.push_back(ShapedType::kDynamic);
      }
    } else {
      for (int i = 0; i < srcTy.getRank(); ++i) {
        if (srcTy.isDynamicDim(i)) {
          newShape.push_back(dimPool[UR(dimPool.size())]);
        } else {
          newShape.push_back(srcTy.getDimSize(i));
        }
      }
    }

    auto destTy = RankedTensorType::get(newShape, srcTy.getElementType());
    auto res = builder.create<tensor::CastOp>(loc, destTy, src.val);
    region.pool.addDynamicShapedTensor(TypeValue(destTy, res), "tensor.cast");
    return res.getOperation();
  };
}

static RankedTensorType
computeTensorReshapeCollapsedType(RankedTensorType type,
                                  ArrayRef<AffineMap> reassociation) {
  auto shape = type.getShape();
  SmallVector<int64_t, 4> newShape;
  newShape.reserve(reassociation.size());

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert(isReassociationValid(reassociation) && "invalid reassociation");
  unsigned currentDim = 0;
  for (AffineMap m : reassociation) {
    unsigned dim = m.getNumResults();
    auto band = shape.slice(currentDim, dim);
    int64_t size = 1;
    if (llvm::is_contained(band, ShapedType::kDynamic))
      size = ShapedType::kDynamic;
    else
      for (unsigned d = 0; d < dim; ++d)
        size *= shape[currentDim + d];
    newShape.push_back(size);
    currentDim += dim;
  }

  return RankedTensorType::get(newShape, type.getElementType());
}

OpGenerator tensorCollapseShapeGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue typeValue) {
          return typeValue.type.dyn_cast<TensorType>().getRank() >= 2;
        });
    if (candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(2);
      shape.push_back(2);
      auto type = randomIntOrFloatType(builder.getContext());
      candidates.push_back(region.pool.generateStaticShapedTensor(
          builder, loc, RankedTensorType::get(shape, type)));
    }
    auto source = sampleTypedValueFrom(candidates, "tensor.collapse_shape");
    auto srcTy = source.type.dyn_cast<RankedTensorType>();
    SmallVector<ReassociationIndices> reIdxes;
    reIdxes.push_back({0, 1});
    for (int i = 2; i < srcTy.getRank(); ++i) {
      ReassociationIndices indices;
      indices.push_back(i);
      if (UR(2) && srcTy.getRank() - i > 1) {
        indices.push_back(i + 1);
        i++;
      }
      reIdxes.push_back(indices);
    }
    auto res =
        builder.create<tensor::CollapseShapeOp>(loc, source.val, reIdxes);
    auto resultType = computeTensorReshapeCollapsedType(
        srcTy, getSymbolLessAffineMaps(convertReassociationIndicesToExprs(
                   builder.getContext(), reIdxes)));
    auto tVal = TypeValue(resultType, res);
    region.pool.addRankedTensor(tVal, "tensor.collapse_shape");
    return res.getOperation();
  };
}

OpGenerator tensorDimGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return tval.type.dyn_cast<RankedTensorType>().getRank() > 0;
        });
    if (candidates.empty()) {
      candidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto source = sampleTypedValueFrom(candidates, "tensor.dim");
    auto srcTy = source.type.dyn_cast<RankedTensorType>();
    auto num = UR(srcTy.getRank());
    auto idx = region.pool.getConstantIndex(builder, loc, num);
    assert(llvm::detail::isPresent(idx));
    auto res = builder.create<tensor::DimOp>(loc, source.val, idx);
    region.pool.addIndex(TypeValue(IndexType::get(builder.getContext()), res),
                         "tensor.dim");
    return res.getOperation();
  };
}

OpGenerator tensorEmptyGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    if (UR(2)) {
      auto tensorTy = randomStaticShapedTensorType(builder.getContext())
                          .dyn_cast<RankedTensorType>();
      auto res = builder.create<tensor::EmptyOp>(loc, tensorTy.getShape(),
                                                 tensorTy.getElementType());

      region.pool.addStaticShapedTensor(TypeValue(tensorTy, res),
                                        "tensor.empty");
      return res.getOperation();
    } else {
      auto shape = shapePool[UR(shapePool.size())];
      SmallVector<Value> dynamicSizes;
      for (size_t i = 0; i < shape.size(); ++i) {
        if (UR(2)) {
          shape[i] = ShapedType::kDynamic;
          dynamicSizes.push_back(
              sampleTypedValueFrom(
                  region.pool.getCandidatesFromIndex(builder, loc),
                  "tensor.empty")
                  .val);
        }
      }
      auto tensorTy = RankedTensorType::get(
          shape, randomIntOrFloatType(builder.getContext()));
      return builder
          .create<tensor::EmptyOp>(loc, tensorTy, ValueRange(dynamicSizes))
          .getOperation();
    }
  };
}

OpGenerator tensorExpandShapeGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue typeValue) {
          auto tensorTy = typeValue.type.dyn_cast<TensorType>();
          return tensorTy.getRank() > 0 &&
                 !tensorTy.isDynamicDim(tensorTy.getRank() - 1);
        });

    auto source = sampleTypedValueFrom(candidates, "tensor.expand_shape");
    auto srcTy = source.type.dyn_cast<RankedTensorType>();

    SmallVector<int64_t> newShape;
    SmallVector<ReassociationIndices> reassociation;
    int idx = 0;
    for (int i = 0; i < srcTy.getRank() - 1; ++i) {
      ReassociationIndices indices;
      indices.push_back(i);
      reassociation.push_back(indices);
      newShape.push_back(srcTy.getDimSize(i));
    }
    ReassociationIndices indices;
    indices.push_back(srcTy.getRank() - 1);
    indices.push_back(srcTy.getRank());
    reassociation.push_back(indices);

    newShape.push_back(srcTy.getDimSize(srcTy.getRank() - 1));
    newShape.push_back(1);
    auto resTy = RankedTensorType::get(newShape, srcTy.getElementType());
    auto res = builder.create<tensor::ExpandShapeOp>(
        loc, resTy, source.val, reassociation, SmallVector<NamedAttribute>());
    region.pool.addRankedTensor(TypeValue(resTy, res), "tensor.expand_shape");
    return res.getOperation();
  };
}

OpGenerator tensorExtractGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedTensor, PoolType::StaticShapedTensor},
        [&](TypeValue) { return true; });
    if (candidates.empty()) {
      candidates.push_back(region.pool.generateDynamicShapedTensor(
          builder, loc,
          UR(2) ? randomStaticShapedTensorType(builder.getContext())
                : randomDynamicShapedTensorType(builder.getContext())));
    }
    auto source = sampleTypedValueFrom(candidates, "tensor.extract");
    auto srcTy = source.type.dyn_cast<ShapedType>();
    auto indices = region.pool.randomIndicesForShapedType(srcTy, builder, loc);
    auto res = builder.create<tensor::ExtractOp>(loc, source.val, indices);
    region.pool.addElement(TypeValue(srcTy.getElementType(), res),
                           "tensor.extract");

    return res.getOperation();
  };
}

OpGenerator tensorExtractSliceGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return tval.type.dyn_cast<RankedTensorType>().getRank() > 0;
        });
    if (srcCandidates.empty()) {
      srcCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "tensor.extract_slice");
    auto srcTy = src.val.getType().dyn_cast<RankedTensorType>();

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    auto idxes = region.pool.getCandidatesFromIndex(builder, loc);
    for (int i = 0; i < srcTy.getRank(); ++i) {
      offsets.push_back(
          sampleTypedValueFrom(idxes, "tensor.extract_slice").val);
      sizes.push_back(sampleTypedValueFrom(idxes, "tensor.extract_slice").val);
      strides.push_back(
          sampleTypedValueFrom(idxes, "tensor.extract_slice").val);
    }

    auto destTy = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
        srcTy.getRank() - 1, srcTy, sizes, offsets, strides);
    auto op = builder.create<tensor::ExtractSliceOp>(loc, destTy, src.val,
                                                     offsets, sizes, strides);
    return op.getOperation();
  };
}

OpGenerator tensorFromElementsGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto elemTy = randomIntOrFloatType(builder.getContext());
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat},
        [&](TypeValue tVal) { return tVal.type == elemTy; });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateTypedValue(builder, loc, elemTy));
    }
    auto tensorTy = randomStaticShapedTensorType(elemTy);
    int elemNum = 1;
    for (int i = 0; i < tensorTy.getRank(); ++i) {
      elemNum = elemNum * tensorTy.getDimSize(i);
    }
    SmallVector<Value> elements;
    for (int i = 0; i < elemNum; ++i) {
      elements.push_back(
          sampleTypedValueFrom(candidates, "tensor.from_elements").val);
    }
    auto res = builder.create<tensor::FromElementsOp>(loc, tensorTy,
                                                      ValueRange(elements));
    region.pool.addStaticShapedTensor(TypeValue(tensorTy, res),
                                      "tensor.from_elements");
    return res.getOperation();
  };
}

// OpGenerator tensorGatherGenerator() {
// }

OpGenerator tensorGenerateGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto opFilter = OpNameFilter(opsForTensorGenerate);
    auto dynTenTy = randomDynamicShapedTensorType(builder.getContext());
    auto idxCandidates = parent.pool.getCandidatesFromIndex(builder, loc);
    SmallVector<Value> dynamicSizes;
    for (int i = 0; i < dynTenTy.getNumDynamicDims(); ++i) {
      dynamicSizes.push_back(
          sampleTypedValueFrom(idxCandidates, "tensor.generate").val);
    }

    auto res = builder.create<tensor::GenerateOp>(
        loc, dynTenTy, dynamicSizes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto region = OpRegion("tensor.generate", parent.depth + 1, parent.cur_child);
          region.pool.merge(parent.pool);
          for (auto val : args) {
            region.pool.addIndex(TypeValue(b.getIndexType(), val),
                                 "args(tensor.generate)");
          }
          RegionGen regionGen(&region, {opFilter});
          regionGen.apply(builder, loc, 4);

          auto retCandidates = region.pool.searchCandidatesFrom(
              {PoolType::IntOrFloat}, [&](TypeValue typeValue) {
                return typeValue.type == dynTenTy.getElementType();
              });
          if (retCandidates.empty()) {
            retCandidates.push_back(region.pool.generateElement(
                builder, loc, dynTenTy.getElementType()));
          }
          auto ret = sampleTypedValueFrom(retCandidates, "tensor.yield");
          builder.create<tensor::YieldOp>(loc, ret.val);
        });

    parent.pool.addRankedTensor(TypeValue(dynTenTy, res), "tensor.generate");
    return res.getOperation();
  };
}

OpGenerator tensorInsertGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedTensor, PoolType::StaticShapedTensor},
        [&](TypeValue) { return true; });
    if (candidates.empty()) {
      candidates.push_back(region.pool.generateDynamicShapedTensor(
          builder, loc,
          UR(2) ? randomStaticShapedTensorType(builder.getContext())
                : randomDynamicShapedTensorType(builder.getContext())));
    }
    auto source = sampleTypedValueFrom(candidates, "tensor.insert");
    auto srcTy = source.type.dyn_cast<ShapedType>();
    auto elemTy = srcTy.getElementType();
    auto elemCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Index},
        [&](TypeValue tval) { return tval.type == elemTy; });
    if (elemCandidates.empty()) {
      elemCandidates.push_back(
          region.pool.generateTypedValue(builder, loc, elemTy));
    }
    auto elem = sampleTypedValueFrom(elemCandidates, "tensor.insert");
    auto indices = region.pool.randomIndicesForShapedType(srcTy, builder, loc);
    return builder.create<tensor::InsertOp>(loc, elem.val, source.val, indices)
        .getOperation();
  };
}

OpGenerator tensorInsertSliceGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto destCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return tval.type.dyn_cast<RankedTensorType>().getRank() > 0;
        });
    auto dest = sampleTypedValueFrom(destCandidates, "tensor.insert_slice");
    auto destTy = dest.val.getType().dyn_cast<RankedTensorType>();

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    auto idxes = region.pool.getCandidatesFromIndex(builder, loc);
    for (int i = 0; i < destTy.getRank(); ++i) {
      offsets.push_back(sampleTypedValueFrom(idxes, "tensor.insert_slice").val);
      sizes.push_back(sampleTypedValueFrom(idxes, "tensor.insert_slice").val);
      strides.push_back(sampleTypedValueFrom(idxes, "tensor.insert_slice").val);
    }

    auto srcTy = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
        destTy.getRank() - 1, destTy, sizes, offsets, strides);
    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        typeEquivalentFilter(srcTy));
    if (srcCandidates.empty()) {
      srcCandidates.push_back(
          region.pool.generateRankedTensor(builder, loc, srcTy));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "tensor.insert_slice");

    return builder
        .create<tensor::InsertSliceOp>(loc, src.val, dest.val, offsets, sizes,
                                       strides)
        .getOperation();
  };
}

//OpGenerator tensorPackGenerator() {
//  return [](OpBuilder &builder, Location loc, OpRegion &region) {
//    auto candidates = region.pool.getCandidatesFromStaticShapedTensor(
//        builder, loc, randomStaticShapedTensorType(builder.getContext()));
//    auto src = sampleTypedValueFrom(candidates, "tensor.pack");
//    auto srcTy = src.type.dyn_cast<TensorType>();
//    auto padCandidates = region.pool.searchCandidatesFrom(
//        {PoolType::IntOrFloat, PoolType::Index},
//        [&](TypeValue tVal) { return tVal.type == srcTy.getElementType(); });
//    if (padCandidates.empty()) {
//      padCandidates.push_back(
//          region.pool.generateElement(builder, loc, srcTy.getElementType()));
//    }
//    auto pad = sampleTypedValueFrom(padCandidates, "tensor.pack");
//
//    SmallVector<int64_t> innerDimsPos;
//    SmallVector<int64_t> innerTileInts;
//    SmallVector<int64_t> outerDimsPos;
//    SmallVector<int64_t> newShape;
//    for (int i = 0; i < srcTy.getRank(); ++i) {
//      innerDimsPos.push_back(i);
//      if (srcTy.getDimSize(i) > 1) {
//        innerTileInts.push_back(2);
//        newShape.push_back(2);
//      } else {
//        innerTileInts.push_back(1);
//        newShape.push_back(1);
//      }
//      outerDimsPos.push_back(i);
//    }
//    SmallVector<OpFoldResult> innerTiles;
//    for (int i = 0; i < srcTy.getRank(); ++i) {
//      newShape.insert(newShape.begin() + i,
//                      (int)ceil(srcTy.getDimSize(i) * 1.0 / innerTileInts[i]));
//      innerTiles.push_back(region.pool.constantIndices[innerTileInts[i]]);
//    }
//    auto destTy = RankedTensorType::get(newShape, srcTy.getElementType());
//    auto dest = region.pool.generateStaticShapedTensor(builder, loc, destTy);
//
//    return builder
//        .create<tensor::PackOp>(loc, src.val, dest.val, innerDimsPos,
//                                innerTiles, llvm::cl::Optional<Value>(pad.val),
//                                outerDimsPos)
//        .getOperation();
//  };
//}

OpGenerator tensorRankGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue) { return true; });
    if (candidates.empty()) {
      candidates.push_back(region.pool.generateStaticShapedTensor(
          builder, loc, randomStaticShapedTensorType(builder.getContext())));
    }
    auto source = sampleTypedValueFrom(candidates, "tensor.rank");
    auto res = builder.create<tensor::RankOp>(loc, source.val);
    region.pool.addIndex(TypeValue(builder.getIndexType(), res), "tensor.rank");
    return res.getOperation();
  };
}

OpGenerator tensorScatterGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(builder.getContext()));
    auto dest = sampleTypedValueFrom(candidates, "tensor.scatter");
    auto destTy = dest.type.dyn_cast<RankedTensorType>();

    SmallVector<int64_t> srcShape;
    SmallVector<int64_t> indicesShape;
    auto dim0 = dimPool[UR(dimPool.size())];
    srcShape.push_back(dim0);
    indicesShape.push_back(dim0);
    indicesShape.push_back(destTy.getRank());
    SmallVector<int64_t> scatterDims;
    for (int i = 0; i < destTy.getRank(); ++i) {
      srcShape.push_back(1);
      scatterDims.push_back(i);
    }

    auto srcTy = RankedTensorType::get(srcShape, destTy.getElementType());
    auto indicesTy =
        RankedTensorType::get(indicesShape, builder.getIndexType());

    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor},
        [&](TypeValue tval) { return tval.type == srcTy; });
    if (srcCandidates.empty()) {
      srcCandidates.push_back(
          region.pool.generateStaticShapedTensor(builder, loc, srcTy));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "tensor.scatter");

    auto indicesCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor},
        [&](TypeValue tval) { return tval.type == indicesTy; });
    if (indicesCandidates.empty()) {
      indicesCandidates.push_back(
          region.pool.generateStaticShapedTensor(builder, loc, indicesTy));
    }
    auto indices = sampleTypedValueFrom(indicesCandidates, "tensor.scatter");

    auto res = builder.create<tensor::ScatterOp>(
        loc, destTy, src.val, dest.val, indices.val, scatterDims, true);
    region.pool.addRankedTensor(TypeValue(res.getType(), res),
                                "tensor.scatter");
    return res.getOperation();
  };
}

OpGenerator tensorSplatGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.getCandidatesFromIntOrFloats(
        builder, loc, randomIntOrFloatType(builder.getContext()));
    auto elem = sampleTypedValueFrom(candidates, "tensor.splat");
    auto tensorTy = randomStaticShapedTensorType(elem.type);

    auto res = builder.create<tensor::SplatOp>(loc, elem.val, tensorTy);
    region.pool.addStaticShapedTensor(TypeValue(tensorTy, res), "tensor.splat");
    return res.getOperation();
  };
}

OpGenerator tensorUnpackGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor}, [&](TypeValue tVal) {
          return tVal.type.dyn_cast<RankedTensorType>().getRank() % 2 == 0;
        });
    if (candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(2);
      shape.push_back(2);
      auto tensorTy = RankedTensorType::get(
          shape, randomIntOrFloatType(builder.getContext()));
      candidates.push_back(
          region.pool.generateStaticShapedTensor(builder, loc, tensorTy));
    }
    auto source = sampleTypedValueFrom(candidates, "tensor.unpack");
    auto srcTy = source.type.dyn_cast<RankedTensorType>();

    SmallVector<int64_t> newShape;
    SmallVector<int64_t> innerDimsPos;
    SmallVector<OpFoldResult> innerTiles;
    SmallVector<int64_t> outerDimsPerm;
    for (int i = 0; i < srcTy.getRank() / 2; ++i) {

      newShape.push_back(srcTy.getDimSize(i) *
                         srcTy.getDimSize(i + srcTy.getRank() / 2));
      innerDimsPos.push_back(i);
      auto dimSize = srcTy.getDimSize(i + srcTy.getRank() / 2);
      innerTiles.push_back(region.pool.getConstantIndex(builder, loc, dimSize));
      outerDimsPerm.push_back(i);
    }
    auto destTy = RankedTensorType::get(newShape, srcTy.getElementType());
    auto destCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor},
        [&](TypeValue tVal) { return tVal.type == destTy; });
    if (destCandidates.empty()) {
      destCandidates.push_back(
          region.pool.generateStaticShapedTensor(builder, loc, destTy));
    }
    auto dest = sampleTypedValueFrom(destCandidates, "tensor.unpack");
    return builder
        .create<tensor::UnPackOp>(loc, source.val, dest.val, innerDimsPos,
                                  innerTiles, outerDimsPerm)
        .getOperation();
  };
}