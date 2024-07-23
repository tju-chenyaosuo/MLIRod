#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

OpGenerator tensorCastGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		const int dim_ub = 32;

		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &operandCandidates);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto src = operandCandidates[rollIdx(operandCandidates.size())];
    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
		SmallVector<int64_t> newShape;
		if (srcTy.hasStaticShape()) {
      for (int i = 0; i < srcTy.getRank(); ++i) {
        newShape.push_back(ShapedType::kDynamic);
      }
    } else {
      for (int i = 0; i < srcTy.getRank(); ++i) {
        if (srcTy.isDynamicDim(i)) {
          newShape.push_back(rollIdx(dim_ub));
        } else {
          newShape.push_back(srcTy.getDimSize(i));
        }
      }
    }
		auto destTy = RankedTensorType::get(newShape, srcTy.getElementType());
		mlir::Value res = builder.create<tensor::CastOp>(loc, destTy, src);
		mb->add2Pool(res);
		return true;
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
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!ranked_tensor_filter(s, v)) { return false; }
				return v.getType().dyn_cast<TensorType>().getRank() >= 2;
			},
			&candidates
		);
		if (candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(2);
      shape.push_back(2);
      auto type = randomIntOrFloatType(builder.getContext());
      candidates.push_back(
				generateStaticShapedTensor(builder, loc, RankedTensorType::get(shape, type)));
    }
		auto source = candidates[rollIdx(candidates.size())];
		auto srcTy = source.getType().dyn_cast<RankedTensorType>();
    SmallVector<ReassociationIndices> reIdxes;
    reIdxes.push_back({0, 1});
		for (int i = 2; i < srcTy.getRank(); ++i) {
      ReassociationIndices indices;
      indices.push_back(i);
      if (rollIdx(2) && srcTy.getRank() - i > 1) {
        indices.push_back(i + 1);
        i++;
      }
      reIdxes.push_back(indices);
    }

		mlir::Value res = builder.create<tensor::CollapseShapeOp>(loc, source, reIdxes);
    auto resultType = computeTensorReshapeCollapsedType(
			srcTy, getSymbolLessAffineMaps(
				convertReassociationIndicesToExprs(builder.getContext(), reIdxes)));
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator tensorDimGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!ranked_tensor_filter(s, v)) { return false; }
				return v.getType().dyn_cast<RankedTensorType>().getRank() > 0;
			},
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto source = candidates[rollIdx(candidates.size())];
		auto srcTy = source.getType().dyn_cast<RankedTensorType>();
		auto num = rollIdx(srcTy.getRank());
		auto idx = generateIndex(builder, loc, num);
		assert(llvm::detail::isPresent(idx));

    mlir::Value res = builder.create<tensor::DimOp>(loc, source, idx);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator tensorEmptyGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    if (rollIdx(2)) {
      auto tensorTy = randomStaticShapedTensorType(builder.getContext())
                          .dyn_cast<RankedTensorType>();
      mlir::Value res = builder.create<tensor::EmptyOp>(loc, tensorTy.getShape(),
                                                 tensorTy.getElementType());
			mb->add2Pool(res);
			return true;
    } else {
			auto shape = getRandomShape();
      SmallVector<Value> dynamicSizes;
			std::vector<mlir::Value> dynamicSizesCandidates = std::vector<mlir::Value>();
			mb->search(index_filter, &dynamicSizesCandidates);
      if (dynamicSizesCandidates.empty()) {
        dynamicSizesCandidates.push_back(generateIndex(builder, loc, rollIdx(100)));
      }
      for (size_t i = 0; i < shape.size(); ++i) {
        if (rollIdx(2)) {
          shape[i] = ShapedType::kDynamic;
					dynamicSizes.push_back(dynamicSizesCandidates[rollIdx(dynamicSizesCandidates.size())]);
        }
      }
      auto tensorTy = RankedTensorType::get(shape, randomIntOrFloatType(builder.getContext()));
      builder.create<tensor::EmptyOp>(loc, tensorTy, ValueRange(dynamicSizes));
    }
    return true;
  };
}

OpGenerator tensorExpandShapeGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!ranked_tensor_filter(s, v)) { return false; }
				auto tensorTy = v.getType().dyn_cast<TensorType>();
				return tensorTy.getRank() > 0 && !tensorTy.isDynamicDim(tensorTy.getRank() - 1);
			},
			&candidates
		);
    if (candidates.empty()) {
      candidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto source = candidates[rollIdx(candidates.size())];
		auto srcTy = source.getType().dyn_cast<RankedTensorType>();

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
    mlir::Value res = builder.create<tensor::ExpandShapeOp>(
			loc, resTy, source, reassociation, SmallVector<NamedAttribute>());
    mb->add2Pool(res);
		return true;
  };
}

OpGenerator tensorExtractGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &candidates);
		if (candidates.empty()) {
      candidates.push_back(
				generateDynamicShapedTensor(
					builder, 
					loc, 
					rollIdx(2) ? 
					randomStaticShapedTensorType(builder.getContext()) : 
					randomDynamicShapedTensorType(builder.getContext()),
          mb));
    }
		auto source = candidates[rollIdx(candidates.size())];
		auto srcTy = source.getType().dyn_cast<ShapedType>();
		auto indices = randomIndicesForShapedType(srcTy, builder, loc);
		mlir::Value res = builder.create<tensor::ExtractOp>(loc, source, indices);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator tensorExtractSliceGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!ranked_tensor_filter(s, v)) { return false; }
				return v.getType().dyn_cast<RankedTensorType>().getRank() > 0;
			},
			&srcCandidates
		);
		if (srcCandidates.empty()) {
      srcCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto src = srcCandidates[rollIdx(srcCandidates.size())];
		auto srcTy = src.getType().dyn_cast<RankedTensorType>();

		SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

		std::vector<mlir::Value> idxes = std::vector<mlir::Value>();
		mb->search(index_filter, &idxes);
		if (idxes.empty()) {
			idxes.push_back(generateIndex(builder, loc, 1));
		}
		for (int i = 0; i < srcTy.getRank(); ++i) {
			offsets.push_back(idxes[rollIdx(idxes.size())]);
			sizes.push_back(idxes[rollIdx(idxes.size())]);
			strides.push_back(idxes[rollIdx(idxes.size())]);
		}

		auto destTy = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
        srcTy.getRank() - 1, srcTy, sizes, offsets, strides);
    builder.create<tensor::ExtractSliceOp>(loc, destTy, src, offsets, sizes, strides);
    return true;
  };
}

OpGenerator tensorFromElementsGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    auto elemTy = randomIntOrFloatType(builder.getContext());
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[&] (std::string s, mlir::Value v) -> bool {
				if (!int_float_filter(s)) { return false; }
				return v.getType() == elemTy;
			},
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateTypedValue(builder, loc, elemTy, mb));
    }
		auto tensorTy = randomStaticShapedTensorType(elemTy);
		int elemNum = 1;
    for (int i = 0; i < tensorTy.getRank(); ++i) {
      elemNum = elemNum * tensorTy.getDimSize(i);
    }
		SmallVector<Value> elements;
    for (int i = 0; i < elemNum; ++i) {
      elements.push_back(candidates[rollIdx(candidates.size())]);
    }
		mlir::Value res = builder.create<tensor::FromElementsOp>(loc, tensorTy, ValueRange(elements));
    mb->add2Pool(res);
		return true;
  };
}

OpGenerator tensorGenerateGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto dynTenTy = randomDynamicShapedTensorType(builder.getContext());
    std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
    mb->search(index_filter, &idxCandidates);
    if (idxCandidates.empty()) {
      idxCandidates.push_back(generateIndex(builder, loc, 1));
    }
    SmallVector<Value> dynamicSizes;
    for (int i = 0; i < dynTenTy.getNumDynamicDims(); ++i) {
      dynamicSizes.push_back(idxCandidates[rollIdx(idxCandidates.size())]);
    }

    mlir::Value res = builder.create<tensor::GenerateOp>(loc, dynTenTy, dynamicSizes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        MutationBlock* childMb = new MutationBlock(*mb);
        for (mlir::Value val : args) {
          childMb->add2Pool(val);
        }
        unsigned statementNum = rollIdx(4);
        for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
          OpGenerator mutator = generators[rollIdx(generators.size())];
          mutator(builder, loc, childMb);
        }
        std::vector<mlir::Value> retCandidates = std::vector<mlir::Value>();
        std::string dynTenTyStr = getValueTypeStr(dynTenTy);
        if (childMb->valuePool.pool.count(dynTenTyStr)) {
          retCandidates.insert(
            retCandidates.end(), 
            childMb->valuePool.pool[dynTenTyStr].begin(), 
            childMb->valuePool.pool[dynTenTyStr].end());
        }
        if (retCandidates.empty()) {
          retCandidates.push_back(generateElement(builder, loc, dynTenTy.getElementType()));
        }
        auto ret = retCandidates[rollIdx(retCandidates.size())];
        builder.create<tensor::YieldOp>(loc, ret);
      });
    
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator tensorInsertGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &candidates);
    if (candidates.empty()) {
      candidates.push_back(generateDynamicShapedTensor(
          builder, loc,
          rollIdx(2) ? randomStaticShapedTensorType(builder.getContext())
                : randomDynamicShapedTensorType(builder.getContext()),
                mb));
    }
    auto source = candidates[rollIdx(candidates.size())];
    auto srcTy = source.getType().dyn_cast<ShapedType>();
    auto elemTy = srcTy.getElementType();
    std::string elemTyStr = getValueTypeStr(elemTy);
    std::vector<mlir::Value> elemCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(elemTyStr)) {
      elemCandidates.insert(
        elemCandidates.begin(), 
        mb->valuePool.pool[elemTyStr].begin(), 
        mb->valuePool.pool[elemTyStr].end()
      );
    }
    if (elemCandidates.empty()) {
      elemCandidates.push_back(generateTypedValue(builder, loc, elemTy, mb));
    }
    auto elem = elemCandidates[rollIdx(elemCandidates.size())];
    auto indices = randomIndicesForShapedType(srcTy, builder, loc);
    builder.create<tensor::InsertOp>(loc, elem, source, indices);
    return true;
  };
}

OpGenerator tensorInsertSliceGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> destCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) { return false; }
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0;
      },
      &destCandidates
    );
    if (destCandidates.empty()) {
      destCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto dest = destCandidates[rollIdx(destCandidates.size())];
    auto destTy = dest.getType().dyn_cast<RankedTensorType>();

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;


    std::vector<mlir::Value> idxes = std::vector<mlir::Value>();
    mb->search(index_filter, &idxes);
    if (idxes.empty()) {
      idxes.push_back(generateIndex(builder, loc ,1));
    }
    for (int i = 0; i < destTy.getRank(); ++i) {
      offsets.push_back(idxes[rollIdx(idxes.size())]);
      sizes.push_back(idxes[rollIdx(idxes.size())]);
      strides.push_back(idxes[rollIdx(idxes.size())]);
    }

    auto srcTy = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
      destTy.getRank() - 1, destTy, sizes, offsets, strides);
    std::string srcTyStr = getValueTypeStr(srcTy);
    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(srcTyStr)) {
      srcCandidates.insert(
        srcCandidates.begin(), 
        mb->valuePool.pool[srcTyStr].begin(), 
        mb->valuePool.pool[srcTyStr].end());
    }
    if (srcCandidates.empty()) {
      srcCandidates.push_back(generateRankedTensor(builder, loc, srcTy, mb));
    }
    auto src = srcCandidates[rollIdx(srcCandidates.size())];
    builder.create<tensor::InsertSliceOp>(loc, src, dest, offsets, sizes, strides);
    return true;
  };
}

OpGenerator tensorPackGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(static_tensor_filter, &candidates);
    if (candidates.empty()) {
      candidates.push_back(
        generateStaticShapedTensor(
          builder, 
          loc, 
          randomStaticShapedTensorType(builder.getContext())));
    }
    auto src = candidates[rollIdx(candidates.size())];
    auto srcTy = src.getType().dyn_cast<TensorType>();
    std::vector<mlir::Value> padCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        bool cond1 = int_float_filter(s) || index_filter(s);
        if (!cond1) return false;
        return v.getType() == srcTy.getElementType();
      }, 
      &padCandidates
    );
    if (padCandidates.empty()) {
      padCandidates.push_back(generateElement(builder, loc, srcTy.getElementType()));
    }
    auto pad = padCandidates[rollIdx(padCandidates.size())];

    SmallVector<int64_t> innerDimsPos;
    SmallVector<int64_t> innerTileInts;
    SmallVector<int64_t> outerDimsPos;
    SmallVector<int64_t> newShape;
    for (int i = 0; i < srcTy.getRank(); ++i) {
      innerDimsPos.push_back(i);
      if (srcTy.getDimSize(i) > 1) {
        innerTileInts.push_back(2);
        newShape.push_back(2);
      } else {
        innerTileInts.push_back(1);
        newShape.push_back(1);
      }
      outerDimsPos.push_back(i);
    }
    SmallVector<OpFoldResult> innerTiles;
    for (int i = 0; i < srcTy.getRank(); ++i) {
      newShape.insert(newShape.begin() + i,
                      (int)ceil(srcTy.getDimSize(i) * 1.0 / innerTileInts[i]));

      innerTiles.push_back(generateIndex(builder, loc, innerTileInts[i]));
    }
    auto destTy = RankedTensorType::get(newShape, srcTy.getElementType());
    auto dest = generateStaticShapedTensor(builder, loc, destTy);
    builder.create<tensor::PackOp>(
      loc, src, dest, innerDimsPos, innerTiles, std::optional<Value>(pad), outerDimsPos);
    return true;
  };
}

OpGenerator tensorRankGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &candidates);
    if (candidates.empty()) {
      candidates.push_back(
        generateStaticShapedTensor(
          builder, 
          loc, 
          randomStaticShapedTensorType(builder.getContext())));
    }
    auto source = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<tensor::RankOp>(loc, source);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator tensorScatterGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(static_tensor_filter, &candidates);
    if (candidates.empty()) {
      candidates.push_back(
        generateStaticShapedTensor(
          builder, 
          loc, 
          randomStaticShapedTensorType(builder.getContext())));
    }
    auto dest = candidates[rollIdx(candidates.size())];
    auto destTy = dest.getType().dyn_cast<RankedTensorType>();

    SmallVector<int64_t> srcShape;
    SmallVector<int64_t> indicesShape;
    auto dim0 = rollIdx(dim_ub) + 1;
    srcShape.push_back(dim0);
    indicesShape.push_back(dim0);
    indicesShape.push_back(destTy.getRank());
    SmallVector<int64_t> scatterDims;
    for (int i = 0; i < destTy.getRank(); ++i) {
      srcShape.push_back(1);
      scatterDims.push_back(i);
    }

    auto srcTy = RankedTensorType::get(srcShape, destTy.getElementType());
    auto indicesTy = RankedTensorType::get(indicesShape, builder.getIndexType());

    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!static_tensor_filter(s, v)) 
          return false;
        return v.getType() == srcTy;
      }, 
      &srcCandidates
    );
    if (srcCandidates.empty()) {
      srcCandidates.push_back(generateStaticShapedTensor(builder, loc, srcTy));
    }
    auto src = srcCandidates[rollIdx(srcCandidates.size())];


    std::vector<mlir::Value> indicesCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!static_tensor_filter(s, v)) 
          return false;
        return v.getType() == indicesTy;
      },
      &indicesCandidates
    );
    if (indicesCandidates.empty()) {
      indicesCandidates.push_back(generateStaticShapedTensor(builder, loc, indicesTy));
    }
    auto indices = indicesCandidates[rollIdx(indicesCandidates.size())];

    mlir::Value res = builder.create<tensor::ScatterOp>(
      loc, destTy, src, dest, indices, scatterDims, true);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator tensorSplatGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_float_filter, &candidates);
    if (candidates.empty()) {
      candidates.push_back(
        generateElement(builder, loc, randomIntOrFloatType(builder.getContext())));
    }
    auto elem = candidates[rollIdx(candidates.size())];
    auto tensorTy = randomStaticShapedTensorType(elem.getType());
    mlir::Value res = builder.create<tensor::SplatOp>(loc, elem, tensorTy);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator tensorUnpackGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!static_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() % 2 == 0;
      },
      &candidates
    );
    if (candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(2);
      shape.push_back(2);
      auto tensorTy = RankedTensorType::get(shape, randomIntOrFloatType(builder.getContext()));
      candidates.push_back(generateStaticShapedTensor(builder, loc, tensorTy));
    }
    auto source = candidates[rollIdx(candidates.size())];
    auto srcTy = source.getType().dyn_cast<RankedTensorType>();

    SmallVector<int64_t> newShape;
    SmallVector<int64_t> innerDimsPos;
    SmallVector<OpFoldResult> innerTiles;
    SmallVector<int64_t> outerDimsPerm;
    for (int i = 0; i < srcTy.getRank() / 2; ++i) {
      newShape.push_back(srcTy.getDimSize(i) * srcTy.getDimSize(i + srcTy.getRank() / 2));
      innerDimsPos.push_back(i);
      auto dimSize = srcTy.getDimSize(i + srcTy.getRank() / 2);
      innerTiles.push_back(generateIndex(builder, loc, dimSize));
      outerDimsPerm.push_back(i);
    }
    auto destTy = RankedTensorType::get(newShape, srcTy.getElementType());

    std::vector<mlir::Value> destCandidates = std::vector<mlir::Value>();
    std::string destTyStr = getValueTypeStr(destTy);
    if (mb->valuePool.pool.count(destTyStr)) {
      destCandidates.insert(
        destCandidates.begin(), 
        mb->valuePool.pool[destTyStr].begin(), 
        mb->valuePool.pool[destTyStr].end());
    }
    if (destCandidates.empty()) {
      destCandidates.push_back(generateStaticShapedTensor(builder, loc, destTy));
    }
    auto dest = destCandidates[rollIdx(destCandidates.size())];
    builder.create<tensor::UnPackOp>(
      loc, 
      source, 
      dest, 
      innerDimsPos, 
      innerTiles, 
      outerDimsPerm);
    return true;
  };
}