//
// Created by Stan Wang on 2022/11/27.
//

#include "smith/RegionGeneration.h"
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"
#include "llvm/Support/Casting.h"

using namespace affine;

OpGenerator affineApplyGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // map should have exactly one result
    auto mapCandidates = region.pool.affineMapPool;
    if (mapCandidates.empty()) {
      std::cout << "empty map candidates! have you init the map pool?\n";
      exit(-1);
    }
    AffineMap map = mapCandidates[UR(mapCandidates.size())];
    SmallVector<Value> operands;
    for (unsigned i = 0; i < map.getNumInputs(); ++i) {
      auto indexI = sampleTypedValueFrom(
          region.pool.getCandidatesFromIndex(builder, loc), "affine.apply");
      operands.push_back(indexI.val);
    }
    auto ret = builder.create<AffineApplyOp>(loc, map, operands);
    auto tVal = TypeValue(builder.getIndexType(), ret);
    region.pool.addIndex(tVal, "affine.apply");
    return ret.getOperation();
  };
}

OpGenerator affineForGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto opFilter = OpNameFilter(opsForAffineFor);
    // Zero successors.
    auto pool = parent.pool;
    int64_t lowerBound = 0;
    int64_t upperBound = UR(128);
    int64_t step = 1;

    bool hasIterArg = UR(2);
    auto iterArgCandidates = pool.searchCandidatesFrom(allTypes, emptyFilter());
    if (iterArgCandidates.empty()) {
      iterArgCandidates.push_back(pool.generateVector(
          builder, loc, randomVectorType(builder.getContext())));
    }
    auto iterArg = sampleTypedValueFrom(iterArgCandidates, "affine.for");
    auto iterArgVal = iterArg.val;

    auto blockBuilder = [&](OpBuilder &b, Location loc,
                            Value iv /*loop iterator*/, ValueRange args) {
      auto region = OpRegion("affine.for", parent.depth + 1, parent.cur_child);
      region.pool.merge(parent.pool);
      region.pool.addIndex(TypeValue(b.getIndexType(), iv), "iter(affine.for)");
      if (hasIterArg) {
        region.pool.addTypeValue(TypeValue(iterArg.type, args.front()),
                                 "arg(affine.for)");
      }

      auto regionGen = RegionGen(&region, {opFilter});
      regionGen.applyAll(builder, loc);
      if (hasIterArg) {
        auto retCandidates = searchTypedValueFrom(region.pool, iterArg.type);
        if (retCandidates.empty()) {
          retCandidates.push_back(
              region.pool.generateTypedValue(builder, loc, iterArg.type));
        }
        auto ret = sampleTypedValueFrom(retCandidates, "scf.yield");
        builder.create<AffineYieldOp>(loc, ret.val);
      } else {
        builder.create<AffineYieldOp>(loc);
      }
    };
    if (hasIterArg) {
      auto res = builder.create<AffineForOp>(loc, lowerBound, upperBound, step,
                                             llvm::makeArrayRef(iterArgVal),
                                             blockBuilder);

      for (auto val : res->getResults()) {
        auto tVal = TypeValue(val.getType(), val);
        pool.addTypeValue(tVal, "affine.for");
      }
      return res.getOperation();
    } else {
      return builder
          .create<AffineForOp>(loc, lowerBound, upperBound, step, std::nullopt,
                               blockBuilder)
          .getOperation();
    }
  };
}

OpGenerator affineIfGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto pool = parent.pool;
    if (pool.integerSetPool.empty()) {
      std::cout << "empty set candidates! have you init the set pool?\n";
      exit(-1);
    }
    auto set = pool.integerSetPool[UR(pool.integerSetPool.size())];
    SmallVector<Value> args;
    //    auto candidates = pool.getCandidatesFromIndex(builder, loc);
    for (uint32_t i = 0; i < set.getNumInputs(); ++i) {
      auto idx = UR(idx_ub);
      auto val = pool.constantIndices[idx];
      args.push_back(val);
    }

    // TODO - error if do not have result.
    bool hasResult = true;
    SmallVector<Type> resultTypes;
    if (hasResult) {
      resultTypes.push_back(randomNonTensorType(builder.getContext()));
    }
    bool withElseRegion = hasResult || UR(2);
    auto ifOp =
        builder.create<AffineIfOp>(loc, resultTypes, set, args, withElseRegion);

    auto point = builder.saveInsertionPoint();

    auto opFilter = OpNameFilter(opsForAffineIf);
    auto blockBuilder = [&]() {
      OpRegion region("affine.if", parent.depth + 1, parent.cur_child);
      region.pool.merge(parent.pool);
      auto regionGen = RegionGen(&region, {opFilter});
      regionGen.apply(builder, loc, 8);
      if (hasResult) {
        auto resultType = resultTypes.front();
        auto candidates = searchTypedValueFrom(region.pool, resultType);
        if (candidates.empty()) {
          candidates.push_back(
              region.pool.generateTypedValue(builder, loc, resultType));
        }
        auto ret = sampleTypedValueFrom(candidates, "affine.yield");
        builder.create<AffineYieldOp>(loc, ValueRange(ret.val));
      }
    };

    // then block
    {
      Block *block = &ifOp.getThenRegion().front();
      block->getOperations().clear();
      if (hasResult) {
        builder.setInsertionPointToEnd(block);
      } else {
        builder.setInsertionPointToStart(block);
      }
      blockBuilder();
      builder.restoreInsertionPoint(point);
    }

    // else block
    if (withElseRegion) {
      Block *block = &ifOp.getElseRegion().front();
      if (hasResult) {
        builder.setInsertionPointToEnd(block);
      } else {
        builder.setInsertionPointToStart(block);
      }
      blockBuilder();

      builder.restoreInsertionPoint(point);
    }
    return ifOp.getOperation();
  };
}

// affine load and store should appear in AffineScope ops.
OpGenerator affineLoadGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto memCandidates = parent.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref},
        emptyFilter());
    if (memCandidates.empty()) {
      memCandidates.push_back(parent.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memCandidates, "affine.load");
    auto shapedType = memref.type.dyn_cast<ShapedType>();
    auto shape = shapedType.getShape();
    auto dim = shape.size();
    auto elemTy = shapedType.getElementType();
    SmallVector<Value> indices;
    for (uint32_t i = 0; i < dim; ++i) {
      int size = parent.pool.constantIndices.size();
      auto idx = UR(size);
      indices.push_back(parent.pool.getConstantIndex(builder, loc, idx));
    }
    assert(memref.val);
    auto res =
        builder.create<AffineLoadOp>(loc, memref.val, ValueRange(indices));
    parent.pool.addElement(TypeValue(elemTy, res), "affine.load");
    return res.getOperation();
  };
}

OpGenerator affineStoreGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto memCandidates = parent.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref},
        emptyFilter());
    if (memCandidates.empty()) {
      memCandidates.push_back(parent.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memCandidates, "affine.store");
    auto shapedType = memref.type.dyn_cast<ShapedType>();
    auto shape = shapedType.getShape();
    auto dim = shape.size();
    auto elemTy = shapedType.getElementType();
    auto candidates = parent.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat},
        [&](TypeValue tval) { return tval.type == elemTy; });
    if (candidates.empty()) {
      candidates.push_back(
          parent.pool.generateTypedValue(builder, loc, elemTy));
    }
    auto elem2Store = sampleTypedValueFrom(candidates, "affine.store");
    SmallVector<Value> indices;
    for (uint32_t i = 0; i < dim; ++i) {
      int size = parent.pool.constantIndices.size();
      indices.push_back(parent.pool.constantIndices[UR(size)]);
    }
    return builder
        .create<AffineStoreOp>(loc, elem2Store.val, memref.val,
                               ValueRange(indices))
        .getOperation();
  };
}

OpGenerator affineMaxGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    SmallVector<Value> mapOperands;
    auto map = parent.pool.affineMapPool[UR(parent.pool.affineMapPool.size())];
    auto candidates = parent.pool.getCandidatesFromIndex(builder, loc);
    auto dimCount = map.getNumDims();
    auto symbolCount = map.getNumSymbols();
    for (int i = 0; i < dimCount + symbolCount; ++i) {
      mapOperands.push_back(sampleTypedValueFrom(candidates, "affine.max").val);
    }
    auto index = builder.create<AffineMaxOp>(loc, map, mapOperands);
    if (index->getResultTypes().size() == 1 &&
        index->getResultTypes()[0].dyn_cast<IndexType>()) {
      auto tVal = TypeValue(builder.getIndexType(), index);
      parent.pool.addIndex(tVal, "affine.max");
    }
    return index.getOperation();
  };
}

OpGenerator affineMinGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    SmallVector<Value> mapOperands;
    auto map = parent.pool.affineMapPool[UR(parent.pool.affineMapPool.size())];
    auto candidates = parent.pool.getCandidatesFromIndex(builder, loc);
    auto dimCount = map.getNumDims();
    auto symbolCount = map.getNumSymbols();
    for (int i = 0; i < dimCount + symbolCount; ++i) {
      mapOperands.push_back(sampleTypedValueFrom(candidates, "affine.max").val);
    }
    auto index = builder.create<AffineMinOp>(loc, map, mapOperands);
    if (index->getResultTypes().size() == 1 &&
        index->getResultTypes()[0].dyn_cast<IndexType>()) {
      auto tVal = TypeValue(builder.getIndexType(), index);
      parent.pool.addIndex(tVal, "affine.min");
    }
    return index.getOperation();
  };
}

OpGenerator affineParallelGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto elemTy = randomIntOrFloatType(builder.getContext());
    SmallVector<int64_t> ranges; // also the resShape
    auto rangeDim = UR(rank_ub) + 1;
    for (int i = 0; i < rangeDim; ++i) {
      ranges.push_back(dimPool[UR(dimPool.size())]);
    }
    bool hasResult = UR(2);
    SmallVector<Type> resultTypes;
    if (hasResult) {
      resultTypes.push_back(MemRefType::get(ranges, elemTy));
    }
    SmallVector<arith::AtomicRMWKind> reductions;
    for (int i = 0; i < resultTypes.size(); ++i) {
      if (elemTy.isa<IntegerType>()) {
        reductions.push_back(intRmwKinds[UR(intRmwKinds.size())]);
      } else {
        reductions.push_back(floatRmwKinds[UR(floatRmwKinds.size())]);
      }
    }
    auto op =
        builder.create<AffineParallelOp>(loc, resultTypes, reductions, ranges);
    for (uint i = 0; i < resultTypes.size(); i++) {
      auto ty = resultTypes[i];
      auto val = op.getResult(i);
      auto tVal = TypeValue(ty, val);
      parent.pool.addTypeValue(tVal, "res(affine.parallel)");
    }
    auto point = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(op.getBody());
    {
      OpRegion region("affine.parallel", parent.depth + 1,  parent.cur_child);
      region.pool.merge(parent.pool);
      auto opFilter = OpNameFilter(opsForAffineParallel);
      RegionGen regionGen(&region, {opFilter});
      regionGen.apply(builder, loc, 0);
      if (!resultTypes.empty()) {
        SmallVector<Value> rets;
        for (auto resultTy : resultTypes) {
          auto retCandidates = region.pool.searchCandidatesFrom(
              {
                  PoolType::IntOrFloat,
              },
              typeEquivalentFilter(elemTy));
          if (retCandidates.empty()) {
            retCandidates.push_back(
                region.pool.generateTypedValue(builder, loc, resultTy));
          }
          auto ret = sampleTypedValueFrom(retCandidates, "affine.yield").val;
          rets.push_back(ret);
        }
        builder.create<AffineYieldOp>(loc, ValueRange(rets));
      }
    }
    builder.restoreInsertionPoint(point);
    return op.getOperation();
  };
}

//OpGenerator affinePrefetchGenerator() {
//  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
//    auto memCandidates = parent.pool.getCandidatesFromStaticShapedMemref(
//        builder, loc, randomStaticShapedMemrefType(builder.getContext()));
//    auto memref = sampleTypedValueFrom(memCandidates, "affine.prefetch");
//    auto shapedType = memref.type.dyn_cast<ShapedType>();
//    auto shape = shapedType.getShape();
//    SmallVector<Value> indices;
//    //    auto dim = shape.size();
//    //    for (uint32_t i = 0; i < dim; ++i) {
//    //      int size = parent.pool.constantIndices.size();
//    //      indices.push_back(parent.pool.constantIndices[UR(size)]);
//    //    }
//    // if there is no map, no need to add indices..
//    auto op = builder.create<AffinePrefetchOp>(loc, memref.val, indices, UR(2),
//                                               UR(4), UR(2));
//    return op.getOperation();
//  };
//}

OpGenerator affineVectorLoadGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto memCandidates = parent.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref},
        emptyFilter());
    if (memCandidates.empty()) {
      memCandidates.push_back(parent.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memCandidates, "affine.vector_load");
    auto idxCandidates = parent.pool.searchCandidatesFrom(
        {PoolType::Index}, [](TypeValue tval) { return isValidDim(tval.val); });
    assert(!idxCandidates.empty());
    SmallVector<Value> indices;
    auto memTy = memref.type.dyn_cast<MemRefType>();
    for (int i = 0; i < memTy.getRank(); ++i) {
      indices.push_back(
          sampleTypedValueFrom(idxCandidates, "affine.vector_load").val);
    }
    auto vecTy = random1DVectorType(memTy.getElementType());
    auto op = builder.create<AffineVectorLoadOp>(loc, vecTy, memref.val,
                                                 ValueRange(indices));
    parent.pool.addVector(TypeValue(op.getVectorType(), op),
                          "affine.vector_load");
    return op.getOperation();
  };
}

OpGenerator affineVectorStoreGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto vecCandidates = parent.pool.searchCandidatesFrom(
        {PoolType::Vector}, [](TypeValue tval) {
          return tval.type.dyn_cast<VectorType>().getRank() == 1;
        });
    if (vecCandidates.empty()) {
      vecCandidates.push_back(parent.pool.generateVector(
          builder, loc, random1DVectorType(builder.getContext())));
    }
    auto vec2Store = sampleTypedValueFrom(vecCandidates, "affine.vector_store");
    auto vecTy = vec2Store.type.dyn_cast<VectorType>();

    auto memCandidates = parent.pool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref},
        [&](TypeValue tval) {
          return tval.type.dyn_cast<ShapedType>().getElementType() ==
                 vecTy.getElementType();
        });
    if (memCandidates.empty()) {
      memCandidates.push_back(parent.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(vecTy.getElementType())));
    }
    auto memref = sampleTypedValueFrom(memCandidates, "affine.vector_store");
    auto idxCandidates = parent.pool.searchCandidatesFrom(
        {PoolType::Index}, [](TypeValue tval) { return isValidDim(tval.val); });
    assert(isValidDim(parent.pool.constantIndices[0]));
    assert(!idxCandidates.empty());
    SmallVector<Value> indices;
    auto memTy = memref.type.dyn_cast<MemRefType>();
    for (int i = 0; i < memTy.getRank(); ++i) {
      indices.push_back(
          sampleTypedValueFrom(idxCandidates, "affine.vector_store").val);
    }
    auto op = builder.create<AffineVectorStoreOp>(
        loc, vec2Store.val, memref.val, ValueRange(indices));
    return op.getOperation();
  };
}