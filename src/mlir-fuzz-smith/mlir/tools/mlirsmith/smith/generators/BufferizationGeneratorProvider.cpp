//
// Created by Stan Wang on 2023/2/9.
//
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"

OpGenerator bufferizationAllocTensorGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    RankedTensorType type;
    bool isDynamic = UR(2);
    int dyDims = 0;
    SmallVector<Value> dyDimVals;
    if (isDynamic) {
      std::vector<int64_t> shape;
      auto rank = UR(rank_ub + 1); // 0-3
      auto indexCandidates = region.pool.getCandidatesFromIndex(builder, loc);
      for (int i = 0; i < rank; ++i) {
        if (UR(2)) {
          dyDimVals.push_back(sampleTypedValueFrom(indexCandidates,
                                                   "bufferization.alloc_tensor")
                                  .val);
          shape.push_back(ShapedType::kDynamic);
        } else {
          shape.push_back(dimPool[UR(dimPool.size())]);
        }
      }
      type = RankedTensorType::get(shape,
                                   randomIntOrFloatType(builder.getContext()));
    } else {
      auto shape = shapePool[UR(shapePool.size())];
      type = RankedTensorType::get(shape,
                                   randomIntOrFloatType(builder.getContext()));
    }
    auto val = builder.create<bufferization::AllocTensorOp>(
        loc, type, ValueRange(dyDimVals));
    if (isDynamic) {
      region.pool.addStaticShapedTensor(TypeValue(type, val),
                                        "bufferization.alloc_tensor");
    } else {
      region.pool.addDynamicShapedTensor(TypeValue(type, val),
                                         "bufferization.alloc_tensor");
    }
    return val.getOperation();
  };
}

OpGenerator bufferizationCloneGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto memrefCandidates = region.pool.getCandidatesFromStaticShapedMemref(
        builder, loc, randomStaticShapedMemrefType(builder.getContext()));
    auto mem = sampleTypedValueFrom(memrefCandidates, "bufferization.clone");
    auto op = builder.create<bufferization::CloneOp>(loc, mem.val);
    region.pool.addStaticShapedMemref(TypeValue(mem.type, op),
                                      "bufferization.clone");
    return op.getOperation();
  };
}

OpGenerator bufferizationDeallocGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto tensorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (tensorCandidates.empty()) {
      tensorCandidates.push_back(region.pool.generateStaticShapedTensor(
          builder, loc, randomStaticShapedTensorType(builder.getContext())));
    }
    auto tensor =
        sampleTypedValueFrom(tensorCandidates, "bufferization.dealloc");
    return builder.create<bufferization::DeallocTensorOp>(loc, tensor.val)
        .getOperation();
  };
}

OpGenerator bufferizationToMemrefGenerator() {
  // Mark，Mutate the result of bufferization.to_memref will cause undefined
  // behaviour
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto tensorCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (tensorCandidates.empty()) {
      tensorCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto tensor =
        sampleTypedValueFrom(tensorCandidates, "bufferization.to_memref");
    auto shapedTy = tensor.type.dyn_cast<ShapedType>();
    auto resultMemTy =
        MemRefType::get(shapedTy.getShape(), shapedTy.getElementType());
    auto op =
        builder.create<bufferization::ToMemrefOp>(loc, resultMemTy, tensor.val);
    region.pool.addRankedMemref(TypeValue(resultMemTy, op),
                                "bufferization.to_memref");
    return op.getOperation();
  };
}

OpGenerator bufferizationToTensorGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto memrefCandidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref},
        emptyFilter());
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref =
        sampleTypedValueFrom(memrefCandidates, "bufferization.to_tensor");
    auto shapedTy = memref.type.dyn_cast<ShapedType>();
    auto resTensorTy =
        RankedTensorType::get(shapedTy.getShape(), shapedTy.getElementType());

    auto op =
        builder.create<bufferization::ToTensorOp>(loc, resTensorTy, memref.val);
    region.pool.addRankedTensor(TypeValue(resTensorTy, op),
                                "bufferization.to_tensor");
    return op.getOperation();
  };
}