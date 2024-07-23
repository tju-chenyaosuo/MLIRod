//
// Created by Stan Wang on 2022/10/28.
//

#include "smith/RegionGeneration.h"
#include "smith/TypeGeneration.h"
#include "smith/ValueGeneration.h"
#include "smith/generators/OpGeneration.h"

std::vector<uint32_t> alignmentFactors = {1, 2, 4, 8, 16};

OpGenerator memrefLoadGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto memCandidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref},
        emptyFilter());
    if (memCandidates.empty()) {
      memCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto typedAlloc = sampleTypedValueFrom(memCandidates);
    auto alloc = typedAlloc.val;
    auto t = typedAlloc.type.dyn_cast<ShapedType>();
    auto elemType = t.getElementType();
    auto loadIndices = typedValuePool.randomIndicesForShapedType(
        typedAlloc.type.dyn_cast<ShapedType>(), builder, loc);
    Value val = builder.create<memref::LoadOp>(loc, alloc, loadIndices);
    auto tVal = TypeValue(elemType, val);
    typedValuePool.addElement(tVal, "memref.load");
    return val.getDefiningOp();
  };
}

OpGenerator memrefStoreGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto memCandidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref},
        emptyFilter());
    if (memCandidates.empty()) {
      memCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto typedAlloc = sampleTypedValueFrom(memCandidates, "memref.store");
    auto alloc = typedAlloc.val;
    auto t = typedAlloc.type.dyn_cast<ShapedType>();
    auto elemType = t.getElementType();

    // Search from intOrFloatPool
    std::vector<TypeValue> candidateValueToStore;
    for (auto elem : typedValuePool.intOrFloatPool) {
      if (elem.type == elemType) {
        candidateValueToStore.push_back(elem);
      }
    }
    if (candidateValueToStore.empty()) {
      candidateValueToStore.push_back(
          region.pool.generateElement(builder, loc, elemType));
    }
    auto tValToStore =
        sampleTypedValueFrom(candidateValueToStore, "memref.store");
    auto storeIndices = typedValuePool.randomIndicesForShapedType(
        typedAlloc.type.dyn_cast<ShapedType>(), builder, loc);
    return builder
        .create<memref::StoreOp>(loc, tValToStore.val, alloc, storeIndices)
        .getOperation();
  };
}

OpGenerator atomicRMWGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;

    auto memCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref},
        [&](TypeValue tval) {
          auto elemTy = tval.type.dyn_cast<MemRefType>().getElementType();
          auto isInteger = elemTy.isa<IntegerType>();
          return elemTy.isa<FloatType>() ||
                 (isInteger && (isLLVMIRIntegerBitWidth(
                                   elemTy.dyn_cast<IntegerType>().getWidth())));
        });
    if (memCandidates.empty()) {
      memCandidates.push_back(region.pool.generateStaticShapedMemref(
          builder, loc, randomStaticShapedMemrefType(builder.getI32Type())));
    }
    auto memref = sampleTypedValueFrom(memCandidates, "memref.atomic_rmw");
    auto shapedType = memref.type.dyn_cast<ShapedType>();
    auto elemType = shapedType.getElementType();
    auto elemCandidates =
        searchTypedValueFrom(typedValuePool.intOrFloatPool, elemType);
    if (elemCandidates.empty()) {
      elemCandidates.push_back(
          typedValuePool.generateTypedValue(builder, loc, elemType));
    }
    auto elem = sampleTypedValueFrom(elemCandidates, "memref.atomic_rmw");
    arith::AtomicRMWKind kind;
    if (elemType.dyn_cast<IntegerType>()) {
      kind = intRmwKinds[UR(intRmwKinds.size())];
    } else if (elemType.dyn_cast<FloatType>()) {
      kind = floatRmwKinds[UR(floatRmwKinds.size())];
    } else {
      std::cout << "unsupported type " << std::endl;
      exit(-1);
    }
    auto loadIndices = typedValuePool.randomIndicesForShapedType(
        memref.type.dyn_cast<ShapedType>(), builder, loc);
    auto res = builder.create<memref::AtomicRMWOp>(
        loc, elemType, kind, elem.val, memref.val, ValueRange(loadIndices));
    auto tVal = TypeValue(elemType, res);
    typedValuePool.addIntOrFloat(tVal, "memref.atomic_rmw");
    return res.getOperation();
  };
}

OpGenerator memrefCopyGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto srcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref},
        emptyFilter());
    if (srcCandidates.empty()) {
      srcCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(srcCandidates, "memref.copy");
    auto srcTy = src.type.dyn_cast<MemRefType>();

    auto destCandidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref},
        typeEquivalentFilter(srcTy));
    if (destCandidates.empty()) {
      destCandidates.push_back(
          region.pool.generateRankedMemref(builder, loc, srcTy));
    }
    auto dest = sampleTypedValueFrom(destCandidates, "memref.copy");

    return builder.create<memref::CopyOp>(loc, src.val, dest.val)
        .getOperation();
  };
}

OpGenerator assumeAlignmentGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto memCandidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref},
        emptyFilter());
    if (memCandidates.empty()) {
      memCandidates.push_back(region.pool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    if (memCandidates.empty()) {
      auto memType = randomStaticShapedMemrefType(builder.getContext());
      auto tVal = TypeValue(
          memType, initMemref(builder, loc, memType.dyn_cast<MemRefType>()));
      memCandidates.push_back(tVal);
      typedValuePool.addStaticShapedMemref(tVal, "new(assume_align)");
    }
    TypeValue mem = sampleTypedValueFrom(memCandidates, "memref.assume_align");
    // TODO-should we unify the alignment factor in the whole program?
    auto align = alignmentFactors[UR(alignmentFactors.size())];
    return builder.create<memref::AssumeAlignmentOp>(loc, mem.val, align)
        .getOperation();
  };
}

OpGenerator allocGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto type = randomRankedMemrefType(builder.getContext());
    SmallVector<Value> dyDims;
    auto indexCandidates = region.pool.getCandidatesFromIndex(builder, loc);
    for (int i = 0; i < type.getNumDynamicDims(); ++i) {
      dyDims.push_back(
          sampleTypedValueFrom(indexCandidates, "memref.alloc").val);
    }
    auto mem = builder.create<memref::AllocOp>(loc, type, dyDims);
    auto tVal = TypeValue(type, mem);
    typedValuePool.addRankedMemref(tVal, "memref.alloc");
    return mem.getOperation();
  };
}

OpGenerator reallocGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    // 'memref.realloc' op operand #0 must be 1D memref of any type values
    auto candidates = typedValuePool.searchCandidatesFrom(
        {PoolType::StaticShapedMemref, PoolType::DynamicShapedMemref},
        [](TypeValue tval) {
          return tval.type.dyn_cast<MemRefType>().getRank() == 1;
        });

    if (candidates.empty()) {
      std::vector<int64_t> shape;
      if (UR(2)) {
        shape.push_back(dimPool[UR(dimPool.size())]);
      } else {
        shape.push_back(ShapedType::kDynamic);
      }
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto memTy = MemRefType::get(shape, elemTy);
      candidates.push_back(
          typedValuePool.generateRankedMemref(builder, loc, memTy));
    }
    auto source = sampleTypedValueFrom(candidates, "memref.realloc");
    std::vector<int64_t> shape;
    shape.push_back(dimPool[UR(dimPool.size())]);
    auto elemTy = source.type.dyn_cast<ShapedType>().getElementType();
    auto newType = MemRefType::get(shape, elemTy);
    auto newMem = builder.create<memref::ReallocOp>(
        loc, newType.dyn_cast<MemRefType>(), source.val);
    auto tVal = TypeValue(newType, newMem);
    typedValuePool.addRankedMemref(tVal, "memref.realloc");
    return newMem.getOperation();
  };
}

OpGenerator tensorStoreGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto srcCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::DynamicShapedTensor, PoolType::StaticShapedTensor},
        emptyFilter());
    if (srcCandidates.empty()) {
      srcCandidates.push_back(typedValuePool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto tensor = sampleTypedValueFrom(srcCandidates, "memref.tensor_store");
    auto srcShape = tensor.type.dyn_cast<ShapedType>();

    auto destMemTy =
        MemRefType::get(srcShape.getShape(), srcShape.getElementType());
    auto destSrcCandidates = region.pool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::DynamicShapedMemref},
        typeEquivalentFilter(destMemTy));

    if (destSrcCandidates.empty()) {
      destSrcCandidates.push_back(
          typedValuePool.generateRankedMemref(builder, loc, destMemTy));
    }
    auto destMem =
        sampleTypedValueFrom(destSrcCandidates, "memref.tensor_store");
    return builder.create<memref::TensorStoreOp>(loc, tensor.val, destMem.val)
        .getOperation();
  };
}

// No side-effecting ops are allowed in the body of GenericAtomicRMWOp.
OpGenerator genericAtomicRMWGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    std::string opName = "memref.generic_atomic_rmw";
    auto typedValuePool = parent.pool;
    auto memCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref},
        emptyFilter());
    if (memCandidates.empty()) {
      memCandidates.push_back(typedValuePool.generateRankedMemref(
          builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto memref = sampleTypedValueFrom(memCandidates, opName);
    auto shapedType = memref.type.dyn_cast<ShapedType>();
    auto elemType = shapedType.getElementType();
    std::set<std::string> ops;
    if (elemType.dyn_cast<IntegerType>()) {
      ops = intOpsForGenericAtomicRMW;
    } else if (elemType.dyn_cast<FloatType>()) {
      ops = floatOpsForGenericAtomicRMW;
    } else {
      std::cout << "unsupported type " << std::endl;
      exit(-1);
    }
    auto point = builder.saveInsertionPoint();
    auto opFilter = OpNameFilter(ops);
    auto indices = typedValuePool.randomIndicesForShapedType(
        memref.type.dyn_cast<ShapedType>(), builder, loc);
    auto op = builder.create<memref::GenericAtomicRMWOp>(loc, memref.val,
                                                         ValueRange(indices));
    auto region = OpRegion(opName, parent.depth + 1, parent.cur_child);
    auto tVal = TypeValue(elemType, op);
    // atomic body is not empty.
    for (auto arg : op.getAtomicBody().getArguments()) {
      auto tVal = TypeValue(elemType, arg);
      region.pool.addIntOrFloat(tVal, "arg(generic_atomic_rmw)");
    }
    region.pool.intOrFloatPool.insert(region.pool.intOrFloatPool.begin(),
                                      typedValuePool.intOrFloatPool.begin(),
                                      typedValuePool.intOrFloatPool.end());
    auto regionGen = RegionGen(&region, {opFilter});
    builder.setInsertionPointToEnd(&op.getAtomicBody().front());
    regionGen.apply(builder, loc, 16);
    // the candidates should not be empty
    auto yieldCandidates =
        searchTypedValueFrom(region.pool.intOrFloatPool, elemType);

    auto yield = sampleTypedValueFrom(yieldCandidates, opName);

    builder.create<memref::AtomicYieldOp>(loc, yield.val);
    // generate ops for block
    builder.restoreInsertionPoint(point);
    typedValuePool.addIntOrFloat(tVal, opName);
    return op.getOperation();
  };
}

OpGenerator allocaGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto type = randomRankedMemrefType(builder.getContext());
    auto indexCandidates = region.pool.getCandidatesFromIndex(builder, loc);
    SmallVector<Value> dynDims;
    for (int i = 0; i < type.getNumDynamicDims(); ++i) {
      dynDims.push_back(
          sampleTypedValueFrom(indexCandidates, "memref.alloca").val);
    }
    auto mem = builder.create<memref::AllocaOp>(loc, type, dynDims);
    auto tVal = TypeValue(type, mem);
    typedValuePool.addRankedMemref(tVal, "memref.alloca");
    return mem.getOperation();
  };
}

#include <iostream>
OpGenerator allocaScopeGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto point = builder.saveInsertionPoint();
    auto opFilter = OpNameFilter(opsForAlloca);
    SmallVector<Type> results;
    if (UR(2)) {
      results.push_back(randomType(builder.getContext()));
    }
    auto op = builder.create<memref::AllocaScopeOp>(loc, results);
    auto region = OpRegion("memref.alloca_scope", parent.depth + 1, parent.cur_child);
    region.pool.merge(parent.pool);
    auto regionGen = RegionGen(&region, {opFilter});
    Block *entry = new Block();

    op.getBodyRegion().push_back(entry);
    // block builder
    {
      builder.setInsertionPointToEnd(&op.getBodyRegion().front());
      regionGen.apply(builder, loc, 32);
      if (!results.empty()) {
        auto retTy = results[0];
        auto retCandidates = region.pool.searchCandidatesFrom(
            allTypes, typeEquivalentFilter(retTy));
        if (retCandidates.empty()) {
          retCandidates.push_back(
              region.pool.generateTypedValue(builder, loc, retTy));
        }
        auto ret =
            sampleTypedValueFrom(retCandidates, "memref.alloca_scope.return");
        builder.create<memref::AllocaScopeReturnOp>(loc, ret.val);
      } else {
        builder.create<memref::AllocaScopeReturnOp>(loc);
      }
      builder.restoreInsertionPoint(point);
    }
    if (!results.empty()) {
      if (!op->getResults().empty()) {
        auto val = op->getResult(0);
        auto tVal = TypeValue(val.getType(), val);
        parent.pool.addTypeValue(tVal, "memref.alloca_scope");
      }
    }
    return op.getOperation();
  };
}

OpGenerator memrefCastGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto typedValuePool = parent.pool;
    auto memrefCandidates = typedValuePool.searchCandidatesFrom({PoolType::DynamicShapedMemref, PoolType::StaticShapedMemref}, emptyFilter());
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(typedValuePool.generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext())));
    }
    auto mem = sampleTypedValueFrom(memrefCandidates, "memref.cast");
    // TODO - consider Unranked Memref here.
    assert(mem.type.dyn_cast<MemRefType>().hasRank());
    // convert between static and dynamic memref here.
    auto shapedType = mem.type.dyn_cast<ShapedType>();
    auto shape = shapedType.getShape();
    auto elemTy = shapedType.getElementType();

    std::vector<int64_t> newShape;
    for (int i = 0; i < shape.size(); ++i) {
      if (ShapedType::isDynamic (shape[i])) {
        if (UR(2)) {
          newShape.push_back(ShapedType::kDynamic);
        } else {
          auto newDim = dimPool[UR(dimPool.size())];
          newShape.push_back(newDim);
        }
      } else {
        if (UR(2)) {
          newShape.push_back(ShapedType::kDynamic);
        } else {
          newShape.push_back(shape[i]);
        }
      }
    }
    auto destMemRefType = MemRefType::get(newShape, elemTy);
    auto res = builder.create<memref::CastOp>(loc, destMemRefType, mem.val);
    auto tVal = TypeValue(destMemRefType, res);
    typedValuePool.addRankedMemref(tVal, "memref.cast");
    return res.getOperation();
  };
}