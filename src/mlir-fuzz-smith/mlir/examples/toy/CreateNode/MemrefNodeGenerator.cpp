#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

std::vector<uint32_t> alignmentFactors = {1, 2, 4, 8, 16};



OpGenerator memrefLoadGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memCandidates);
		if (memCandidates.empty()) {
      memCandidates.push_back(
				generateRankedMemref(
					builder, 
					loc, 
					randomRankedMemrefType(builder.getContext()),
					mb));
    }
		auto alloc = memCandidates[rollIdx(memCandidates.size())];
		auto t = alloc.getType().dyn_cast<ShapedType>();
		auto elemType = t.getElementType();

		SmallVector<Value, 8> loadIndices = randomIndicesForShapedType(t, builder, loc);
		mlir::Value val = builder.create<memref::LoadOp>(loc, alloc, loadIndices);
		mb->add2Pool(val);
		return true;
  };
}

OpGenerator memrefStoreGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memCandidates);
		if (memCandidates.empty()) {
      memCandidates.push_back(
				generateRankedMemref(
					builder, 
					loc, 
					randomRankedMemrefType(builder.getContext()), 
					mb));
    }
		auto alloc = memCandidates[rollIdx(memCandidates.size())];
		auto t = alloc.getType().dyn_cast<ShapedType>();
		auto elemType = t.getElementType();

		std::string elemTypeStr = getValueTypeStr(elemType);
		std::vector<mlir::Value> candidateValueToStore = std::vector<mlir::Value>();
		if (mb->valuePool.pool.count(elemTypeStr)) {
			candidateValueToStore.insert(
				candidateValueToStore.begin(), 
				mb->valuePool.pool[elemTypeStr].begin(), 
				mb->valuePool.pool[elemTypeStr].end());
		}
		if (candidateValueToStore.empty()) {
			candidateValueToStore.push_back(generateElement(builder, loc, elemType));
		}
		auto tValToStore = candidateValueToStore[rollIdx(candidateValueToStore.size())];
		auto storeIndices = randomIndicesForShapedType(t, builder, loc);
		builder.create<memref::StoreOp>(
			loc, tValToStore, alloc, storeIndices);
		return true;
  };
}

OpGenerator atomicRMWGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = ranked_memref_filter(s, v);
				if (!cond1) { return false; }
				auto elemTy = v.getType().dyn_cast<MemRefType>().getElementType();
				auto isInteger = elemTy.isa<IntegerType>();
				return elemTy.isa<FloatType>() ||
								(isInteger && (isLLVMIRIntegerBitWidth(
																	elemTy.dyn_cast<IntegerType>().getWidth())));
			},
			&memCandidates
		);
		if (memCandidates.empty()) {
      memCandidates.push_back(
				generateStaticShapedMemref(
					builder, loc, randomStaticShapedMemrefType(builder.getI32Type())));
    }
		auto memref = memCandidates[rollIdx(memCandidates.size())];
		auto shapedType = memref.getType().dyn_cast<ShapedType>();
    auto elemType = shapedType.getElementType();

		std::vector<mlir::Value> elemCandidates = std::vector<mlir::Value>();
		std::string elemTypeStr = getValueTypeStr(elemType);
		if (mb->valuePool.pool.count(elemTypeStr)) {
			elemCandidates.insert(
				elemCandidates.end(), 
				mb->valuePool.pool[elemTypeStr].begin(),
				mb->valuePool.pool[elemTypeStr].end());
		}
		if (elemCandidates.empty()) {
      elemCandidates.push_back(generateTypedValue(builder, loc, elemType, mb));
    }
		auto elem = elemCandidates[rollIdx(elemCandidates.size())];
		arith::AtomicRMWKind kind;
		if (elemType.dyn_cast<IntegerType>()) {
      kind = intRmwKinds[rollIdx(intRmwKinds.size())];
    } else if (elemType.dyn_cast<FloatType>()) {
      kind = floatRmwKinds[rollIdx(floatRmwKinds.size())];
    } else {
      llvm::outs() << "unsupported type \n";
      exit(-1);
    }
		auto loadIndices = randomIndicesForShapedType(shapedType, builder, loc);
		auto res = builder.create<memref::AtomicRMWOp>(
			loc, elemType, kind, elem, memref, ValueRange(loadIndices));
		return true;
  };
}

OpGenerator memrefCopyGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &srcCandidates);
		if (srcCandidates.empty()) {
      srcCandidates.push_back(
				generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
    }
		auto src = srcCandidates[rollIdx(srcCandidates.size())];
		auto srcTy = src.getType().dyn_cast<MemRefType>();

		std::string srcTyStr = getValueTypeStr(srcTy);
		std::vector<mlir::Value> destCandidates = std::vector<mlir::Value>();
		if (mb->valuePool.pool.count(srcTyStr)) {
			destCandidates.insert(
				destCandidates.begin(), 
				mb->valuePool.pool[srcTyStr].begin(), 
				mb->valuePool.pool[srcTyStr].end());
		}
		if (destCandidates.empty()) {
      destCandidates.push_back(generateRankedMemref(builder, loc, srcTy, mb));
    }
		auto dest = destCandidates[rollIdx(destCandidates.size())];
		builder.create<memref::CopyOp>(loc, src, dest);
		return true;
  };
}

OpGenerator assumeAlignmentGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memCandidates);
		if (memCandidates.empty()) {
      memCandidates.push_back(
				generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
    }
		if (memCandidates.empty()) {
      auto memType = randomStaticShapedMemrefType(builder.getContext());
      mlir::Value v = builder.create<memref::AllocOp>(
				loc, memType.dyn_cast<MemRefType>());
			memCandidates.push_back(v);
			mb->add2Pool(v);
    }
		mlir::Value mem = memCandidates[rollIdx(memCandidates.size())];
		auto align = alignmentFactors[rollIdx(alignmentFactors.size())];
    builder.create<memref::AssumeAlignmentOp>(loc, mem, align);
		return true;
  };
}

OpGenerator allocGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto type = randomRankedMemrefType(builder.getContext());
    SmallVector<Value> dyDims;
		std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
		mb->search(index_filter, &indexCandidates);
		if (indexCandidates.empty()) {
			indexCandidates.push_back(generateIndex(builder, loc, 1));
		}
		for (int i = 0; i < type.getNumDynamicDims(); ++i) {
      dyDims.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
    }
		mlir::Value mem = builder.create<memref::AllocOp>(loc, type, dyDims);
		mb->add2Pool(mem);
		return true;
  };
}

OpGenerator reallocGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		const int dim_ub = 32;

		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_tensor_filter_has_dim, &candidates);

		if (candidates.empty()) {
      std::vector<int64_t> shape;
      if (rollIdx(2)) {
        shape.push_back(rollIdx(dim_ub)+1);
      } else {
        shape.push_back(ShapedType::kDynamic);
      }
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto memTy = MemRefType::get(shape, elemTy);
      candidates.push_back(generateRankedMemref(builder, loc, memTy, mb));
    }
		auto source = candidates[rollIdx(candidates.size())];
		std::vector<int64_t> shape;
		shape.push_back(rollIdx(dim_ub)+1);
		auto elemTy = source.getType().dyn_cast<ShapedType>().getElementType();
		auto newType = MemRefType::get(shape, elemTy);
		mlir::Value newMem = builder.create<memref::ReallocOp>(
			loc, newType.dyn_cast<MemRefType>(), source);
		mb->add2Pool(newMem);
		return true;
  };
}

OpGenerator tensorStoreGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &srcCandidates);
		if (srcCandidates.empty()) {
      srcCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto tensor = srcCandidates[rollIdx(srcCandidates.size())];
		auto srcShape = tensor.getType().dyn_cast<ShapedType>();

		auto destMemTy = MemRefType::get(srcShape.getShape(), srcShape.getElementType());
    std::string destMemTyStr = getValueTypeStr(destMemTy);
		std::vector<mlir::Value> destSrcCandidates = std::vector<mlir::Value>();
		if (mb->valuePool.pool.count(destMemTyStr)) {
			destSrcCandidates.insert(
				destSrcCandidates.begin(), 
				mb->valuePool.pool[destMemTyStr].begin(), 
				mb->valuePool.pool[destMemTyStr].end());
		}
		if (destSrcCandidates.empty()) {
      destSrcCandidates.push_back(generateRankedMemref(builder, loc, destMemTy, mb));
    }
		auto destMem = destSrcCandidates[rollIdx(destSrcCandidates.size())];
		builder.create<memref::TensorStoreOp>(loc, tensor, destMem);
		return true;
  };
}

OpGenerator genericAtomicRMWGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memCandidates);
		if (memCandidates.empty()) {
      memCandidates.push_back(
				generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
    }
		auto memref = memCandidates[rollIdx(memCandidates.size())];
		auto shapedType = memref.getType().dyn_cast<ShapedType>();
		auto elemType = shapedType.getElementType();
		std::vector<OpGenerator> ops;
		if (elemType.dyn_cast<IntegerType>()) {
      ops = intOpsForGenericAtomicRMW;
    } else if (elemType.dyn_cast<FloatType>()) {
      ops = floatOpsForGenericAtomicRMW;
    } else {
      llvm::outs() << "unsupported type \n";
      exit(-1);
    }
		
		auto point = builder.saveInsertionPoint();
		auto indices = randomIndicesForShapedType(
			memref.getType().dyn_cast<ShapedType>(), builder, loc);
		auto op = builder.create<memref::GenericAtomicRMWOp>(
			loc, memref, ValueRange(indices));
		MutationBlock* childMb = new MutationBlock(*mb);
		for (auto arg : op.getAtomicBody().getArguments()) {
      childMb->add2Pool(arg);
    }
		builder.setInsertionPointToEnd(&op.getAtomicBody().front());
		unsigned statementNum = rollIdx(16);
		for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
			OpGenerator mutator = ops[rollIdx(ops.size())];
			mutator(builder, loc, childMb);
		}
		std::vector<mlir::Value> yieldCandidates = std::vector<mlir::Value>();
		std::string elemTypeStr = getValueTypeStr(elemType);
		if (childMb->valuePool.pool.count(elemTypeStr)) {
			yieldCandidates.insert(
				yieldCandidates.end(), 
				childMb->valuePool.pool[elemTypeStr].begin(), 
				childMb->valuePool.pool[elemTypeStr].end());
		}
		if (yieldCandidates.empty()) {
			yieldCandidates.push_back(generateElement(builder, loc, elemType));
		}
		
		auto yield = yieldCandidates[rollIdx(yieldCandidates.size())];
		builder.create<memref::AtomicYieldOp>(loc, yield);
    builder.restoreInsertionPoint(point);

		auto res = op->getResult(0);

		mb->add2Pool(res);
		return true;
  };
}

OpGenerator allocaGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto type = randomRankedMemrefType(builder.getContext());
		std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
		mb->search(index_filter, &indexCandidates);
		if (indexCandidates.empty()) {
			indexCandidates.push_back(generateIndex(builder, loc));
		}
		SmallVector<Value> dynDims;
		for (int i = 0; i < type.getNumDynamicDims(); ++i) {
      dynDims.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
    }
		mlir::Value mem = builder.create<memref::AllocaOp>(loc, type, dynDims);
		mb->add2Pool(mem);
		return true;
  };
}

OpGenerator allocaScopeGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto point = builder.saveInsertionPoint();
		SmallVector<Type> results;
		if (rollIdx(2)) {
      results.push_back(randomType(builder.getContext()));
    }
		auto op = builder.create<memref::AllocaScopeOp>(loc, results);
		MutationBlock* childMb = new MutationBlock(*mb);
		Block *entry = new Block();
		op.getBodyRegion().push_back(entry);
		{
      builder.setInsertionPointToEnd(&op.getBodyRegion().front());
			unsigned statementNum = rollIdx(32);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
			if (!results.empty()) {
        auto retTy = results[0];
				std::string retTyStr = getValueTypeStr(retTy);
				std::vector<mlir::Value> retCandidates = std::vector<mlir::Value>();
				if (mb->valuePool.pool.count(retTyStr)) {
					retCandidates.insert(
						retCandidates.begin(), 
						mb->valuePool.pool[retTyStr].begin(), 
						mb->valuePool.pool[retTyStr].end());
				}
				if (retCandidates.empty()) {
					retCandidates.push_back(generateTypedValue(builder, loc, retTy, mb));
				}
				mlir::Value ret = retCandidates[rollIdx(retCandidates.size())];

				llvm::outs() << "11111\n";

				builder.create<memref::AllocaScopeReturnOp>(loc, ret);

				llvm::outs() << "22222\n";
      } else {
        builder.create<memref::AllocaScopeReturnOp>(loc);
      }
		}
		if (!results.empty() && !op->getResults().empty()) {
			auto val = op->getResult(0);
			mb->add2Pool(val);
		}
		return true;
  };
}

OpGenerator memrefCastGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		const int dim_ub = 32;

		std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memrefCandidates);
		if (memrefCandidates.empty()) {
			memrefCandidates.push_back(
				generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
		}
		auto mem = memrefCandidates[rollIdx(memrefCandidates.size())];
		assert(mem.getType().dyn_cast<MemRefType>().hasRank());
		auto shapedType = mem.getType().dyn_cast<ShapedType>();
    auto shape = shapedType.getShape();
    auto elemTy = shapedType.getElementType();

		std::vector<int64_t> newShape;
		for (int i = 0; i < shape.size(); ++i) {
      if (ShapedType::isDynamic(shape[i])) {
        if (rollIdx(2)) {
          newShape.push_back(ShapedType::kDynamic);
        } else {
          auto newDim = rollIdx(dim_ub)+1;
          newShape.push_back(newDim);
        }
      } else {
        if (rollIdx(2)) {
          newShape.push_back(ShapedType::kDynamic);
        } else {
          newShape.push_back(shape[i]);
        }
      }
    }
		auto destMemRefType = MemRefType::get(newShape, elemTy);
    mlir::Value res = builder.create<memref::CastOp>(loc, destMemRefType, mem);
		mb->add2Pool(res);
    return true;
  };
}