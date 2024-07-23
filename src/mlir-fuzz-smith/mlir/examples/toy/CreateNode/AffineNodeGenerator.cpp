#include "NodeGenerator.h"

std::vector<int64_t> constants = {1, 2, 4, 8, 16, 32, 64, 128};

using AffineExprGen = std::function<AffineExpr(std::vector<AffineExpr> &, std::vector<int64_t>)>;

std::vector<AffineExprGen> exprGens = {
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto expr1 = exprs[rollIdx(exprs.size())];
	return expr0 + expr1;
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto expr1 = exprs[rollIdx(exprs.size())];
	return expr0 - expr1;
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto constant = constants[rollIdx(constants.size())];
	return expr0 + constant;
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto constant = constants[rollIdx(constants.size())];
	return expr0 - constant;
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto constant = constants[rollIdx(constants.size())];
	return expr0 * constant;
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto constant = constants[rollIdx(constants.size())];
	return expr0.floorDiv(constant);
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto constant = constants[rollIdx(constants.size())];
	return expr0.ceilDiv(constant);
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	auto constant = constants[rollIdx(constants.size())];
	return expr0 % constant;
},
[](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
	auto expr0 = exprs[rollIdx(exprs.size())];
	return -expr0;
}};

std::vector<AffineExpr> randomAffineExprs(OpBuilder &builder, int dimNum, int symbolNum) {
  const int affine_expr_comb_ub = 16;
	std::vector<AffineExpr> affineExprs;
  for (int j = 0; j < dimNum; ++j) {
    affineExprs.push_back(builder.getAffineDimExpr(j));
  }
  for (int j = 0; j < affine_expr_comb_ub; ++j) {
    auto exprGen = exprGens[rollIdx(exprGens.size())];
    affineExprs.push_back(exprGen(affineExprs, constants));
  }
  return affineExprs;
}

AffineMap randomAffineMap(OpBuilder &builder) {
  const int affine_expr_dim_ub = 4;
  int dimCount = rollIdx(affine_expr_dim_ub) + 1;
  auto affineExprs = randomAffineExprs(builder, dimCount, 0);
  auto resultExpr = affineExprs[rollIdx(affineExprs.size())];
  return AffineMap::get(dimCount, rollIdx(2), resultExpr);
}

// TODO: randomly generate a integer set
IntegerSet randomIntegerSet(OpBuilder &builder) {
	const int affine_expr_dim_ub = 4;
	const int integer_set_exprs_ub = 4;

	int dimCount = rollIdx(affine_expr_dim_ub) + 1;
	int exprCount = rollIdx(integer_set_exprs_ub) + 1;

	SmallVector<AffineExpr> constraints;
	SmallVector<bool> eqFlags;

	auto affineExprs = randomAffineExprs(builder, dimCount, 0);
	for (int j = 0; j < exprCount; ++j) {
		auto constraint = affineExprs[rollIdx(affineExprs.size())];
		constraints.push_back(constraint);
		eqFlags.push_back(rollIdx(2));
	}
	IntegerSet set = IntegerSet::get(dimCount, 0, constraints, eqFlags);
	return set;
}

//-----------------------------------------------------------------------
// The generators~
OpGenerator affineApplyGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    AffineMap map = randomAffineMap(builder);
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    SmallVector<Value> operands;
    mb->search(index_filter, &candidates);
    if (candidates.empty())
      candidates.push_back(generateIndex(builder, loc, rollIdx(10)));
    for (unsigned i = 0; i < map.getNumInputs(); ++i) {
      mlir::Value indexI = candidates[rollIdx(candidates.size())];
      operands.push_back(indexI);
    }
    mlir::Value ret = builder.create<mlir::affine::AffineApplyOp>(loc, map, operands);
    return true;
  };
}

OpGenerator affineForGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Zero successors.
    int64_t lowerBound = 0;
    int64_t upperBound = rollIdx(128);
    int64_t step = rollIdx(upperBound);

		// Strange logic in MLIRSmith, why argument and return value decide together?
    bool hasIterArg = rollIdx(2);
		bool hasRetVal = rollIdx(2);
		if (mb->valuePool.pool.empty()) {
			return false;
		}
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->getAll(&candidates);
		mlir::Value iterArgVal = candidates[rollIdx(candidates.size())];

		auto blockBuilder = [&](OpBuilder &b, Location loc,
                            Value iv /*loop iterator*/, ValueRange args) {
			// create the new region, and add all of the variables
      if (hasIterArg) {
				mlir::Value x = args.front();
				mb->add2Pool(x);
			}
			
			// generate for sub block
			MutationBlock* childMb = new MutationBlock(*mb);
      unsigned statementNum = rollIdx(10);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
			delete childMb;
			//end generation of sub block

			// generate the last line of Affine.for
      if (hasRetVal) {
				std::string ty = getValueTypeStr(iterArgVal);
				if (mb->valuePool.pool.count(ty)) {
					mlir::Value ret = mb->search(ty, "Unknown");
					builder.create<mlir::affine::AffineYieldOp>(loc, ret);
				} else {
					builder.create<mlir::affine::AffineYieldOp>(loc);
				}
      } else {
        builder.create<mlir::affine::AffineYieldOp>(loc);
      }
    };

		if (hasIterArg) {
      auto res = builder.create<mlir::affine::AffineForOp>(loc, lowerBound, upperBound, step,
                                             llvm::makeArrayRef(iterArgVal),
                                             blockBuilder);
    } else {
      auto value = builder.create<mlir::affine::AffineForOp>(loc, lowerBound, upperBound, step, 
																							 std::nullopt, blockBuilder);
    }
		return true;
  };
}

OpGenerator affineIfGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    IntegerSet set = randomIntegerSet(builder);
		// Get first operand
    SmallVector<Value> args;
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);

    if (candidates.size() == 0) {
      auto supportedTys = getSupportedIntTypes(builder.getContext());
      auto integerTy = supportedTys[rollIdx(supportedTys.size())].dyn_cast<IntegerType>();
      candidates.push_back(generateInteger(builder, loc, integerTy));
    }

		for (uint32_t i = 0; i < set.getNumInputs(); ++i) {
      unsigned idx = rollIdx(candidates.size());
      mlir::Value val = candidates[idx];
      args.push_back(val);
    }
		SmallVector<Type> resultTypes;
    bool hasResult = true;
		resultTypes.push_back(randomNonTensorType(builder.getContext()));
		bool withElseRegion = rollIdx(2);
    // Create if

    llvm::outs() << "[affineIfGenerator] length of resultTypes: " << std::to_string(resultTypes.size()) << "\n";

    auto ifOp = builder.create<mlir::affine::AffineIfOp>(loc, resultTypes, set, args, withElseRegion);
    auto point = builder.saveInsertionPoint();
    // Block Builder
    auto blockBuilder = [&]() {
			MutationBlock* childMb = new MutationBlock(*mb);
      unsigned statementNum = rollIdx(10);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
			if (hasResult) {
        auto resultType = resultTypes.front();
        std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
        std::string resTyStr = getValueTypeStr(resultType);
        if (childMb->valuePool.pool.count(resTyStr)) {
          candidates.insert(candidates.end(), childMb->valuePool.pool[resTyStr].begin(), childMb->valuePool.pool[resTyStr].end());  
        }
        if (candidates.empty()) {
          candidates.push_back(generateTypedValue(builder, loc, resultType, childMb));
        }
        auto ret = candidates[rollIdx(candidates.size())];
        builder.create<mlir::affine::AffineYieldOp>(loc, ValueRange(ret));
      }
      delete childMb;
    };
    // Then block
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
    // Else block
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
    return true;
  };
}

// affine load and store should appear in AffineScope ops.
OpGenerator affineLoadGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
    parentMb->search(ranked_memref_filter, &memCandidates);
    if (memCandidates.empty()) 
      memCandidates.push_back(generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), parentMb));
    auto memref = memCandidates[rollIdx(memCandidates.size())];
    auto shapedType = memref.getType().dyn_cast<ShapedType>();
    auto shape = shapedType.getShape();
    auto dim = shape.size();
    auto elemTy = shapedType.getElementType();
    // Get operand indices
    SmallVector<Value> indices;
    std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
    parentMb->search(index_filter, &idxCandidates);
    if (idxCandidates.empty()) 
      idxCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));
    for (uint32_t i = 0; i < dim; ++i) 
      indices.push_back(idxCandidates[rollIdx(idxCandidates.size())]);
    // Generate operation
    assert(memref);
    mlir::Value res = builder.create<mlir::affine::AffineLoadOp>(loc, memref, ValueRange(indices));
    return true;
  };
}

OpGenerator affineStoreGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
    parentMb->search(ranked_memref_filter, &memCandidates);
    if (memCandidates.empty()) 
      memCandidates.push_back(generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), parentMb));
    auto memref = memCandidates[rollIdx(memCandidates.size())];
    auto shapedType = memref.getType().dyn_cast<ShapedType>();
    auto shape = shapedType.getShape();
    auto dim = shape.size();
    auto elemTy = shapedType.getElementType();
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    std::string elemTyStr = getValueTypeStr(elemTy);
    if (parentMb->valuePool.pool.count(elemTyStr)) {
      candidates.insert(
        candidates.end(), 
        parentMb->valuePool.pool[elemTyStr].begin(), 
        parentMb->valuePool.pool[elemTyStr].end());
    }
    if (candidates.empty()) 
      candidates.push_back(generateTypedValue(builder, loc, elemTy, parentMb));
    mlir::Value elem2Store = candidates[rollIdx(candidates.size())];
    // Get operand indices
    SmallVector<Value> indices;
    std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
    parentMb->search(index_filter, &idxCandidates);
    if (idxCandidates.empty()) 
      idxCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));
    for (uint32_t i = 0; i < dim; ++i) 
      indices.push_back(idxCandidates[rollIdx(idxCandidates.size())]);
    builder.create<mlir::affine::AffineStoreOp>(loc, elem2Store, memref, ValueRange(indices));
    return true;
  };
}

OpGenerator affineMaxGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    SmallVector<Value> mapOperands;
    AffineMap map = randomAffineMap(builder);
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    parentMb->search(index_filter, &candidates);
    if (candidates.empty()) 
      candidates.push_back(generateIndex(builder, loc, rollIdx(10)));
    auto dimCount = map.getNumDims();
    auto symbolCount = map.getNumSymbols();
    for (int i = 0; i < dimCount + symbolCount; ++i) 
      mapOperands.push_back(candidates[rollIdx(candidates.size())]);
    auto index = builder.create<mlir::affine::AffineMaxOp>(loc, map, mapOperands);
    // if (index->getResultTypes().size() == 1 && index->getResultTypes()[0].dyn_cast<IndexType>()) 
    //   parentMb->add2Pool(index->getResult(0));
    return true;
  };
}

OpGenerator affineMinGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    SmallVector<Value> mapOperands;
    AffineMap map = randomAffineMap(builder);
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    parentMb->search(index_filter, &candidates);
    if (candidates.empty()) 
      candidates.push_back(generateIndex(builder, loc, rollIdx(10)));
    auto dimCount = map.getNumDims();
    auto symbolCount = map.getNumSymbols();
    for (int i = 0; i < dimCount + symbolCount; ++i) 
      mapOperands.push_back(candidates[rollIdx(candidates.size())]);
    auto index = builder.create<mlir::affine::AffineMinOp>(loc, map, mapOperands);
    // if (index->getResultTypes().size() == 1 && index->getResultTypes()[0].dyn_cast<IndexType>()) 
    //   parentMb->add2Pool(index->getResult(0));
    return true;
  };
}

OpGenerator affineParallelGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    const int rank_ub = 3;
    const int dim_ub = 32;
    
    auto elemTy = randomIntOrFloatType(builder.getContext());
    SmallVector<int64_t> ranges; // also the resShape
    auto rangeDim = rollIdx(rank_ub) + 1;
    for (int i = 0; i < rangeDim; ++i) 
      ranges.push_back(rollIdx(dim_ub)+1);
    bool hasResult = rollIdx(2);
    SmallVector<Type> resultTypes;
    if (hasResult) 
      resultTypes.push_back(MemRefType::get(ranges, elemTy));
    SmallVector<arith::AtomicRMWKind> reductions;
    for (int i = 0; i < resultTypes.size(); ++i) {
      if (elemTy.isa<IntegerType>()) {
        reductions.push_back(intRmwKinds[rollIdx(intRmwKinds.size())]);
      } else {
        reductions.push_back(floatRmwKinds[rollIdx(floatRmwKinds.size())]);
      }
    }
    auto op = builder.create<mlir::affine::AffineParallelOp>(loc, resultTypes, reductions, ranges);
    for (uint i = 0; i < resultTypes.size(); i++) {
      mlir::Value val = op.getResult(i);
      parentMb->add2Pool(val);
    }
    auto point = builder.saveInsertionPoint();
    auto &loopBody = op.getRegion();
    auto &block = loopBody.front();
    builder.setInsertionPointToStart(&block);
    {
      MutationBlock* childMb = new MutationBlock(*parentMb);
      auto args = loopBody.getArguments();
      for (uint i = 0; i < args.size(); ++i)
        childMb->add2Pool(args[i]);
      unsigned statementNum = rollIdx(10);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
        unsigned idx = rollIdx(generators.size());
        llvm::outs() << "[affineParallelGenerator] Selected generator: " << generatorsStr[idx] << "\n";
				OpGenerator mutator = generators[idx];
				mutator(builder, loc, childMb);
			}
			if (!resultTypes.empty()) {
        SmallVector<Value> rets;
        std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
        for (auto resultTy : resultTypes) {
          // auto ret = childMb->search(getValueTypeStr(elemTy), "None");
          // rets.push_back(ret);
          std::string resultTyStr = getValueTypeStr(resultTy);
          std::vector<mlir::Value> retsCandidates = std::vector<mlir::Value>();
          if (childMb->valuePool.pool.count(resultTyStr)) {
            retsCandidates.insert(
              retsCandidates.end(), 
              childMb->valuePool.pool[resultTyStr].begin(), 
              childMb->valuePool.pool[resultTyStr].end());
          }
          if (retsCandidates.empty()) {
            retsCandidates.push_back(generateTypedValue(builder, loc, resultTy, childMb));
          }
          rets.push_back(retsCandidates[rollIdx(retsCandidates.size())]);
        }
        builder.create<mlir::affine::AffineYieldOp>(loc, ValueRange(rets));
      }
      delete childMb;
    }
    builder.restoreInsertionPoint(point);
    return true;
  };
}

OpGenerator affinePrefetchGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
    parentMb->search(static_memref_filter, &memCandidates);
    if (memCandidates.empty())
      memCandidates.push_back(generateStaticShapedMemref(builder, loc, randomStaticShapedMemrefType(builder.getContext())));
    auto memref = memCandidates[rollIdx(memCandidates.size())];
    SmallVector<Value> indices;
    builder.create<mlir::affine::AffinePrefetchOp>(loc, memref, indices, rollIdx(2), rollIdx(4), rollIdx(2), randomAffineMap(builder));
    return true;
  };
}

OpGenerator affineVectorLoadGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
    parentMb->search(ranked_memref_filter, &memCandidates);
    if (memCandidates.empty()) 
      memCandidates.push_back(generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), parentMb));
    auto memref = memCandidates[rollIdx(memCandidates.size())];
    
    std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
    parentMb->search(mlir::affine::isValidDim, &idxCandidates);
    if (idxCandidates.empty())
      return false;
    SmallVector<Value> indices;
    auto memTy = memref.getType().dyn_cast<MemRefType>();
    for (int i = 0; i < memTy.getRank(); ++i) 
      indices.push_back(idxCandidates[rollIdx(idxCandidates.size())]);
    auto vecTy = random1DVectorType(memTy.getElementType());
    mlir::Value op = builder.create<mlir::affine::AffineVectorLoadOp>(loc, vecTy, memref, ValueRange(indices));
    parentMb->add2Pool(op);
    return true;
  };
}

OpGenerator affineVectorStoreGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* parentMb) -> bool {
    std::vector<mlir::Value> vecCandidates = std::vector<mlir::Value>();
    parentMb->search(
      [](std::string s, mlir::Value v) -> bool {
          bool cond1 = s.find("vector");
          if (!cond1) return false;
          if (auto ty = v.getType().dyn_cast<VectorType>()) {
            return ty.getRank() == 1;
          }
          return false;
        }, 
        &vecCandidates
    );

    if (vecCandidates.empty()) 
        vecCandidates.push_back(generateVector(builder, loc, random1DVectorType(builder.getContext()), parentMb));
    auto vec2Store = vecCandidates[rollIdx(vecCandidates.size())];
    auto vecTy = vec2Store.getType().dyn_cast<VectorType>();

    llvm::outs() << "[affineVectorStoreGenerator] second param\n";
    
    std::vector<mlir::Value> memCandidates = std::vector<mlir::Value>();
    parentMb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        bool cond1 = (s.find("memref") == 0);
        if (!cond1)  // trancate
          return false;
        mlir::Type valueType = v.getType();
        bool cond2 = false;
        if (auto rankedMemRefType = valueType.dyn_cast<MemRefType>()) 
          cond2 = v.getType().dyn_cast<ShapedType>().getElementType() == vecTy.getElementType();
        return cond2;
      },
      &memCandidates
    );
    if (memCandidates.empty()) 
      memCandidates.push_back(generateRankedMemref(builder, loc, 
                                                  randomRankedMemrefType(vecTy.getElementType()), parentMb));
    auto memref = memCandidates[rollIdx(memCandidates.size())];

    std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
    parentMb->search(mlir::affine::isValidDim, &idxCandidates);
    
    if (idxCandidates.empty())
      idxCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));

    SmallVector<Value> indices;
    auto memTy = memref.getType().dyn_cast<MemRefType>();
    for (int i = 0; i < memTy.getRank(); ++i)
      indices.push_back(idxCandidates[rollIdx(idxCandidates.size())]);

    llvm::outs() << "[affineVectorStoreGenerator] third param\n";

    auto op = builder.create<mlir::affine::AffineVectorStoreOp>(loc, vec2Store, memref, ValueRange(indices));
    return true;
  };
}