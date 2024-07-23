#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

OpGenerator scfIfGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> i1Candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = int_float_filter(s);
				if (!cond1) { return false; }
				return v.getType().isInteger(1);
			},
			&i1Candidates
		);
		if (i1Candidates.empty()) {
      i1Candidates.push_back(generateElement(builder, loc, builder.getI1Type()));
    }
		auto condition = i1Candidates[rollIdx(i1Candidates.size())];
		SmallVector<Type> resultTypes;
		if (rollIdx(2)) {
      resultTypes.push_back(randomNonTensorType(builder.getContext()));
    }
		bool hasElse = !resultTypes.empty() || rollIdx(2);

		auto blockBuilder = [&](OpBuilder &b, Location loc) {
			MutationBlock* childMb = new MutationBlock(*mb);
			unsigned statementNum = rollIdx(8);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}

			if (!resultTypes.empty()) {
        auto retTy = resultTypes[0];
				std::string retTyStr = getValueTypeStr(retTy);
				std::vector<mlir::Value> retCandidates = std::vector<mlir::Value>();
				if (childMb->valuePool.pool.count(retTyStr)) {
					retCandidates.insert(
						retCandidates.begin(), 
						childMb->valuePool.pool[retTyStr].begin(), 
						childMb->valuePool.pool[retTyStr].end());
				}
				if (retCandidates.empty()) {
          retCandidates.push_back(generateTypedValue(builder, loc, retTy, childMb));
        }
				auto ret = retCandidates[rollIdx(retCandidates.size())];
				builder.create<scf::YieldOp>(loc, ret);
      } else {
        builder.create<scf::YieldOp>(loc);
      }
    };
		scf::IfOp ifOp;
    if (hasElse) {
      ifOp = builder.create<scf::IfOp>(loc, condition, blockBuilder, blockBuilder);
    } else {
      ifOp = builder.create<scf::IfOp>(loc, condition, blockBuilder);
    }
		for (mlir::Value val : ifOp->getResults()) {
			mb->add2Pool(val);
    }
    return true;
  };
}

OpGenerator executeRegionGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		auto point = builder.saveInsertionPoint();
		SmallVector<Type> resultTypes;
		if (rollIdx(2)) {
      resultTypes.push_back(randomType(builder.getContext()));
    }

		auto blockBuilder = [&](OpBuilder &b, Location loc) {
			MutationBlock* childMb = new MutationBlock(*mb);
			unsigned statementNum = rollIdx(16);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
			if (!resultTypes.empty()) {
        auto retTy = resultTypes[0];
				std::string retTyStr = getValueTypeStr(retTy);
				std::vector<mlir::Value> retCandidates = std::vector<mlir::Value>();
				if (childMb->valuePool.pool.count(retTyStr)) {
					retCandidates.insert(
						retCandidates.begin(), 
						childMb->valuePool.pool[retTyStr].begin(), 
						childMb->valuePool.pool[retTyStr].end());
				}
				if (retCandidates.empty()) {
          retCandidates.push_back(generateTypedValue(builder, loc, retTy, childMb));
        }
				auto ret = retCandidates[rollIdx(retCandidates.size())];
        builder.create<scf::YieldOp>(loc, ret);
      } else {
        builder.create<scf::YieldOp>(loc);
      }
    };
		auto op = builder.create<scf::ExecuteRegionOp>(loc, resultTypes);

		Block *entry = new Block();
    op.getRegion().push_back(entry);
    builder.setInsertionPointToEnd(&op.getRegion().front());
    blockBuilder(builder, loc);

		for (mlir::Value val : op->getResults()) {
      mb->add2Pool(val);
    }
    builder.restoreInsertionPoint(point);
    return true;
  };
}

OpGenerator scfForGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		const int dim_ub = 32;

    llvm::outs() << "11111\n";

		std::vector<mlir::Value> lowerBoundCandidates = std::vector<mlir::Value>();
		mb->search(index_filter, &lowerBoundCandidates);
		if (lowerBoundCandidates.empty()) {
			lowerBoundCandidates.push_back(generateIndex(builder, loc, 1));
		}
		mlir::Value lowerBound = lowerBoundCandidates[rollIdx(lowerBoundCandidates.size())];
		mlir::Value upperBound = lowerBoundCandidates[rollIdx(lowerBoundCandidates.size())];

		std::vector<mlir::Value> stepCandidates;
		for (int i = 1; i < dim_ub; ++i) {
      stepCandidates.push_back(generateIndex(builder, loc, rollIdx(dim_ub)+1));
    }
		auto step = stepCandidates[rollIdx(stepCandidates.size())];
		bool hasIterArg = true;

		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->getAll(&candidates);
		if (candidates.empty()) {
			candidates.push_back(
				generateTypedValue(builder, loc, randomType(builder.getContext()), mb));
		}

    llvm::outs() << "22222\n";

		auto iterArgVal = candidates[rollIdx(candidates.size())];

		auto blockBuilder = [&](OpBuilder &b, Location loc,
                            Value iv /*loop iterator*/, ValueRange args) {

			llvm::outs() << "33333\n"; 

      MutationBlock* childMb = new MutationBlock(*mb);
			childMb->add2Pool(iv);
			if (hasIterArg) {
        auto arg = args.front();
        childMb->add2Pool(arg);
      }
			unsigned statementNum = rollIdx(16);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
        unsigned selectedIndex = rollIdx(opsForScfFor.size());
        llvm::outs() << "[scfForGenerator] selected one: " << generatorsStr[selectedIndex] << "\n";
				OpGenerator mutator = opsForScfFor[selectedIndex];
				mutator(builder, loc, childMb);
			}

      llvm::outs() << "33333\n"; 

			if (hasIterArg) {
				std::vector<mlir::Value> retCandidates = std::vector<mlir::Value>();
				std::string iterArgValTyStr = getValueTypeStr(iterArgVal);
				if (childMb->valuePool.pool.count(iterArgValTyStr)) {
					retCandidates.insert(
						retCandidates.begin(), 
						childMb->valuePool.pool[iterArgValTyStr].begin(), 
						childMb->valuePool.pool[iterArgValTyStr].end());
				}
				if (retCandidates.empty()) {
          retCandidates.push_back(generateTypedValue(builder, loc, iterArgVal.getType(), childMb));
        }
				auto ret = retCandidates[rollIdx(retCandidates.size())];
        builder.create<scf::YieldOp>(loc, ret);
      } else {
        builder.create<scf::YieldOp>(loc);
      }

      llvm::outs() << "44444\n";
    };

		auto res = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                          llvm::makeArrayRef(iterArgVal),
                                          blockBuilder);

		for (mlir::Value val : res->getResults()) {
			mb->add2Pool(val);
    }
    return true;
  };
}

OpGenerator indexSwitchGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		bool hasResult = rollIdx(2);
		std::vector<Type> resultTypes;
		if (hasResult) {
      resultTypes.push_back(randomType(builder.getContext()));
    }
		std::vector<mlir::Value> argCandadites = std::vector<mlir::Value>();
		mb->search(index_filter, &argCandadites);
		if (argCandadites.empty()) {
			argCandadites.push_back(generateIndex(builder, loc, 1));
		}
		mlir::Value arg = argCandadites[rollIdx(argCandadites.size())];
		int caseRegionCount = rollIdx(4) + 1;
		SmallVector<int64_t> cases;
    for (int i = 0; i < caseRegionCount; ++i) {
      cases.push_back(i + 1);
    }
		auto idxSwitchOp = builder.create<scf::IndexSwitchOp>(
			loc, TypeRange(resultTypes), arg, cases, caseRegionCount);
		auto blockBuilder = [&](OpBuilder &b, Location loc) {
			MutationBlock* childMb = new MutationBlock(*mb);
			unsigned statementNum = rollIdx(16);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
      if (hasResult) {
        mlir::Type ty = resultTypes[0];
				std::string tyStr = getValueTypeStr(ty);
				std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
				if (childMb->valuePool.pool.count(tyStr)) {
					candidates.insert(
						candidates.begin(), 
						childMb->valuePool.pool[tyStr].begin(), 
						childMb->valuePool.pool[tyStr].end());
				}
				if (candidates.empty()) {
          candidates.push_back(generateTypedValue(builder, loc, ty, childMb));
        }
				auto ret = candidates[rollIdx(candidates.size())];
        builder.create<scf::YieldOp>(loc, ret);
      } else {
        builder.create<scf::YieldOp>(loc);
      }
    };
		auto point = builder.saveInsertionPoint();
    for (auto &region : idxSwitchOp.getCaseRegions()) {
      Block *block = builder.createBlock(&region);
      builder.setInsertionPointToEnd(block);
      blockBuilder(builder, loc);
    }
		auto block = builder.createBlock(&idxSwitchOp.getDefaultRegion());
    builder.setInsertionPointToEnd(block);
    blockBuilder(builder, loc);
    builder.restoreInsertionPoint(point);
    return true;
  };
}

OpGenerator scfWhileGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operand0Candidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        bool cond1 = int_float_filter(s);
        bool cond2 = static_memref_filter(s, v);
        bool cond3 = vector_filter(s);
        bool cond4 = ranked_tensor_filter(s, v);
        return cond1 || cond2 || cond3 || cond4;
      },
      &operand0Candidates
    );
    if (operand0Candidates.empty()) {
      operand0Candidates.push_back(
        generateVector(builder, loc, randomVectorType(builder.getContext()), mb));
    }
    auto operand0 = operand0Candidates[rollIdx(operand0Candidates.size())];

    auto resultType = operand0.getType();

    auto doBuilder = [&](OpBuilder &b, Location loc, ValueRange args) {
      MutationBlock* childMb = new MutationBlock(*mb);
      auto arg = args.front();
      childMb->add2Pool(arg);
      unsigned statementNum = rollIdx(16);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
      std::string resultTypeStr = getValueTypeStr(resultType);
      std::vector<mlir::Value> retCandidates = std::vector<mlir::Value>();
      if (childMb->valuePool.pool.count(resultTypeStr)) {
        retCandidates.insert(
          retCandidates.end(), 
          childMb->valuePool.pool[resultTypeStr].begin(), 
          childMb->valuePool.pool[resultTypeStr].end());
      }
      if (retCandidates.empty()) {
        retCandidates.push_back(generateTypedValue(builder, loc, resultType, childMb));
      }
      auto ret = retCandidates[rollIdx(retCandidates.size())];
      builder.create<scf::YieldOp>(loc, ret);
    };
    auto condBuilder = [&](OpBuilder &b, Location loc, ValueRange args) {
      MutationBlock* childMb = new MutationBlock(*mb);
      auto arg = args.front();
      childMb->add2Pool(arg);
      unsigned statementNum = rollIdx(8);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
      std::vector<mlir::Value> i1Candidates = std::vector<mlir::Value>();
      childMb->search(
        [] (std::string s) -> bool {
          return s == "i1";
        },
        &i1Candidates
      );
      if (i1Candidates.empty()) {
        auto tVal = generateElement(builder, loc, builder.getI1Type());
        childMb->add2Pool(tVal);
      }
      auto cond = i1Candidates[rollIdx(i1Candidates.size())];
      std::string resultTypeStr = getValueTypeStr(resultType);
      std::vector<mlir::Value> retCandidates = std::vector<mlir::Value>();
      if (childMb->valuePool.pool.count(resultTypeStr)) {
        retCandidates.insert(
          retCandidates.end(), 
          childMb->valuePool.pool[resultTypeStr].begin(), 
          childMb->valuePool.pool[resultTypeStr].end());
      }
      if (retCandidates.empty()) {
        retCandidates.push_back(generateTypedValue(builder, loc, resultType, childMb));
      }
      auto condOperand = retCandidates[rollIdx(retCandidates.size())];
      builder.create<scf::ConditionOp>(loc, cond, ValueRange(condOperand));
    };
    builder.create<scf::WhileOp>(
      loc, TypeRange(resultType), ValueRange(operand0), condBuilder, doBuilder);
    return true;
  };
}

OpGenerator scfParallelGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
    mb->search(index_filter, &indexCandidates);
    if (indexCandidates.empty()) {
      indexCandidates.push_back(generateIndex(builder, loc, 1));
    }
    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    SmallVector<Value> inits;
    int iterNum = rollIdx(2) + 1;
    std::vector<mlir::Value> nonNegativeConstantIndexes = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!index_filter(s)) { return false; }
        IntegerAttr step;
        return matchPattern(v, m_Constant(&step)) && step.getInt() > 0;
      },
      &nonNegativeConstantIndexes
    );
    if (nonNegativeConstantIndexes.empty()) {
      return false;
    }
    for (int i = 0; i < iterNum; ++i) {
      lowerBounds.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
      upperBounds.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
      steps.push_back(nonNegativeConstantIndexes[rollIdx(nonNegativeConstantIndexes.size())]);
    }
    bool hasOperand = rollIdx(2);
    if (hasOperand) {
      std::vector<mlir::Value> initCandidates = std::vector<mlir::Value>();
      mb->search(
        [] (std::string s, mlir::Value v) -> bool {
          bool cond1 = static_memref_filter(s, v);
          bool cond2 = int_float_filter(s);
          bool cond3 = ranked_tensor_filter(s, v);
          bool cond4 = vector_filter(s);
          return cond1 || cond2 || cond3 || cond4;
        },
        &initCandidates
      );
      assert(!initCandidates.empty());
      inits.push_back(initCandidates[rollIdx(initCandidates.size())]);
    }
    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange args) {
      MutationBlock* childMb = new MutationBlock(*mb);
      for (mlir::Value arg : ivs) {
        childMb->add2Pool(arg);
      }
      for (mlir::Value arg : args) {
        childMb->add2Pool(arg);
      }
      unsigned statementNum = rollIdx(16);
			for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
				OpGenerator mutator = generators[rollIdx(generators.size())];
				mutator(builder, loc, childMb);
			}
      if (hasOperand) {
        assert(!inits.empty());
        auto reduceArgTy = inits.front().getType();
        std::string reduceArgTyStr = getValueTypeStr(reduceArgTy);
        std::vector<mlir::Value> reduceArgCandidates = std::vector<mlir::Value>();
        if (childMb->valuePool.pool.count(reduceArgTyStr)) {
          reduceArgCandidates.insert(
            reduceArgCandidates.end(), 
            childMb->valuePool.pool[reduceArgTyStr].begin(), 
            childMb->valuePool.pool[reduceArgTyStr].end());
        }
        if (reduceArgCandidates.empty()) {
          reduceArgCandidates.push_back(generateTypedValue(builder, loc, reduceArgTy, childMb));
        }
        auto reduceArg = reduceArgCandidates[rollIdx(reduceArgCandidates.size())];
        b.create<scf::ReduceOp>(
            loc, reduceArg,
            [&](OpBuilder &builder1, Location loc, Value lhs, Value rhs) {
              MutationBlock* childChildMb = new MutationBlock(*childMb);
              childChildMb->add2Pool(lhs);
              childChildMb->add2Pool(lhs);
              unsigned statementNum = rollIdx(8);
              for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
                OpGenerator mutator = generators[rollIdx(generators.size())];
                mutator(builder, loc, childChildMb);
              }
              auto reduceReturnTy = lhs.getType();
              std::vector<mlir::Value> reduceReturnCandidates = std::vector<mlir::Value>();
              std::string reduceReturnTyStr = getValueTypeStr(reduceReturnTy);
              if (childChildMb->valuePool.pool.count(reduceReturnTyStr)) {
                reduceReturnCandidates.insert(
                  reduceReturnCandidates.end(), 
                  childChildMb->valuePool.pool[reduceReturnTyStr].begin(), 
                  childChildMb->valuePool.pool[reduceReturnTyStr].end());
              }
              if (reduceReturnCandidates.empty()) {
                reduceReturnCandidates.push_back(
                  generateTypedValue(builder, loc, reduceReturnTy, childChildMb));
              }
              auto reduceRet = reduceReturnCandidates[rollIdx(reduceReturnCandidates.size())];
              builder1.create<scf::ReduceReturnOp>(loc, reduceRet);
            });
      }
    };
    builder.create<scf::ParallelOp>(
      loc, 
      ValueRange(lowerBounds), 
      ValueRange(upperBounds), 
      ValueRange(steps), 
      ValueRange(inits), 
      bodyBuilder);
    return true;
  };
}