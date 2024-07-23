//
// Created by Stan Wang on 2022/11/15.
//
#include "mlir/IR/Matchers.h"
#include "smith/RegionGeneration.h"
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"
#include "llvm/Support/Casting.h"

OpGenerator scfIfGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto opFilter = OpNameFilter(opsForScfIf);
    std::vector<TypeValue> i1Candidates;
    for (auto elem : parent.pool.intOrFloatPool) {
      if (elem.type.isInteger(1)) {
        i1Candidates.push_back(elem);
      }
    }
    if (i1Candidates.empty()) {
      auto tVal =
          parent.pool.generateElement(builder, loc, builder.getI1Type());
      i1Candidates.push_back(tVal);
    }
    auto condition = sampleTypedValueFrom(i1Candidates, "scf.if");
    SmallVector<Type> resultTypes;
    if (UR(2)) {
      resultTypes.push_back(randomNonTensorType(builder.getContext()));
    }
    bool hasElse = !resultTypes.empty() || UR(2);

    auto blockBuilder = [&](OpBuilder &b, Location loc) {
      auto region = OpRegion("scf.if", parent.depth + 1, parent.cur_child);
      region.pool.merge(parent.pool);
      auto regionGen = RegionGen(&region, {opFilter});
      regionGen.apply(builder, loc, 8);

      if (!resultTypes.empty()) {
        auto retTy = resultTypes[0];
        auto retCandidates = searchTypedValueFrom(region.pool, resultTypes[0]);
        if (retCandidates.empty()) {
          auto newVal = region.pool.generateTypedValue(builder, loc, retTy);
          retCandidates.push_back(newVal);
        }
        auto ret = sampleTypedValueFrom(retCandidates, "scf.yield");
        builder.create<scf::YieldOp>(loc, ret.val);
      } else {
        builder.create<scf::YieldOp>(loc);
      }
    };
    scf::IfOp ifOp;
    if (hasElse) {
      ifOp = builder.create<scf::IfOp>(loc, condition.val, blockBuilder,
                                       blockBuilder);
    } else {
      ifOp = builder.create<scf::IfOp>(loc, condition.val, blockBuilder);
    }

    for (auto val : ifOp->getResults()) {
      auto tVal = TypeValue(val.getType(), val);
      parent.pool.addTypeValue(tVal, "ret(scf.if)");
    }
    return ifOp.getOperation();
  };
}

OpGenerator executeRegionGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto point = builder.saveInsertionPoint();
    auto opFilter = OpNameFilter(opsForExecuteRegion);
    SmallVector<Type> resultTypes;
    if (UR(2)) {
      resultTypes.push_back(randomType(builder.getContext()));
    }

    auto blockBuilder = [&](OpBuilder &b, Location loc) {
      auto region = OpRegion("scf.execute_region", parent.depth + 1, parent.cur_child);
      region.pool.merge(parent.pool);
      auto regionGen = RegionGen(&region, {opFilter});
      regionGen.apply(builder, loc, 16);

      if (!resultTypes.empty()) {
        auto retTy = resultTypes[0];
        auto retCandidates = searchTypedValueFrom(region.pool, resultTypes[0]);
        if (retCandidates.empty()) {
          auto newVal = region.pool.generateTypedValue(builder, loc, retTy);
          retCandidates.push_back(newVal);
        }
        auto ret = sampleTypedValueFrom(retCandidates, "scf.yield");
        builder.create<scf::YieldOp>(loc, ret.val);
      } else {
        builder.create<scf::YieldOp>(loc);
      }
    };
    auto op = builder.create<scf::ExecuteRegionOp>(loc, resultTypes);

    Block *entry = new Block();
    op.getRegion().push_back(entry);
    builder.setInsertionPointToEnd(&op.getRegion().front());
    blockBuilder(builder, loc);

    for (auto val : op->getResults()) {
      auto tVal = TypeValue(val.getType(), val);
      parent.pool.addTypeValue(tVal, "ret(scf.execute_region)");
    }
    builder.restoreInsertionPoint(point);
    return op.getOperation();
  };
}

OpGenerator scfForGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto opFilter = OpNameFilter(opsForScfFor);
    // Zero successors.
    auto exFilter = excludeFilter(regionedOps);
    auto pool = parent.pool;
    auto lowerBound = sampleTypedValueFrom(
                          pool.getCandidatesFromIndex(builder, loc), "scf.for")
                          .val;
    auto upperBound = sampleTypedValueFrom(
                          pool.getCandidatesFromIndex(builder, loc), "scf.for")
                          .val;

    std::vector<TypeValue> stepCandidates;
    for (int i = 1; i < dim_ub; ++i) {
      stepCandidates.push_back(
          TypeValue(builder.getIndexType(), pool.constantIndices[i]));
    }
    auto step = sampleTypedValueFrom(stepCandidates, "scf.for").val;
    // this must be true
    bool hasIterArg = true;

    auto candidates = pool.searchCandidatesFrom(
        allTypes, [&](TypeValue tVal) { return true; });
    if (candidates.empty()) {
      candidates.push_back(parent.pool.generateTypedValue(
          builder, loc, randomType(builder.getContext())));
    }

    auto iterArg = sampleTypedValueFrom(candidates, "scf.for");
    auto iterArgVal = iterArg.val;

    auto blockBuilder = [&](OpBuilder &b, Location loc,
                            Value iv /*loop iterator*/, ValueRange args) {
      auto region = OpRegion("scf.for", parent.depth + 1, parent.cur_child);
      region.pool.merge(parent.pool);
      region.pool.addIndex(TypeValue(b.getIndexType(), iv), "iter(scf.for)");
      if (hasIterArg) {
        region.pool.addTypeValue(TypeValue(iterArg.type, args.front()),
                                 "arg(scf.for)");
      }

      auto regionGen = RegionGen(&region, {opFilter, exFilter});
      regionGen.apply(builder, loc, 16);
      if (hasIterArg) {
        auto retCandidates = region.pool.searchCandidatesFrom(
            allTypes, typeEquivalentFilter(iterArg.type));
        if (retCandidates.empty()) {
          retCandidates.push_back(
              region.pool.generateTypedValue(builder, loc, iterArg.type));
        }
        auto ret = sampleTypedValueFrom(retCandidates, "scf.for");
        builder.create<scf::YieldOp>(loc, ret.val);
      } else {
        builder.create<scf::YieldOp>(loc);
      }
    };
    auto res = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                          llvm::makeArrayRef(iterArgVal),
                                          blockBuilder);

    for (auto val : res->getResults()) {
      auto tVal = TypeValue(val.getType(), val);
      parent.pool.addTypeValue(tVal, "scf.for");
    }
    return res.getOperation();
  };
}

OpGenerator indexSwitchGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto pool = parent.pool;
    auto opFilter = OpNameFilter(opsForIndexSwitch);

    bool hasResult = UR(2);
    std::vector<Type> resultTypes;
    if (hasResult) {
      resultTypes.push_back(randomType(builder.getContext()));
    }
    auto arg = sampleTypedValueFrom(pool.getCandidatesFromIndex(builder, loc),
                                    "scf.index_switch");
    int caseRegionCount = UR(4) + 1; // at least one default region?
    SmallVector<int64_t> cases;
    for (int i = 0; i < caseRegionCount; ++i) {
      cases.push_back(i + 1);
    }
    auto idxSwitchOp = builder.create<scf::IndexSwitchOp>(
        loc, TypeRange(resultTypes), arg.val, cases, caseRegionCount);
    auto blockBuilder = [&](OpBuilder &b, Location loc) {
      OpRegion opRegion = OpRegion("scf.index_switch", parent.depth + 1, parent.cur_child);
      opRegion.pool.merge(parent.pool);
      auto regionGen = RegionGen(&opRegion, {opFilter});
      regionGen.apply(builder, loc, 16);
      if (hasResult) {
        auto ty = resultTypes[0];
        auto candidates = searchTypedValueFrom(opRegion.pool, ty);
        if (candidates.empty()) {
          auto tVal = opRegion.pool.generateTypedValue(builder, loc, ty);
          candidates.push_back(tVal);
        }
        auto ret = sampleTypedValueFrom(candidates, "scf.yield");
        builder.create<scf::YieldOp>(loc, ret.val);
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
    return idxSwitchOp.getOperation();
  };
}

OpGenerator scfWhileGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto opFilter = OpNameFilter(opsForScfWhile);
    auto pool = parent.pool;
    auto operand0Candidates = pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::StaticShapedMemref, PoolType::Vector,
         PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operand0Candidates.empty()) {
      operand0Candidates.push_back(pool.generateVector(
          builder, loc, randomVectorType(builder.getContext())));
    }
    auto operand0 = sampleTypedValueFrom(operand0Candidates, "scf.while");

    auto resultType = operand0.type;

    auto doBuilder = [&](OpBuilder &b, Location loc, ValueRange args) {
      OpRegion opRegion = OpRegion("scf.while", parent.depth + 1, parent.cur_child);
      opRegion.pool.merge(parent.pool);
      opRegion.pool.addTypeValue(TypeValue(resultType, args.front()),
                                 "arg(scf.while)");
      auto regionGen = RegionGen(&opRegion, {opFilter});
      regionGen.apply(builder, loc, 16);
      auto retCandidates = searchTypedValueFrom(opRegion.pool, resultType);
      if (retCandidates.empty()) {
        retCandidates.push_back(
            opRegion.pool.generateTypedValue(builder, loc, resultType));
      }
      auto ret = sampleTypedValueFrom(retCandidates, "scf.yield");
      builder.create<scf::YieldOp>(loc, ret.val);
    };
    auto condBuilder = [&](OpBuilder &b, Location loc, ValueRange args) {
      OpRegion opRegion = OpRegion("scf.while", parent.depth + 1, parent.cur_child);
      opRegion.pool.merge(parent.pool);
      auto regionGen = RegionGen(&opRegion, {opFilter});
      opRegion.pool.addTypeValue(TypeValue(resultType, args.front()),
                                 "arg(scf.while)");
      regionGen.apply(builder, loc, 8);

      // loop condition
      auto i1Candidates = searchTypedValueFrom(opRegion.pool.intOrFloatPool,
                                               builder.getI1Type());
      if (i1Candidates.empty()) {
        auto tVal =
            opRegion.pool.generateElement(builder, loc, builder.getI1Type());
        i1Candidates.push_back(tVal);
      }
      auto cond = sampleTypedValueFrom(i1Candidates, "scf.condition");

      // loop arg
      auto condOperandCandidates =
          searchTypedValueFrom(opRegion.pool, resultType);
      if (condOperandCandidates.empty()) {
        auto tVal = opRegion.pool.generateTypedValue(builder, loc, resultType);
        condOperandCandidates.push_back(tVal);
      }
      auto condOperand =
          sampleTypedValueFrom(condOperandCandidates, "scf.condition");

      builder.create<scf::ConditionOp>(loc, cond.val,
                                       ValueRange(condOperand.val));
    };
    return builder
        .create<scf::WhileOp>(loc, TypeRange(resultType),
                              ValueRange(operand0.val), condBuilder, doBuilder)
        .getOperation();
  };
}

// seems to be deprecated after d8b8911d
// OpGenerator scfForEachThreadGenerator() {
//  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
//    auto indexes = parent.pool.getCandidatesFromIndex(builder, loc);
//    SmallVector<Value> numThreads;
//    for (int i = 0; i < UR(rank_ub) + 1; ++i) {
//      numThreads.push_back(
//          sampleTypedValueFrom(indexes, "scf.foreach_group").val);
//    }
//    auto tensorCandidates = parent.pool.getCandidatesFromStaticShapedTensor(
//        builder, loc, randomStaticShapedTensorType(builder.getContext()));
//    SmallVector<Value> sharedOuts;
//    for (int i = 0; i < UR(3); ++i) {
//      sharedOuts.push_back(
//          sampleTypedValueFrom(tensorCandidates, "scf.foreach_group").val);
//    }
//    SmallVector<Attribute> map;
//    auto op = builder.create<scf::ForeachThreadOp>(
//        loc, sharedOuts, numThreads, map,
//        [&](OpBuilder &b, Location loc, ValueRange args) {
//          OpRegion region("scf.foreach_thread", parent.depth + 1, parent.cur_child);
//          for (auto arg : args) {
//            region.pool.addTypeValue(TypeValue(arg.getType(), arg),
//                                     "arg(scf.foreach_thread)");
//          }
//          region.pool.merge(parent.pool);
//          auto filter = OpNameFilter(opsForScfForeachThread);
//          RegionGen gen(&region, {filter});
//          gen.apply(b, loc, 16);
//          b.create<scf::PerformConcurrentlyOp>(loc);
//        });
//
//    return op.getOperation();
//  };
//}

OpGenerator scfParallelGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto indexCandidates = parent.pool.getCandidatesFromIndex(builder, loc);
    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    SmallVector<Value> inits;
    int iterNum = UR(2) + 1;
    auto nonNegativeConstantIndexes = parent.pool.searchCandidatesFrom(
        {PoolType::Index}, [&](TypeValue tval) {
          IntegerAttr step;
          return matchPattern(tval.val, m_Constant(&step)) && step.getInt() > 0;
        });
    assert(!nonNegativeConstantIndexes.empty());
    for (int i = 0; i < iterNum; ++i) {
      lowerBounds.push_back(
          sampleTypedValueFrom(indexCandidates, "scf.parallel").val);
      upperBounds.push_back(
          sampleTypedValueFrom(indexCandidates, "scf.parallel").val);
      steps.push_back(
          sampleTypedValueFrom(nonNegativeConstantIndexes, "scf.parallel").val);
    }
    auto argRange = {PoolType::StaticShapedMemref, PoolType::IntOrFloat,
                     PoolType::StaticShapedTensor,
                     PoolType::DynamicShapedTensor, PoolType::Vector};

    bool hasOperand = UR(2);
    if (hasOperand) {
      auto initCandidates =
          parent.pool.searchCandidatesFrom(argRange, emptyFilter());
      assert(!initCandidates.empty());
      inits.push_back(sampleTypedValueFrom(initCandidates, "scf.parallel").val);
    }
    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange ivs,
                           ValueRange args) {
      OpRegion region("scf.parallel", parent.depth + 1, parent.cur_child);
      for (auto arg : ivs) {
        region.pool.addIndex(TypeValue(arg.getType(), arg), "iv(scf.parallel)");
      }
      for (auto arg : args) {
        region.pool.addTypeValue(TypeValue(arg.getType(), arg),
                                 "init(scf.parallel");
      }
      region.pool.merge(parent.pool);
      auto filter = OpNameFilter(opsForScfParallel);
      RegionGen gen(&region, {filter});
      gen.apply(b, loc, 16);
      if (hasOperand) {
        assert(!inits.empty());
        auto reduceArgTy = inits.front().getType();
        auto reduceArgCandidates = region.pool.searchCandidatesFrom(
            argRange, typeEquivalentFilter(reduceArgTy));
        if (reduceArgCandidates.empty()) {
          reduceArgCandidates.push_back(
              region.pool.generateTypedValue(builder, loc, reduceArgTy));
        }
        auto reduceArg =
            sampleTypedValueFrom(reduceArgCandidates, "scf.reduce");

        b.create<scf::ReduceOp>(
            loc, reduceArg.val,
            [&](OpBuilder &builder1, Location loc, Value lhs, Value rhs) {
              OpRegion scfReduceRegion("scf.reduce", region.depth + 1);
              auto reduceFilter = OpNameFilter(opsForScfReduce);
              scfReduceRegion.pool.addTypeValue(TypeValue(lhs.getType(), lhs),
                                                "arg()");
              scfReduceRegion.pool.addTypeValue(TypeValue(lhs.getType(), lhs),
                                                "arg()");
              scfReduceRegion.pool.merge(region.pool);
              RegionGen gen1(&scfReduceRegion, {reduceFilter});
              gen1.apply(builder1, loc, 8);
              auto reduceReturnTy = lhs.getType();
              auto reduceReturnCandidates =
                  scfReduceRegion.pool.searchCandidatesFrom(
                      argRange, typeEquivalentFilter(reduceReturnTy));
              if (reduceReturnCandidates.empty()) {
                reduceReturnCandidates.push_back(
                    scfReduceRegion.pool.generateTypedValue(builder, loc,
                                                            reduceReturnTy));
              }
              auto reduceRet = sampleTypedValueFrom(reduceReturnCandidates,
                                                    "scf.reduce.return");
              builder1.create<scf::ReduceReturnOp>(loc, reduceRet.val);
            });
      }
    };

    return builder
        .create<scf::ParallelOp>(loc, ValueRange(lowerBounds),
                                 ValueRange(upperBounds), ValueRange(steps),
                                 ValueRange(inits), bodyBuilder)
        .getOperation();
  };
}
