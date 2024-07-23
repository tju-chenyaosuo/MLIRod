#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"
#include<bits/stdc++.h>

OpGenerator linalgGenericGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		const int linalg_operands_num_ub = 4;
		const int rank_ub = 3;
		const int dim_ub = 32;

		int input_num = rollIdx(linalg_operands_num_ub)+1;
		auto elemTy = randomIntOrFloatType(builder.getContext());

		SmallVector<AffineMap> indexingMaps;
    SmallVector<Type> argsType;
    Type initTy;
		auto outputRank = rollIdx(rank_ub) + 1;
		SmallVector<int64_t> dimMap;
    SmallVector<utils::IteratorType> iteratorTypes;
    SmallVector<Type> operandTys;
		for (int i = 0; i < outputRank; ++i) {
      if (rollIdx(2)) {
        dimMap.push_back(rollIdx(dim_ub)+1);
      } else {
        dimMap.push_back(ShapedType::kDynamic);
      }
      if (rollIdx(2)) {
        iteratorTypes.push_back(utils::IteratorType::parallel);
      } else {
        iteratorTypes.push_back(utils::IteratorType::reduction);
      }
    }
		iteratorTypes[rollIdx(iteratorTypes.size())] = utils::IteratorType::parallel;
		SmallVector<AffineExpr> affineExprs0;
    SmallVector<int64_t> shape0;
    for (int j = 0; j < outputRank; ++j) {
        affineExprs0.push_back(builder.getAffineDimExpr(j));
        shape0.push_back(dimMap[j]);
    }
    indexingMaps.push_back(AffineMap::get(/*dimCount=*/outputRank,
                                          /*symbolCount=*/0, affineExprs0,
                                          builder.getContext()));
		if (rollIdx(2)) {
      operandTys.push_back(RankedTensorType::get(shape0, elemTy));
    } else {
      operandTys.push_back(MemRefType::get(shape0, elemTy));
    }

		for (int i = 1; i < input_num; i++) {
      SmallVector<AffineExpr> affineExprs;
      SmallVector<int64_t> shape;
      for (int j = 0; j < outputRank; ++j) {
        if (rollIdx(2)) {
          affineExprs.push_back(builder.getAffineDimExpr(j));
          shape.push_back(dimMap[j]);
        }
      }
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/outputRank,
                                            /*symbolCount=*/0, affineExprs,
                                            builder.getContext()));
      if (rollIdx(2)) {
        operandTys.push_back(RankedTensorType::get(shape, elemTy));
      } else {
        operandTys.push_back(MemRefType::get(shape, elemTy));
      }
    }
		SmallVector<AffineExpr> fullRankExprs;
    SmallVector<int64_t> initShape;
    for (int i = 0; i < outputRank; ++i) {
      if (iteratorTypes[i] == utils::IteratorType::parallel) {
        fullRankExprs.push_back(builder.getAffineDimExpr(i));
        initShape.push_back(dimMap[i]);
      }
    }
		initTy = RankedTensorType::get(initShape, elemTy);
    indexingMaps.push_back(AffineMap::get(outputRank, 0, fullRankExprs, builder.getContext()));
		
		SmallVector<Value> operands;
    for (mlir::Type operandTy : operandTys) {
			std::string tyStr = getValueTypeStr(operandTy);
			std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
			if (mb->valuePool.pool.count(tyStr)) {
				candidates.insert(
					candidates.end(), mb->valuePool.pool[tyStr].begin(), mb->valuePool.pool[tyStr].end());
			}
			if (candidates.empty()) {
				candidates.push_back(generateTypedValue(builder, loc, operandTy, mb));
			}
			operands.push_back(candidates[rollIdx(candidates.size())]);
    }

		std::string tyStr = getValueTypeStr(initTy);
		std::vector<mlir::Value> initCandidates = std::vector<mlir::Value>();
		if (mb->valuePool.pool.count(tyStr)) {
			initCandidates.insert(
				initCandidates.end(), mb->valuePool.pool[tyStr].begin(), mb->valuePool.pool[tyStr].end());
		}
		if (initCandidates.empty()) {
			initCandidates.push_back(generateTypedValue(builder, loc, initTy, mb));
		}
		auto init = initCandidates[rollIdx(initCandidates.size())];

		auto op = builder.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/
        TypeRange(initTy),
        /*inputs=*/ValueRange(operands),
        /*outputs=*/ValueRange(init),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &b, Location loc, ValueRange args) {
					MutationBlock* childMb = new MutationBlock(*mb);
          for (mlir::Value it : args) { childMb->add2Pool(it); }
					unsigned statementNum = rollIdx(10);
					for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
						OpGenerator mutator = generators[rollIdx(generators.size())];
						mutator(builder, loc, childMb);
					}
					std::vector<mlir::Value> yieldCandidates = std::vector<mlir::Value>();
					std::string retTyStr = getValueTypeStr(elemTy);
					if (childMb->valuePool.pool.count(retTyStr)) {
						yieldCandidates.insert(
							yieldCandidates.end(), 
							childMb->valuePool.pool[retTyStr].begin(), 
							childMb->valuePool.pool[retTyStr].end());
					}
					if (yieldCandidates.empty()) {
						yieldCandidates.push_back(generateTypedValue(builder, loc, elemTy, childMb));
					}
					auto value2Yield = yieldCandidates[rollIdx(yieldCandidates.size())];
					b.create<linalg::YieldOp>(loc, value2Yield);
        });
		for (mlir::Value res : op->getResults()) { mb->add2Pool(res); }
    return true;
  };
}

OpGenerator linalgBroadCastGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> inputCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_tensor_filter, &inputCandidates);
    if (inputCandidates.empty()) {
      Type t;
      if (rollIdx(2)) {
        t = randomRankedTensorType(builder.getContext());
      } else {
        t = randomRankedMemrefType(builder.getContext());
      }
      inputCandidates.push_back(generateTypedValue(builder, loc, t, mb));
    }
    auto operand = inputCandidates[rollIdx(inputCandidates.size())];
    auto srcShapeTy = operand.getType().dyn_cast<ShapedType>();
    llvm::outs() << srcShapeTy.getRank() << ": src\n";
    SmallVector<int64_t> initShape(srcShapeTy.getShape());
    initShape.push_back(rollIdx(dim_ub)+1);

    SmallVector<int64_t> dimension = {
        operand.getType().dyn_cast<ShapedType>().getRank()};
    Type initTy;
    if (rollIdx(2)) {
      initTy = MemRefType::get(
        initShape, operand.getType().dyn_cast<ShapedType>().getElementType());
    } else {
      initTy = RankedTensorType::get(
        initShape, operand.getType().dyn_cast<ShapedType>().getElementType());
    }
    llvm::outs() << initTy.dyn_cast<ShapedType>().getRank() << ": dest\n";
    
    std::vector<mlir::Value> initCandidates = std::vector<mlir::Value>();
    std::string initTyStr = getValueTypeStr(initTy);
    if (mb->valuePool.pool.count(initTyStr)) {
      initCandidates.insert(
        initCandidates.begin(), 
        mb->valuePool.pool[initTyStr].begin(), 
        mb->valuePool.pool[initTyStr].end());
    }
    if (initCandidates.empty()) {
      if (initTy.isa<MemRefType>() && !initTy.dyn_cast<ShapedType>().hasStaticShape()) {
        initCandidates.push_back(
          generateDynamicShapedMemref(builder, loc, initTy.dyn_cast<MemRefType>(), mb));
      } else {
        initCandidates.push_back(generateTypedValue(builder, loc, initTy, mb));
      }
    }
    auto init = initCandidates[rollIdx(initCandidates.size())];
    
    SmallVector<NamedAttribute> attrs;
    auto op = builder.create<linalg::BroadcastOp>(loc, operand, init, dimension, attrs);
    if (!op.getResult().empty()) {
      for (mlir::Value res : op->getResults()) {
        mb->add2Pool(res);
      }
    }
    return true;
  };
}

OpGenerator linalgTransposeGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> fromCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_tensor_filter_has_dim, &fromCandidates);
    if (fromCandidates.empty()) {
      Type t;
      if (rollIdx(2)) {
        t = randomRankedTensorType(builder.getContext());
      } else {
        t = randomRankedMemrefType(builder.getContext());
      }
      fromCandidates.push_back(generateTypedValue(builder, loc, t, mb));
    }
    auto from = fromCandidates[rollIdx(fromCandidates.size())];
    SmallVector<int64_t> permutation;
    auto shapeTy = from.getType().dyn_cast<ShapedType>();
    for (int i = 0; i < shapeTy.getRank(); ++i) {
      permutation.push_back(i);
    }

    struct timespec ts;
    std::shuffle(permutation.begin(), permutation.end(),
                 std::default_random_engine((time_t)ts.tv_nsec));

    SmallVector<int64_t> initShape;
    for (int i = 0; i < shapeTy.getRank(); ++i) {
      auto idx = permutation[i];
      initShape.push_back(shapeTy.getDimSize(idx));
    }
    Type initTy;
    if (rollIdx(2)) {
      initTy = RankedTensorType::get(initShape, shapeTy.getElementType());
    } else {
      initTy = MemRefType::get(initShape, shapeTy.getElementType());
    }
    std::vector<mlir::Value> initCandidates = std::vector<mlir::Value>();
    std::string initTyStr = getValueTypeStr(initTy);
    if (mb->valuePool.pool.count(initTyStr)) {
      initCandidates.insert(
        initCandidates.begin(),
        mb->valuePool.pool[initTyStr].begin(), 
        mb->valuePool.pool[initTyStr].end());
    }
    if (initCandidates.empty()) {
      if (initTy.isa<MemRefType>() && !initTy.dyn_cast<ShapedType>().hasStaticShape()) {
        initCandidates.push_back(
          generateDynamicShapedMemref(builder, loc, initTy.dyn_cast<MemRefType>(), mb));
      } else {
        initCandidates.push_back(generateTypedValue(builder, loc, initTy, mb));
      }
    }
    auto init = initCandidates[rollIdx(initCandidates.size())];
    assert(!permutation.empty());
    SmallVector<NamedAttribute> attributes;
    auto op = builder.create<linalg::TransposeOp>(loc, from, init, permutation, attributes);
    if (!op->getResults().empty()) {
      for (mlir::Value res : op->getResults()) {
        mb->add2Pool(res);
      }
    }
    return true;
  };
}

//'linalg.copy' op result #0 must be ranked tensor
OpGenerator linalgCopyGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &tensorCandidates);
    if (tensorCandidates.empty()) {
      tensorCandidates.push_back(
        generateStaticShapedTensor(
          builder, loc, randomStaticShapedTensorType(builder.getContext())));
    }
    auto in = tensorCandidates[rollIdx(tensorCandidates.size())];
    std::vector<mlir::Value> outCandidates = std::vector<mlir::Value>();
    std::string inTypeStr = getValueTypeStr(in);
    if (mb->valuePool.pool.count(inTypeStr)) {
      outCandidates.insert(
        outCandidates.end(),
        mb->valuePool.pool[inTypeStr].begin(),
        mb->valuePool.pool[inTypeStr].end());
    }
    if (outCandidates.empty()) {
      outCandidates.push_back(generateTypedValue(builder, loc, in.getType(), mb));
    }
    auto out = outCandidates[rollIdx(outCandidates.size())];
    SmallVector<NamedAttribute> attributes;
    auto op = builder.create<linalg::CopyOp>(
      loc, 
      TypeRange({out.getType()}), 
      ValueRange({in}), 
      ValueRange({out}), 
      attributes);
    auto res = op.getResult(0);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator linalgMapGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int linalg_operands_num_ub = 4;

    auto point = builder.saveInsertionPoint();
    int input_num = rollIdx(linalg_operands_num_ub - 1) + 1;
    std::vector<Value> ins;
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_tensor_filter, &candidates);
    if (candidates.empty()) {
      Type t;
      if (rollIdx(2)) {
        t = randomRankedTensorType(builder.getContext());
      } else {
        t = randomRankedMemrefType(builder.getContext());
      }
      candidates.push_back(generateTypedValue(builder, loc, t, mb));
    }
    auto operand0 = candidates[rollIdx(candidates.size())];
    auto shape = operand0.getType().dyn_cast<ShapedType>();
    ins.push_back(operand0);

    SmallVector<mlir::Type> types;
    std::vector<mlir::Value> operandi_candidates = searchShapedInputFrom(shape, candidates);
    // input operands
    for (int i = 1; i < input_num; ++i) {
      auto operandi = operandi_candidates[rollIdx(operandi_candidates.size())];
      ins.push_back(operandi);
      types.push_back(shape.getElementType());
    }
    types.push_back(shape.getElementType());

    auto locs = SmallVector<Location>(input_num, loc);

    auto initType = RankedTensorType::get(shape.getShape(), shape.getElementType());
    auto init = generateTypedValue(builder, loc, initType, mb);

    auto val = builder.create<linalg::MapOp>(
      loc, TypeRange(initType), ValueRange(ins), init);
    Block *block = builder.createBlock(&val.getMapper());
    block->addArguments(TypeRange(types), locs);

    MutationBlock* childMb = new MutationBlock(*mb);
    for (auto arg : block->getArguments()) {
      childMb->add2Pool(arg);
    }
    // Create Block
    unsigned statementNum = rollIdx(32);
    for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
      OpGenerator mutator = generators[rollIdx(generators.size())];
      mutator(builder, loc, childMb);
    }
    // generate result
    builder.setInsertionPointToEnd(block);
    std::vector<mlir::Value> yieldCandidates = std::vector<mlir::Value>();
    std::string elementTypeStr = getValueTypeStr(shape.getElementType());
    if (childMb->valuePool.pool.count(elementTypeStr)) {
      yieldCandidates.insert(
        yieldCandidates.end(), 
        childMb->valuePool.pool[elementTypeStr].begin(), 
        childMb->valuePool.pool[elementTypeStr].end());
    }
    if (yieldCandidates.empty()) {
      yieldCandidates.push_back(generateElement(builder, loc, shape.getElementType()));
    }
    auto yield = yieldCandidates[rollIdx(yieldCandidates.size())];
    builder.create<linalg::YieldOp>(loc, ValueRange({yield}));

    // auto res = val.getResult(0);
    // mb->add2Pool(res);

    builder.restoreInsertionPoint(point);
    return true;
  };
}

OpGenerator linalgReduceGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int rank_ub = 3;
    const int dim_ub = 32;

    std::vector<mlir::Value> inputCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_tensor_filter_has_dim, &inputCandidates);
    if (inputCandidates.empty()) {
      auto rank = rollIdx(rank_ub) + 1; // do not generate 0-rank tensor
      SmallVector<int64_t> shape;
      for (int i = 0; i < rank; ++i) {
        shape.push_back(rollIdx(dim_ub)+1);
      }
      auto shapedTy = RankedTensorType::get(
        shape, randomIntOrFloatType(builder.getContext())).dyn_cast<ShapedType>();
      Type ty = randomMemrefOrRankedTensorType(shapedTy);
      inputCandidates.push_back(generateTypedValue(builder, loc, ty, mb));
    }
    auto operand = inputCandidates[rollIdx(inputCandidates.size())];
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    auto rank = shapedType.getRank();
    assert(rank > 0);
    SmallVector<int64_t> dimensions;

    for (int i = 0; i < rank; ++i) {
      // each dim has 1/2 prob to be reduced.
      if (!rollIdx(2)) {
        dimensions.push_back(i);
      }
    }
    if (dimensions.empty()) {
      dimensions.push_back(rollIdx(rank));
    }

    SmallVector<int64_t> reducedShape;
    int i = 0;
    int j = 0;
    for (; i < rank; ++i) {
      if (j < dimensions.size() && i == dimensions[j]) {
        j++;
        continue;
      }
      reducedShape.push_back(shapedType.getDimSize(i));
    }

    Type initTy;
    if (rollIdx(2)) {
      initTy = RankedTensorType::get(reducedShape, shapedType.getElementType());
    } else {
      initTy = MemRefType::get(reducedShape, shapedType.getElementType());
    }

    std::vector<mlir::Value> initCandidates = std::vector<mlir::Value>();
    std::string initTyStr = getValueTypeStr(initTy);
    if (mb->valuePool.pool.count(initTyStr)) {
      initCandidates.insert(
        initCandidates.end(), 
        mb->valuePool.pool[initTyStr].begin(), 
        mb->valuePool.pool[initTyStr].end());
    }
    if (initCandidates.empty()) {
      initCandidates.push_back(generateTypedValue(builder, loc, initTy, mb));
    }
    auto init = initCandidates[rollIdx(initCandidates.size())];

    SmallVector<NamedAttribute> attributes;
    auto op = builder.create<linalg::ReduceOp>(
        loc, ValueRange(operand), ValueRange(init), dimensions,
        [&](OpBuilder &b, Location l, ValueRange args) {
          MutationBlock* childMb = new MutationBlock(*mb);
          for (mlir::Value arg : args) {
            childMb->add2Pool(arg);
          }
          unsigned statementNum = rollIdx(10);
					for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
						OpGenerator mutator = generators[rollIdx(generators.size())];
						mutator(builder, loc, childMb);
					}
          assert(shapedType.getElementType().isIntOrFloat());
          std::vector<mlir::Value> yieldCandidates = std::vector<mlir::Value>();
          std::string yieldTyStr = getValueTypeStr(shapedType.getElementType());
          if (childMb->valuePool.pool.count(yieldTyStr)) {
            yieldCandidates.insert(
              yieldCandidates.end(),
              childMb->valuePool.pool[yieldTyStr].begin(), 
              childMb->valuePool.pool[yieldTyStr].end());
          }
          if (yieldCandidates.empty()) {
            yieldCandidates.push_back(
              generateTypedValue(builder, loc, shapedType.getElementType(), childMb));
          }
          auto yield = yieldCandidates[rollIdx(yieldCandidates.size())];
          builder.create<linalg::YieldOp>(loc, yield);
        },
        attributes);

    if (!op->getResults().empty()) {
      for (mlir::Value res : op->getResults()) {
        mb->add2Pool(res);
      }
    }
    return true;
  };
}

OpGenerator linalgDotGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> operand0Candidates = std::vector<mlir::Value>();
    mb->search(static_memref_tensor_filter_has_dim, &operand0Candidates);
    if (operand0Candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(rollIdx(dim_ub)+1);
      auto shapeTy = RankedTensorType::get(
                         shape, randomIntOrFloatType(builder.getContext()))
                         .dyn_cast<ShapedType>();
      operand0Candidates.push_back(
        generateTypedValue(builder, loc, randomMemrefOrRankedTensorType(shapeTy), mb));
    }
    auto operand0 = operand0Candidates[rollIdx(operand0Candidates.size())];
    
    auto operand1Ty = randomMemrefOrRankedTensorType(
      operand0.getType().dyn_cast<ShapedType>());
    std::string operand1TyStr = getValueTypeStr(operand1Ty);
    std::vector<mlir::Value> operand1Candidates = std::vector<mlir::Value>();
    if (operand1Candidates.empty()) {
      operand1Candidates.push_back(generateTypedValue(builder, loc, operand1Ty, mb));
    }
    auto operand1 = operand1Candidates[rollIdx(operand1Candidates.size())];
    
    auto resTy = RankedTensorType::get(SmallVector<int64_t>(),
        operand0.getType().dyn_cast<ShapedType>().getElementType());
    std::vector<mlir::Value> resCandidates = std::vector<mlir::Value>();
    std::string resTyStr = getValueTypeStr(resTy);
    if (mb->valuePool.pool.count(resTyStr)) {
      resCandidates.insert(
        resCandidates.begin(), 
        mb->valuePool.pool[resTyStr].begin(), 
        mb->valuePool.pool[resTyStr].end());
    }
    if (resCandidates.empty()) {
      resCandidates.push_back(generateTypedValue(builder, loc, resTy, mb));
    }
    auto res = resCandidates[rollIdx(resCandidates.size())];
    
    auto op = builder.create<linalg::DotOp>(
        loc, 
        ValueRange({operand0, operand1}), 
        ValueRange({res}), 
        SmallVector<NamedAttribute>());
    for (mlir::Value res : op->getResults()) {
      mb->add2Pool(res);
    }
    return true;
  };
}

OpGenerator linalgMatMulGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> operand1Candidate = std::vector<mlir::Value>();
    mb->search(
      [](std::string s, mlir::Value v) -> bool {
        bool cond1 = ranked_memref_tensor_filter(s, v);
        if (!cond1) { return false; }
        auto shapeTy = v.getType().dyn_cast<ShapedType>();
        return shapeTy.getRank() == 2 && !shapeTy.isDynamicDim(1);
      },
      &operand1Candidate
    );
    if (operand1Candidate.empty()) {
      auto elemTy = randomIntOrFloatType(builder.getContext());
      SmallVector<int64_t> shape;
      shape.push_back(rollIdx(dim_ub)+1);
      shape.push_back(rollIdx(dim_ub)+1);
      Type type;
      if (rollIdx(2)) {
        type = RankedTensorType::get(shape, elemTy);
      } else {
        type = MemRefType::get(shape, elemTy);
      }
      auto tVal = generateTypedValue(builder, loc, type, mb);
      operand1Candidate.push_back(tVal);
    }
    auto operand1 = operand1Candidate[rollIdx(operand1Candidate.size())];
    auto shapedType1 = operand1.getType().dyn_cast<ShapedType>();
    auto elemTy = shapedType1.getElementType();
    auto x = shapedType1.getDimSize(0);
    
    std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
    mb->search(
      [&](std::string s, mlir::Value v) -> bool {
        bool cond1 = ranked_memref_tensor_filter(s, v);
        if (!cond1) { return false; }
        auto ty = v.getType().dyn_cast<ShapedType>();
        return ty.getElementType() == elemTy && ty.getRank() == 2 &&
                ty.getDimSize(0) == shapedType1.getDimSize(1);
      },
      &operand2Candidates
    );
    if (operand2Candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(shapedType1.getDimSize(1));
      shape.push_back(rollIdx(dim_ub)+1);
      RankedTensorType shapedTy = RankedTensorType::get(shape, elemTy);
      auto type = randomMemrefOrRankedTensorType(shapedTy);
      auto tVal = generateTypedValue(builder, loc, type, mb);
      operand2Candidates.push_back(tVal);
    }
    auto operand2 = operand2Candidates[rollIdx(operand2Candidates.size())];
    auto shapedType2 = operand2.getType().dyn_cast<ShapedType>();
    auto y = shapedType2.getDimSize(1);
    auto resTensorType = RankedTensorType::get({x, y}, elemTy);
    
    std::vector<mlir::Value> initCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        bool cond1 = ranked_tensor_filter(s, v);
        if (!cond1) { return false; }
        auto shapeTy = v.getType().dyn_cast<ShapedType>();
        return shapeTy.getRank() == 2 && shapeTy.getElementType() == elemTy &&
                shapeTy.getDimSize(0) == x && shapeTy.getDimSize(1) == y;
      },
      &initCandidates
    );
    if (initCandidates.empty()) {
      auto initTy = RankedTensorType::get({x, y}, elemTy);
      initCandidates.push_back(generateRankedTensor(builder, loc, initTy, mb));
    }
    auto init = initCandidates[rollIdx(initCandidates.size())];
    auto resTy = init.getType();
    mlir::Value res = builder.create<linalg::MatmulOp>(
      loc, 
      TypeRange({resTy}),
      ValueRange({operand1, operand2}),
      ValueRange{init}).getResult(0);
    mb->add2Pool(res);
    return true;
  };
}
