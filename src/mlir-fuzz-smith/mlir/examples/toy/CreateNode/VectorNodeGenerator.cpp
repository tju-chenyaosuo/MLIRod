#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

OpGenerator vectorBroadcastGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(int_float_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(
				generateElement(builder, loc, randomIntOrFloatType(builder.getContext())));
		}
		auto operand = candidates[rollIdx(candidates.size())];
		auto vectorTy = randomVectorType(operand.getType());
		mlir::Value vector = builder.create<vector::BroadcastOp>(loc, vectorTy, operand);
		mb->add2Pool(vector);
		return true;
  };
}

OpGenerator vectorBitCastGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
				return v.getType().dyn_cast<VectorType>().getRank() > 0;
			},
			&candidates
		);
		if (candidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      candidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
		auto operand = candidates[rollIdx(candidates.size())];
		auto srcShapeTy = operand.getType().dyn_cast<ShapedType>();
    auto srcShape = srcShapeTy.getShape();
    auto srcElemTy = srcShapeTy.getElementType();
    auto srcElemTyWidth = srcElemTy.getIntOrFloatBitWidth();
    auto totalBitWidth = srcElemTyWidth;
		for (auto dimSize : srcShape) {
      totalBitWidth *= dimSize;
    }
    auto destElemTy = randomIntOrFloatType(builder.getContext());
    auto destElemTyWidth = destElemTy.getIntOrFloatBitWidth();

	  SmallVector<int64_t> destShape;
    for (auto dimSize : srcShape) {
      destShape.push_back(dimSize);
    }
    if (destElemTyWidth != srcElemTyWidth) {
      destElemTy = srcElemTy;
    }
    auto resultTy = VectorType::get(destShape, destElemTy);
    mlir::Value vector = builder.create<vector::BitCastOp>(loc, resultTy, operand);
    mb->add2Pool(vector);
    return true;
  };
}

OpGenerator vectorCompressStoreGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memrefCandidates);
		if (memrefCandidates.empty()) {
      memrefCandidates.push_back(
				generateTypedValue(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
    }
		auto memref = memrefCandidates[rollIdx(memrefCandidates.size())];
		auto memrefTy = memref.getType().dyn_cast<ShapedType>();
    auto elemTy = memrefTy.getElementType();

		std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
		mb->search(
			[&] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
				auto vType = v.getType().dyn_cast<VectorType>();
				return vType.getRank() == 1 && vType.getElementType() == elemTy;
			},
			&vectorCandidates
		);
		if (vectorCandidates.empty()) {
      auto vectorTy = random1DVectorType(elemTy);
      vectorCandidates.push_back(
				generateVector(builder, loc, vectorTy.dyn_cast<VectorType>(), mb));
    }
		auto vectorOperand = vectorCandidates[rollIdx(vectorCandidates.size())];
		auto vecTy = vectorOperand.getType().dyn_cast<VectorType>();

		std::vector<mlir::Value> maskCandidates = std::vector<mlir::Value>();
		mb->search(
			[&] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
				auto vType = v.getType().dyn_cast<VectorType>();
				return vType.getElementType() == builder.getI1Type() &&
								vType.getRank() == 1 && vType.getDimSize(0) == vecTy.getDimSize(0);
			},
			&maskCandidates
		);
		if (maskCandidates.empty()) {
      auto vectorTy = VectorType::get(
				vectorOperand.getType().dyn_cast<ShapedType>().getShape(), builder.getI1Type());
      maskCandidates.push_back(generateVector(builder, loc, vectorTy, mb));
    }
		auto maskOperand = maskCandidates[rollIdx(maskCandidates.size())];

		auto indices = randomIndicesForShapedType(memrefTy, builder, loc);
    builder.create<vector::CompressStoreOp>(loc, memref, indices, maskOperand, vectorOperand);
		return true;
  };
}

OpGenerator vectorConstantMaskGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    auto vectorTy = randomVectorType(builder.getI1Type());
    auto shape = vectorTy.dyn_cast<VectorType>().getShape();
    SmallVector<Attribute> maskDimSizes;
    for (auto dimSize : shape) {
      maskDimSizes.push_back(builder.getI64IntegerAttr(rollIdx(dimSize) + 1));
    }
    auto arrAttr = ArrayAttr::get(builder.getContext(), maskDimSizes);
    mlir::Value val = builder.create<vector::ConstantMaskOp>(loc, vectorTy, arrAttr);
		mb->add2Pool(val);
		return true;
  };
}

OpGenerator vectorCreateMaskGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    auto vectorTy = randomVectorType(builder.getI1Type());
    auto shape = vectorTy.dyn_cast<VectorType>().getShape();
    SmallVector<Attribute> maskDimSizes;
		std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
		mb->search(index_filter, &indexCandidates);
		if (indexCandidates.empty()) {
			indexCandidates.push_back(generateIndex(builder, loc, 1));
		}
    SmallVector<Value> indices;
    for (size_t i = 0; i < shape.size(); ++i) {
      indices.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
    }
    mlir::Value val = builder.create<vector::CreateMaskOp>(loc, vectorTy, indices);
		mb->add2Pool(val);
		return true;
  };
}

std::vector<vector::CombiningKind> intKinds = {
    vector::CombiningKind::ADD,   vector::CombiningKind::MUL,
    vector::CombiningKind::MINUI, vector::CombiningKind::MINSI,
    vector::CombiningKind::MAXUI, vector::CombiningKind::MAXSI,
    vector::CombiningKind::AND,   vector::CombiningKind::OR,
    vector::CombiningKind::XOR};
std::vector<vector::CombiningKind> floatKinds = {
    vector::CombiningKind::ADD, vector::CombiningKind::MUL,
    vector::CombiningKind::MINF, vector::CombiningKind::MAXF};

OpGenerator vectorContractGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> lhsCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
        if (auto t = v.getType().dyn_cast<VectorType>()) {
          return t.getRank() > 0;
        }
				return false;
			},
			&lhsCandidates
		);
		if (lhsCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      lhsCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
		auto vectorLhs = lhsCandidates[rollIdx(lhsCandidates.size())];
		auto lhsVTy = vectorLhs.getType().dyn_cast<VectorType>();
		std::vector<mlir::Value> rhsCandidates = std::vector<mlir::Value>();
		mb->search(
			[&] (std::string s, mlir::Value v) -> bool {
        if (!v.getType().dyn_cast<VectorType>()) return false;
				if (v.getType().dyn_cast<VectorType>().getElementType() != lhsVTy.getElementType()) {
            return false;
          }
          auto rhsVTy = v.getType().dyn_cast<VectorType>();
          for (int i = 0; i < lhsVTy.getRank(); i++) {
            for (int j = 0; j < rhsVTy.getRank(); ++j) {
              auto dimSizeLhs = lhsVTy.getDimSize(i);
              auto dimSizeRhs = rhsVTy.getDimSize(j);
              if (dimSizeLhs == dimSizeRhs) {
                return true;
              }
            }
          }
          return false;
			},
			&rhsCandidates
		);
		if (rhsCandidates.empty()) {
      rhsCandidates.push_back(generateVector(builder, loc, lhsVTy, mb));
    }

    auto vectorRhs = rhsCandidates[rollIdx(rhsCandidates.size())];
    auto rhsVTy = vectorRhs.getType().dyn_cast<VectorType>();

		SmallVector<AffineMap> indexingMaps;
    SmallVector<Attribute> iteratorTypes;

	  int lhsReducePos;
    int rhsReducePos;

		for (lhsReducePos = 0; lhsReducePos < lhsVTy.getRank(); lhsReducePos++) {
      bool flag = false;
      for (rhsReducePos = 0; rhsReducePos < rhsVTy.getRank(); rhsReducePos++) {
        auto dimSizeLhs = lhsVTy.getDimSize(lhsReducePos);
        auto dimSizeRhs = rhsVTy.getDimSize(rhsReducePos);
        if (dimSizeLhs == dimSizeRhs) {
          flag = true;
          break;
        }
      }
      if (flag) {
        break;
      }
    }

		SmallVector<int64_t> accShape;
    SmallVector<AffineExpr> leftExprs;
    SmallVector<AffineExpr> rightExprs;
    SmallVector<AffineExpr> accExprs;
    auto dimCount = lhsVTy.getRank() + rhsVTy.getRank();
    for (int i = 0; i < lhsReducePos; ++i) {
      auto exprI = builder.getAffineDimExpr(i);
      leftExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(lhsVTy.getDimSize(i));
      iteratorTypes.push_back(
				vector::IteratorTypeAttr::get(builder.getContext(), vector::IteratorType::parallel));
    }
    // The reduced dim
    leftExprs.push_back(builder.getAffineDimExpr(dimCount - 2));
    for (int i = lhsReducePos + 1; i < lhsVTy.getRank(); ++i) {
      // skip the reduced dim
      auto exprI = builder.getAffineDimExpr(i - 1);
      leftExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(lhsVTy.getDimSize(i));
      iteratorTypes.push_back(
				vector::IteratorTypeAttr::get(builder.getContext(), vector::IteratorType::parallel));
    }

		for (int i = 0; i < rhsReducePos; ++i) {
      auto exprI = builder.getAffineDimExpr(i + lhsVTy.getRank() - 1);
      rightExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(rhsVTy.getDimSize(i));
      iteratorTypes.push_back(
				vector::IteratorTypeAttr::get(builder.getContext(), vector::IteratorType::parallel));
    }
    // The reduced dim
    rightExprs.push_back(builder.getAffineDimExpr(dimCount - 2));
    for (int i = rhsReducePos + 1; i < rhsVTy.getRank(); ++i) {
      auto exprI = builder.getAffineDimExpr(i + lhsVTy.getRank() - 2);
      rightExprs.push_back(exprI);
      accExprs.push_back(exprI);
      accShape.push_back(rhsVTy.getDimSize(i));
      iteratorTypes.push_back(
				vector::IteratorTypeAttr::get(builder.getContext(), vector::IteratorType::parallel));
    }
    iteratorTypes.push_back(
			vector::IteratorTypeAttr::get(builder.getContext(), vector::IteratorType::reduction));
    
		indexingMaps.push_back(AffineMap::get(dimCount - 1, 0, leftExprs, builder.getContext()));
    indexingMaps.push_back(AffineMap::get(dimCount - 1, 0, rightExprs, builder.getContext()));
    indexingMaps.push_back(AffineMap::get(dimCount - 1, 0, accExprs, builder.getContext()));
    Type accTy;
    if (accExprs.empty()) {
      accTy = lhsVTy.getElementType();
    } else {
      accTy = VectorType::get(accShape, lhsVTy.getElementType());
    }

		std::vector<mlir::Value> accCandidates = std::vector<mlir::Value>();
		std::string accTyStr = getValueTypeStr(accTy);
		if (mb->valuePool.pool.count(accTyStr)) {
			accCandidates.insert(
				accCandidates.begin(), 
				mb->valuePool.pool[accTyStr].begin(), 
				mb->valuePool.pool[accTyStr].end());
		}
		if (accCandidates.empty()) {
      accCandidates.push_back(generateTypedValue(builder, loc, accTy, mb));
    }
		auto accOperand = accCandidates[rollIdx(accCandidates.size())];

		std::vector<vector::CombiningKind> kinds =
        lhsVTy.getElementType().dyn_cast<IntegerType>() ? intKinds : floatKinds;
    auto kind = kinds[rollIdx(kinds.size())];
    auto indexingMapAttr = builder.getAffineMapArrayAttr(indexingMaps);
    auto iteratorTypesAttr = builder.getArrayAttr(iteratorTypes);

	  builder.create<vector::ContractionOp>(
			loc, vectorLhs, vectorRhs, accOperand, indexingMapAttr, iteratorTypesAttr, kind);
		return true;
  };
}

OpGenerator vectorExpandLoadGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memrefCandidates);
		if (memrefCandidates.empty()) {
      memrefCandidates.push_back(
				generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
    }
		auto base = memrefCandidates[rollIdx(memrefCandidates.size())];
		auto elemTy = base.getType().dyn_cast<MemRefType>().getElementType();
		std::vector<mlir::Value> passThruCandidates = std::vector<mlir::Value>();
		mb->search(
			[&] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
				return v.getType().dyn_cast<VectorType>().getRank() == 1 && 
					v.getType().dyn_cast<VectorType>().getElementType() == 
					base.getType().dyn_cast<MemRefType>().getElementType();
			},
			&passThruCandidates
		);
		if (passThruCandidates.empty()) {
      auto passThruTy = random1DVectorType(elemTy);
      passThruCandidates.push_back(generateVector(builder, loc, passThruTy, mb));
    }
		auto passThruVector = passThruCandidates[rollIdx(passThruCandidates.size())];
		auto resultTy = passThruVector.getType();
		auto maskTy = VectorType::get(
			resultTy.dyn_cast<VectorType>().getShape(), builder.getI1Type());
		std::vector<mlir::Value> maskCandidates = std::vector<mlir::Value>();
		std::string maskTyStr = getValueTypeStr(maskTy);
		if (mb->valuePool.pool.count(maskTyStr)) {
			maskCandidates.insert(
				maskCandidates.begin(), 
				mb->valuePool.pool[maskTyStr].begin(), 
				mb->valuePool.pool[maskTyStr].end());
		}
		if (maskCandidates.empty()) {
      maskCandidates.push_back(generateVector(builder, loc, maskTy, mb));
    }
		auto maskVector = maskCandidates[rollIdx(maskCandidates.size())];

		SmallVector<Value> indices;
		std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
    mb->search(index_filter, &indexCandidates);
    if (indexCandidates.empty()) {
      indexCandidates.push_back(generateIndex(builder, loc, 1));
    }
    for (int i = 0; i < base.getType().dyn_cast<MemRefType>().getRank(); ++i) {
      indices.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
    }
    mlir::Value val = builder.create<vector::ExpandLoadOp>(
      loc, resultTy, base, indices, maskVector, passThruVector);
    mb->add2Pool(val);
    return true;
  };
}

OpGenerator vectorExtractGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() > 0;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
    auto vectorOperand = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vectorTy = vectorOperand.getType().dyn_cast<VectorType>();
    SmallVector<int64_t> indices;
    auto indicesSize = max(1, rollIdx(vectorTy.getRank())); // at least extract 1 dim.
    for (int i = 0; i < indicesSize; ++i) {
      indices.push_back(rollIdx(vectorTy.getDimSize(i)));
    }
    auto arrAttr = vector::getVectorSubscriptAttr(builder, indices);
    Type resultTy;
    if ((int)indices.size() == vectorTy.getRank()) {
      resultTy = vectorTy.getElementType();
    } else {
      auto n = std::min<size_t>(indicesSize, vectorTy.getRank() - 1);
      resultTy = VectorType::get(vectorTy.getShape().drop_front(n),
                                 vectorTy.getElementType());
    }

    mlir::Value res = builder.create<vector::ExtractOp>(loc, vectorOperand, arrAttr);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorExtractElementGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() <= 1;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      if (rollIdx(2)) {
        shape.push_back(rollIdx(dim_ub) + 1);
      }
      vectorCandidates.push_back(
        generateVector(
          builder, loc, VectorType::get(shape, randomIntOrFloatType(builder.getContext())), mb));
    }
    auto vector = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vectorTy = vector.getType().dyn_cast<VectorType>();
    mlir::Value res;
    if (vectorTy.getRank() == 0) {
      res = builder.create<vector::ExtractElementOp>(loc, vector);
    } else {
      std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
      mb->search(index_filter, &idxCandidates);
      if (idxCandidates.empty()) {
        idxCandidates.push_back(generateIndex(builder, loc, rollIdx(100)));
      }

      auto position = idxCandidates[rollIdx(idxCandidates.size())];
      res = builder.create<vector::ExtractElementOp>(loc, vector, position);
    }
    mb->add2Pool(res);
    return true;
  };
}

// Inference works as follows:
//   1. Add 'sizes' from prefix of dims in 'offsets'.
//   2. Add sizes from 'vectorType' for remaining dims.
static Type inferStridedSliceOpResultType(VectorType vectorType,
                                          SmallVector<int64_t> offsets,
                                          SmallVector<int64_t> sizes,
                                          SmallVector<int64_t> strides) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  SmallVector<int64_t, 4> shape;
  shape.reserve(vectorType.getRank());
  unsigned idx = 0;
  for (unsigned e = offsets.size(); idx < e; ++idx)
    shape.push_back(sizes[idx]);
  for (unsigned e = vectorType.getShape().size(); idx < e; ++idx)
    shape.push_back(vectorType.getShape()[idx]);

  return VectorType::get(shape, vectorType.getElementType());
}

OpGenerator vectorExtractStridedSliceGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() > 0;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
    auto srcVector = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vectorTy = srcVector.getType().dyn_cast<VectorType>();
    auto sliceDimSize = max(1, rollIdx(vectorTy.getRank()));
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
    for (int i = 0; i < sliceDimSize; ++i) {
      int64_t offset = rollIdx(max(vectorTy.getDimSize(i) - 1, 1));
      offsets.push_back(offset);
      int64_t size = rollIdx(vectorTy.getDimSize(i) - offset) + 1;
      sizes.push_back(size);
      strides.push_back(1);
    }
    auto resultTy = inferStridedSliceOpResultType(vectorTy, offsets, sizes, strides);
    mlir::Value res = builder.create<vector::ExtractStridedSliceOp>(
      loc, srcVector, offsets, sizes, strides);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorFMAGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(vector_filter, &vectorCandidates);
    if (vectorCandidates.empty()) {
      vectorCandidates.push_back(
        generateVector(builder, loc, randomVectorType(builder.getContext()), mb));
    }
    std::vector<mlir::Value> lhsCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().isa<FloatType>();
      },
      &lhsCandidates
    );
    if (lhsCandidates.empty()) {
      lhsCandidates.push_back(
        generateVector(
          builder, loc, randomVectorType(FloatType::getF32(builder.getContext())), mb));
    }
    auto lhsOperand = lhsCandidates[rollIdx(lhsCandidates.size())];
    auto lhsVTy = lhsOperand.getType().dyn_cast<VectorType>();
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    std::string lhsVTyStr = getValueTypeStr(lhsVTy);
    if (mb->valuePool.pool.count(lhsVTyStr)) {
      candidates.insert(
        candidates.begin(), 
        mb->valuePool.pool[lhsVTyStr].begin(), 
        mb->valuePool.pool[lhsVTyStr].end());
    }
    if (candidates.empty()) {
      candidates.push_back(generateTypedValue(builder, loc, lhsVTy, mb));
    }
    auto rhsOperand = candidates[rollIdx(candidates.size())];
    auto accOperand = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<vector::FMAOp>(loc, lhsOperand, rhsOperand, accOperand);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorFlatTransposeGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() == 1;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      vectorCandidates.push_back(
        generateTypedValue(builder, loc, random1DVectorType(builder.getContext()), mb));
    }
    auto source = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vectorTy = source.getType().dyn_cast<VectorType>();
    auto size = vectorTy.getDimSize(0);
    int mid = (int)ceil(sqrt(size));
    int factor;
    for (factor = mid; factor > 0; factor--) {
      if (size % factor == 0) {
        break;
      }
    }
    auto rows = IntegerAttr::get(builder.getI32Type(), factor);
    auto columns = IntegerAttr::get(builder.getI32Type(), size / factor);
    mlir::Value val = builder.create<vector::FlatTransposeOp>(
      loc, vectorTy, source, rows, columns);
    mb->add2Pool(val);
    return true;
  };
}

OpGenerator vectorGatherGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    auto baseTy = randomMemrefOrRankedTensorType(builder.getContext()).dyn_cast<ShapedType>();
    auto baseElemTy = baseTy.getElementType();
    std::vector<mlir::Value> baseCandidates = std::vector<mlir::Value>();
    std::string baseTyStr = getValueTypeStr(baseTy);
    if (mb->valuePool.pool.count(baseTyStr)) {
      baseCandidates.insert(
        baseCandidates.end(), 
        mb->valuePool.pool[baseTyStr].begin(), 
        mb->valuePool.pool[baseTyStr].end());
    }
    if (baseCandidates.empty()) {
      baseCandidates.push_back(generateTypedValue(builder, loc, baseTy, mb));
    }
    mlir::Value base = baseCandidates[rollIdx(baseCandidates.size())];

    auto resTy = randomVectorType(baseElemTy);
    auto passThruTy = resTy;

    std::vector<mlir::Value> passThruCandidates = std::vector<mlir::Value>();
    std::string passThruTyStr = getValueTypeStr(passThruTy);
    if (mb->valuePool.pool.count(passThruTyStr)) {
      passThruCandidates.insert(
        passThruCandidates.end(), 
        mb->valuePool.pool[passThruTyStr].begin(), 
        mb->valuePool.pool[passThruTyStr].end());
    }
    if (passThruCandidates.empty()) {
      passThruCandidates.push_back(generateVector(builder, loc, passThruTy, mb));
    }
    auto passThru = passThruCandidates[rollIdx(passThruCandidates.size())];

    auto maskTy = VectorType::get(passThruTy.getShape(), builder.getI1Type());
    std::vector<mlir::Value> maskCandidates = std::vector<mlir::Value>();
    std::string maskTyStr = getValueTypeStr(maskTy);
    if (mb->valuePool.pool.count(maskTyStr)) {
      maskCandidates.insert(
        maskCandidates.end(), 
        mb->valuePool.pool[maskTyStr].begin(), 
        mb->valuePool.pool[maskTyStr].end());
    }
    if (maskCandidates.empty()) {
      maskCandidates.push_back(generateVector(builder, loc, maskTy, mb));
    }
    auto mask = maskCandidates[rollIdx(maskCandidates.size())];

    auto indexVecTy = VectorType::get(passThruTy.getShape(), builder.getI32Type());
    std::vector<mlir::Value> indexVecCandidates = std::vector<mlir::Value>();
    std::string indexVecTyStr = getValueTypeStr(indexVecTy);
    if (mb->valuePool.pool.count(indexVecTyStr)) {
      indexVecCandidates.insert(
        indexVecCandidates.end(), 
        mb->valuePool.pool[indexVecTyStr].begin(), 
        mb->valuePool.pool[indexVecTyStr].end());
    }
    if (indexVecCandidates.empty()) {
      indexVecCandidates.push_back(generateVector(builder, loc, indexVecTy, mb));
    }
    auto indexVec = indexVecCandidates[rollIdx(indexVecCandidates.size())];

    std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
    mb->search(index_filter, &idxCandidates);
    if (idxCandidates.empty()) {
      idxCandidates.push_back(generateIndex(builder, loc, 1));
    }
    SmallVector<Value> indices;
    for (int i = 0; i < baseTy.getRank(); ++i) {
      indices.push_back(idxCandidates[rollIdx(idxCandidates.size())]);
    }
    mlir::Value res = builder.create<vector::GatherOp>(
      loc, resTy, base, indices, indexVec, mask, passThru);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorInsertElementGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() <= 1;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      if (rollIdx(2)) {
        shape.push_back(rollIdx(dim_ub) + 1);
      }
      vectorCandidates.push_back(
        generateVector(
          builder, 
          loc, 
          VectorType::get(shape, randomIntOrFloatType(builder.getContext())), mb));
    }
    auto vector = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vectorTy = vector.getType().dyn_cast<VectorType>();

    std::vector<mlir::Value> elemCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!int_float_filter(s)) return false;
        return v.getType() == vectorTy.getElementType();
      },
      &elemCandidates
    );
    if (elemCandidates.empty()) {
      elemCandidates.push_back(
        generateElement(builder, loc, vectorTy.getElementType()));
    }
    auto source = elemCandidates[rollIdx(elemCandidates.size())];
    if (vectorTy.getRank() == 0) {
      return builder.create<vector::InsertElementOp>(loc, source, vector);
    } else {
      std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
      mb->search(index_filter, &idxCandidates);
      if (idxCandidates.empty()) {
        idxCandidates.push_back(generateIndex(builder, loc, 1));
      }
      auto position = idxCandidates[rollIdx(idxCandidates.size())];
      builder.create<vector::InsertElementOp>(loc, source, vector, position);
    }
    return true;
  };
}

OpGenerator vectorInsertGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() > 0;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
    auto vectorOperand = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vectorTy = vectorOperand.getType().dyn_cast<VectorType>();
    SmallVector<int64_t> indices;
    auto indicesSize = max(rollIdx(vectorTy.getRank()), 1); // at least extract 1 dim.
    for (int i = 0; i < indicesSize; ++i) {
      indices.push_back(rollIdx(vectorTy.getDimSize(i)));
    }
    Type sourceTy;
    if ((int)indices.size() == vectorTy.getRank()) {
      sourceTy = vectorTy.getElementType();
    } else {
      auto n = std::min<size_t>(indicesSize, vectorTy.getRank() - 1);
      sourceTy = VectorType::get(vectorTy.getShape().drop_front(n),
                                 vectorTy.getElementType());
    }


    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    std::string sourceTyStr = getValueTypeStr(sourceTy);
    if (mb->valuePool.pool.count(sourceTyStr)) {
      srcCandidates.insert(
        srcCandidates.end(), 
        mb->valuePool.pool[sourceTyStr].begin(), 
        mb->valuePool.pool[sourceTyStr].end());
    }
    if (srcCandidates.empty()) {
      srcCandidates.push_back(generateTypedValue(builder, loc, sourceTy, mb));
    }
    auto src = srcCandidates[rollIdx(srcCandidates.size())];
    builder.create<vector::InsertOp>(loc, src, vectorOperand, indices);
    return true;
  };
}

OpGenerator vectorInsertStridedSliceGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() > 0;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
    auto destVector = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vectorTy = destVector.getType().dyn_cast<VectorType>();
    auto sliceDimSize = rollIdx(vectorTy.getRank()) + 1;
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
    for (int i = 0; i < sliceDimSize; ++i) {
      int64_t offset = rollIdx(max(vectorTy.getDimSize(i) - 1, 1));
      offsets.push_back(offset);
      int64_t size = rollIdx(vectorTy.getDimSize(i) - offset) + 1;
      sizes.push_back(size);
      strides.push_back(1);
    }
    for (int i = sliceDimSize; i < vectorTy.getRank(); ++i) {
      offsets.push_back(1);
    }
    auto srcTy = VectorType::get(sizes, vectorTy.getElementType());
    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    std::string srcTyStr = getValueTypeStr(srcTy);
    if (mb->valuePool.pool.count(srcTyStr)) {
      srcCandidates.insert(
        srcCandidates.end(), 
        mb->valuePool.pool[srcTyStr].begin(), 
        mb->valuePool.pool[srcTyStr].end());
    }
    if (srcCandidates.empty()) {
      srcCandidates.push_back(
        generateVector(builder, loc, srcTy.dyn_cast<VectorType>(), mb));
    }
    auto srcVector = srcCandidates[rollIdx(srcCandidates.size())];
    builder.create<vector::InsertStridedSliceOp>(
      loc, srcVector, destVector, offsets, strides);
    return true;
  };
}

OpGenerator vectorLoadGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_filter, &memrefCandidates);
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(
        generateRankedMemref(
          builder, 
          loc, 
          randomRankedMemrefType(builder.getContext()), 
          mb));
    }
    auto memref = memrefCandidates[rollIdx(memrefCandidates.size())];
    auto memrefTy = memref.getType().dyn_cast<ShapedType>();
    auto indices = randomIndicesForShapedType(memrefTy, builder, loc);
    auto vectorTy = randomVectorType(memrefTy.getElementType());
    mlir::Value res = builder.create<vector::LoadOp>(loc, vectorTy, memref, indices);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorMaskGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    MutationBlock* maskOpBlock = new MutationBlock(*mb);
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() > 0;
      }, 
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
    auto operand = vectorCandidates[rollIdx(vectorCandidates.size())];
    Operation *maskableOp;
    if (true) {
      auto vTy = operand.getType().dyn_cast<VectorType>();
      SmallVector<bool> reductionMask;
      SmallVector<int64_t> newShape;
      reductionMask.push_back(false);
      newShape.push_back(vTy.getDimSize(0));
      for (int i = 1; i < vTy.getRank(); ++i) {
        if (rollIdx(2)) {
          reductionMask.push_back(true);
        } else {
          reductionMask.push_back(false);
          newShape.push_back(vTy.getDimSize(i));
        }
      }
      auto accType = newShape.empty() ? vTy.getElementType()
              : (Type)VectorType::get(newShape, vTy.getElementType());
      std::vector<mlir::Value> accCandidates = std::vector<mlir::Value>();
      std::string accTypeStr = getValueTypeStr(accType);
      if (mb->valuePool.pool.count(accTypeStr)) {
        accCandidates.insert(
          accCandidates.end(), 
          mb->valuePool.pool[accTypeStr].begin(), 
          mb->valuePool.pool[accTypeStr].end());
      }
      if (accCandidates.empty()) {
        accCandidates.push_back(generateTypedValue(builder, loc, accType, mb));
      }
      auto acc = accCandidates[rollIdx(accCandidates.size())];
      std::vector<vector::CombiningKind> kinds =
          vTy.getElementType().dyn_cast<IntegerType>() ? intKinds : floatKinds;
      auto kind = kinds[rollIdx(kinds.size())];
      maskableOp = builder.create<vector::MultiDimReductionOp>(
        loc, operand, acc, reductionMask, kind).getOperation();
    }
    auto regionBuilder = [](OpBuilder &b, Operation *maskableOp) {
      vector::createMaskOpRegion(b, maskableOp);
    };
    auto maskTy = VectorType::get(
      operand.getType().dyn_cast<ShapedType>().getShape(), builder.getI1Type());
    std::vector<mlir::Value> maskCandidates = std::vector<mlir::Value>();
    std::string maskTypeStr = getValueTypeStr(maskTy);
    if (mb->valuePool.pool.count(maskTypeStr)) {
      maskCandidates.insert(
        maskCandidates.end(), 
        mb->valuePool.pool[maskTypeStr].begin(), 
        mb->valuePool.pool[maskTypeStr].end());
    }
    if (maskCandidates.empty()) {
      maskCandidates.push_back(generateVector(builder, loc, maskTy, mb));
    }
    auto mask = maskCandidates[rollIdx(maskCandidates.size())];

    llvm::outs() << "before create\n";
    
    Value res = builder.create<vector::MaskOp>(
      loc, 
      TypeRange({maskableOp->getResult(0).getType()}), 
      mask, 
      maskableOp, 
      regionBuilder).getResult(0);
    llvm::outs() << "after create\n";
    return true;
  };
}

OpGenerator vectorMaskedLoadGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_filter, &memrefCandidates);
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(
        generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
    }
    auto memref = memrefCandidates[rollIdx(memrefCandidates.size())];
    auto memrefTy = memref.getType().dyn_cast<ShapedType>();
    auto indices = randomIndicesForShapedType(memrefTy, builder, loc);

    auto vectorTy = random1DVectorType(memrefTy.getElementType());

    std::vector<mlir::Value> passThruCandidates = std::vector<mlir::Value>();
    std::string vectorTyStr = getValueTypeStr(vectorTy);
    if (mb->valuePool.pool.count(vectorTyStr)) {
      passThruCandidates.insert(
        passThruCandidates.end(), 
        mb->valuePool.pool[vectorTyStr].begin(), 
        mb->valuePool.pool[vectorTyStr].end());
    }
    if (passThruCandidates.empty()) {
      passThruCandidates.push_back(generateVector(builder, loc, vectorTy, mb));
    }
    auto passThru = passThruCandidates[rollIdx(passThruCandidates.size())];

    auto maskTy = VectorType::get(vectorTy.getShape(), builder.getI1Type());
    std::vector<mlir::Value> maskCandidates = std::vector<mlir::Value>();
    std::string maskTyStr = getValueTypeStr(maskTy);
    if (mb->valuePool.pool.count(maskTyStr)) {
      maskCandidates.insert(
        maskCandidates.begin(), 
        mb->valuePool.pool[maskTyStr].begin(), 
        mb->valuePool.pool[maskTyStr].end());
    }
    if (maskCandidates.empty()) {
      maskCandidates.push_back(generateVector(builder, loc, maskTy, mb));
    }
    auto mask = maskCandidates[rollIdx(maskCandidates.size())];

    mlir::Value res = builder.create<vector::MaskedLoadOp>(
      loc, vectorTy, memref, indices, mask, passThru);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorStoreGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_filter, &memrefCandidates);
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(
        generateRankedMemref(builder, loc, randomRankedMemrefType(builder.getContext()), mb));
    }
    auto memref = memrefCandidates[rollIdx(memrefCandidates.size())];
    auto memrefTy = memref.getType().dyn_cast<ShapedType>();
    auto indices = randomIndicesForShapedType(memrefTy, builder, loc);
    auto vectorTy = randomVectorType(memrefTy.getElementType());

    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    std::string vectorTyStr = getValueTypeStr(vectorTy);
    if (mb->valuePool.pool.count(vectorTyStr)) {
      srcCandidates.insert(
        srcCandidates.begin(), 
        mb->valuePool.pool[vectorTyStr].begin(), 
        mb->valuePool.pool[vectorTyStr].end());
    }
    if (srcCandidates.empty()) {
      srcCandidates.push_back(generateVector(builder, loc, vectorTy, mb));
    }
    auto src = srcCandidates[rollIdx(srcCandidates.size())];
    builder.create<vector::StoreOp>(loc, src, memref, indices);
    return true;
  };
}

OpGenerator vectorMaskedStoreGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_filter, &memrefCandidates);
    if (memrefCandidates.empty()) {
      memrefCandidates.push_back(
        generateRankedMemref(
          builder, 
          loc, 
          randomRankedMemrefType(builder.getContext()),
          mb));
    }
    auto memref = memrefCandidates[rollIdx(memrefCandidates.size())];
    auto memrefTy = memref.getType().dyn_cast<ShapedType>();
    auto indices = randomIndicesForShapedType(memrefTy, builder, loc);
    auto vectorTy = random1DVectorType(memrefTy.getElementType());

    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    std::string vectorTyStr = getValueTypeStr(vectorTy);
    if (mb->valuePool.pool.count(vectorTyStr)) {
      srcCandidates.insert(
        srcCandidates.begin(), 
        mb->valuePool.pool[vectorTyStr].begin(), 
        mb->valuePool.pool[vectorTyStr].end());
    }
    if (srcCandidates.empty()) {
      srcCandidates.push_back(generateVector(builder, loc, vectorTy, mb));
    }
    auto src = srcCandidates[rollIdx(srcCandidates.size())];

    auto maskTy = VectorType::get(vectorTy.getShape(), builder.getI1Type());
    std::vector<mlir::Value> maskCandidates = std::vector<mlir::Value>();
    std::string maskTyStr = getValueTypeStr(maskTy);
    if (mb->valuePool.pool.count(maskTyStr)) {
      maskCandidates.insert(
        maskCandidates.begin(), 
        mb->valuePool.pool[maskTyStr].begin(), 
        mb->valuePool.pool[maskTyStr].end());
    }
    if (maskCandidates.empty()) {
      maskCandidates.push_back(generateVector(builder, loc, maskTy, mb));
    }
    auto mask = maskCandidates[rollIdx(maskCandidates.size())];
    builder.create<vector::MaskedStoreOp>(loc, memref, indices, mask, src);
    return true;
  };
}

int gcd(int a, int b) {
  int m = a > b ? a : b;
  int n = a < b ? a : b;

  int r = m % n;
  while (r != 0) {
    m = n;
    n = r;
    r = m % n;
  }
  return n;
}

OpGenerator vectorMatrixMultiplyGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> lhsCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() == 1;
      },
      &lhsCandidates
    );
    if (lhsCandidates.empty()) {
      lhsCandidates.push_back(
        generateVector(builder, loc, random1DVectorType(builder.getContext()), mb));
    }
    auto lhs = lhsCandidates[rollIdx(lhsCandidates.size())];
    auto lhsVTy = lhs.getType().dyn_cast<VectorType>();

    std::vector<mlir::Value> rhsCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        auto vTy = v.getType().dyn_cast<VectorType>();
        return vTy.getRank() == 1 && 
          vTy.getElementType() == lhsVTy.getElementType();
      },
      &rhsCandidates
    );
    if (rhsCandidates.empty()) {
      rhsCandidates.push_back(
        generateVector(builder, loc, random1DVectorType(lhsVTy.getElementType()), mb));
    }
    auto rhs = rhsCandidates[rollIdx(rhsCandidates.size())];
    auto rhsVTy = rhs.getType().dyn_cast<VectorType>();

    auto reducedDimSize = gcd(lhsVTy.getDimSize(0), rhsVTy.getDimSize(0));

    SmallVector<int64_t> resShape;
    resShape.push_back(lhsVTy.getDimSize(0) / reducedDimSize *
                       rhsVTy.getDimSize(0) / reducedDimSize);

    auto resVTy = VectorType::get(resShape, lhsVTy.getElementType());
    mlir::Value val = builder.create<vector::MatmulOp>(
      loc, 
      lhs, 
      rhs, 
      lhsVTy.getDimSize(0) / reducedDimSize,
      reducedDimSize, 
      rhsVTy.getDimSize(0) / reducedDimSize);
    mb->add2Pool(val);
    return true;
  };
}

OpGenerator vectorMultiReductionGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() > 0;
      },
      &vectorCandidates
    );
    if (vectorCandidates.empty()) {
      SmallVector<int64_t> shape;
      shape.push_back(1);
      auto elemTy = randomIntOrFloatType(builder.getContext());
      auto vecTy = VectorType::get(shape, elemTy);
      vectorCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
    mlir::Value operand = vectorCandidates[rollIdx(vectorCandidates.size())];
    auto vTy = operand.getType().dyn_cast<VectorType>();
    SmallVector<bool> reductionMask;
    SmallVector<int64_t> newShape;
    for (int i = 0; i < vTy.getRank(); ++i) {
      if (rollIdx(2)) {
        reductionMask.push_back(true);
      } else {
        reductionMask.push_back(false);
        newShape.push_back(vTy.getDimSize(i));
      }
    }
    auto accType = newShape.empty()
                       ? vTy.getElementType()
                       : (Type)VectorType::get(newShape, vTy.getElementType());
    std::string accTypeStr = getValueTypeStr(accType);
    std::vector<mlir::Value> accCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(accTypeStr)) {
      accCandidates.insert(
        accCandidates.begin(), 
        mb->valuePool.pool[accTypeStr].begin(), 
        mb->valuePool.pool[accTypeStr].end());
    }
    if (accCandidates.empty()) {
      accCandidates.push_back(generateTypedValue(builder, loc, accType, mb));
    }
    mlir::Value acc = accCandidates[rollIdx(accCandidates.size())];
    
    std::vector<vector::CombiningKind> kinds =
        vTy.getElementType().dyn_cast<IntegerType>() ? intKinds : floatKinds;
    auto kind = kinds[rollIdx(kinds.size())];
    mlir::Value val = builder.create<vector::MultiDimReductionOp>(
        loc, operand, acc, reductionMask, kind);
    mb->add2Pool(val);
    return true;
  };
}

// OpGenerator vectorOuterProductGenerator() {
//   return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
//     std::vector<mlir::Value> lhsCandidates = std::vector<mlir::Value>();
//     mb->search(
//       [] (std::string s, mlir::Value v) -> bool {
//         if (!vector_filter(s)) return false;
//         return v.getType().dyn_cast<VectorType>().getRank() == 1;
//       },
//       &lhsCandidates
//     );
//      if (lhsCandidates.empty()) {
//       lhsCandidates.push_back(
//         generateVector(builder, loc, random1DVectorType(builder.getContext()), mb));
//     }
//     auto lhs = lhsCandidates[rollIdx(lhsCandidates.size())];
//     auto lhsElemTy = lhs.getType().dyn_cast<VectorType>().getElementType();
//     std::string lhsTyStr = getValueTypeStr(lhs);
//     std::vector<mlir::Value> rhsCandidates = std::vector<mlir::Value>();
//     if (mb->valuePool.pool.count(lhsTyStr)) {
//       rhsCandidates.insert(
//         rhsCandidates.begin(), 
//         mb->valuePool.pool[lhsTyStr].begin(), 
//         mb->valuePool.pool[lhsTyStr].end());
//     }
//     if (rhsCandidates.empty()) {
//       rhsCandidates.push_back(
//         generateVector(builder, loc, random1DVectorType(lhsElemTy), mb));
//     }
//     auto rhs = rhsCandidates[rollIdx(rhsCandidates.size())];

//     SmallVector<int64_t> accShape;
//     accShape.push_back(lhs.getType().dyn_cast<VectorType>().getDimSize(0));
//     accShape.push_back(rhs.getType().dyn_cast<VectorType>().getDimSize(0));
//     auto accTy = VectorType::get(accShape, lhsElemTy);
//     std::string accTyStr = getValueTypeStr(accTy);
//     std::vector<mlir::Value> accCandidates = std::vector<mlir::Value>();
//     if (mb->valuePool.pool.count(accTyStr)) {
//       accCandidates.insert(
//         accCandidates.begin(), 
//         mb->valuePool.pool[accTyStr].begin(), 
//         mb->valuePool.pool[accTyStr].end());
//     }
//     if (accCandidates.empty()) {
//       accCandidates.push_back(generateVector(builder, loc, accTy, mb));
//     }
//     auto acc = accCandidates[rollIdx(accCandidates.size())];

//     std::vector<vector::CombiningKind> kinds =
//         lhsElemTy.dyn_cast<IntegerType>() ? intKinds : floatKinds;
//     auto kind = kinds[rollIdx(kinds.size())];

//     mlir::Value res = builder.create<vector::OuterProductOp>(loc, acc, lhs,rhs, acc, kind);

//     mb->add2Pool(res);
//     return true;
//   };
// }

OpGenerator vectorPrintGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    mb->search(vector_filter, &vectorCandidates);
    if (vectorCandidates.empty()) {
      vectorCandidates.push_back(
        generateVector(
          builder, 
          loc, 
          randomVectorType(randomIntOrFloatType(builder.getContext())), 
          mb));
    }
    auto operand = vectorCandidates[rollIdx(vectorCandidates.size())];
    builder.create<vector::PrintOp>(loc, operand);
    return true;
  };
}

OpGenerator vectorReductionGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() == 1;
      },
      &candidates
    );
    if (candidates.empty()) {
      candidates.push_back(
        generateVector(builder, loc, random1DVectorType(builder.getContext()), mb));
    }
    auto src = candidates[rollIdx(candidates.size())];

    std::vector<vector::CombiningKind> kinds =
        src.getType().dyn_cast<VectorType>().getElementType().dyn_cast<IntegerType>()
            ? intKinds
            : floatKinds;
    auto kind = kinds[rollIdx(kinds.size())];

    auto resTy = src.getType().dyn_cast<VectorType>().getElementType();
    mlir::Value res = builder.create<vector::ReductionOp>(loc, kind, src);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorScanGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;
    const int rank_ub = 3;
    
    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return v.getType().dyn_cast<VectorType>().getRank() > 1;
      },
      &srcCandidates
    );
    if (srcCandidates.empty()) {
      int rank = max(rollIdx(rank_ub) + 1, 2);
      SmallVector<int64_t> shape;
      for (int i = 0; i < rank; ++i) {
        shape.push_back(rollIdx(dim_ub) + 1);
      }
      auto vTy = VectorType::get(shape, randomIntOrFloatType(builder.getContext()));
      srcCandidates.push_back(generateVector(builder, loc, vTy, mb));
    }
    auto src = srcCandidates[rollIdx(srcCandidates.size())];
    auto srcVTy = src.getType().dyn_cast<VectorType>();

    std::vector<vector::CombiningKind> kinds =
        srcVTy.getElementType().dyn_cast<IntegerType>() ? intKinds : floatKinds;
    auto kind = kinds[rollIdx(kinds.size())];

    SmallVector<int64_t> accShape;
    int64_t reduction_dim = rollIdx(srcVTy.getRank());
    for (int i = 0; i < srcVTy.getRank(); ++i) {
      if (i == reduction_dim) {
        continue;
      }
      accShape.push_back(srcVTy.getDimSize(i));
    }
    auto accTy = VectorType::get(accShape, srcVTy.getElementType());
    std::string accTyStr = getValueTypeStr(accTy);
    std::vector<mlir::Value> accCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(accTyStr)) {
      accCandidates.insert(
        accCandidates.begin(), 
        mb->valuePool.pool[accTyStr].begin(), 
        mb->valuePool.pool[accTyStr].end());
    }
    if (accCandidates.empty()) {
      accCandidates.push_back(generateTypedValue(builder, loc, accTy, mb));
    }
    auto init = accCandidates[rollIdx(accCandidates.size())];

    bool inclusive = rollIdx(2);
    auto results = builder.create<vector::ScanOp>(
      loc, kind, src, init, reduction_dim, inclusive);
    mlir::Value res0 = results.getResult(0);
    mlir::Value res1 = results.getResult(1);
    mb->add2Pool(res0);
    mb->add2Pool(res1);
    return true;
  };
}

OpGenerator vectorScatterGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> baseCandidates = std::vector<mlir::Value>();
    mb->search(ranked_memref_filter, &baseCandidates);
    if (baseCandidates.empty()) {
      baseCandidates.push_back(
        generateRankedMemref(
          builder, 
          loc, 
          randomRankedMemrefType(builder.getContext()),
          mb));
    }
    auto base = baseCandidates[rollIdx(baseCandidates.size())];
    auto indices = randomIndicesForShapedType(
      base.getType().dyn_cast<ShapedType>(), builder, loc);
    
    auto indexVTy = random1DVectorType(builder.getIndexType());
    std::string indexVTyStr = getValueTypeStr(indexVTy);
    std::vector<mlir::Value> indexVecCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(indexVTyStr)) {
      indexVecCandidates.insert(
        indexVecCandidates.begin(), 
        mb->valuePool.pool[indexVTyStr].begin(), 
        mb->valuePool.pool[indexVTyStr].end());
    }
    if (indexVecCandidates.empty()) {
      indexVecCandidates.push_back(generateVector(builder, loc, indexVTy, mb));
    }
    auto indexVec = indexVecCandidates[rollIdx(indexVecCandidates.size())];

    auto maskVTy = VectorType::get(indexVTy.getShape(), builder.getI1Type());
    std::string maskVTyStr = getValueTypeStr(maskVTy);
    std::vector<mlir::Value> maskCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(maskVTyStr)) {
      maskCandidates.insert(
        maskCandidates.begin(), 
        mb->valuePool.pool[maskVTyStr].begin(), 
        mb->valuePool.pool[maskVTyStr].end());
    }
    if (maskCandidates.empty()) {
      maskCandidates.push_back(generateVector(builder, loc, maskVTy, mb));
    }
    auto mask = maskCandidates[rollIdx(maskCandidates.size())];
    auto vecToStoreTy = VectorType::get(
      indexVTy.getShape(), 
      base.getType().dyn_cast<MemRefType>().getElementType());
    std::string vecToStoreTyStr = getValueTypeStr(vecToStoreTy);
    std::vector<mlir::Value> vecToStoreCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(vecToStoreTyStr)) {
      vecToStoreCandidates.insert(
        vecToStoreCandidates.begin(), 
        mb->valuePool.pool[vecToStoreTyStr].begin(), 
        mb->valuePool.pool[vecToStoreTyStr].end());
    }
    if (vecToStoreCandidates.empty()) {
      vecToStoreCandidates.push_back(generateVector(builder, loc, vecToStoreTy, mb));
    }
    auto vecToStore = vecToStoreCandidates[rollIdx(vecToStoreCandidates.size())];
    builder.create<vector::ScatterOp>(
      loc, base, ValueRange(indices),indexVec, mask, vecToStore);
    return true;
  };
}

bool hasSameTrailingShape(VectorType v1, VectorType v2) {
  if (v1.getRank() != v2.getRank()) {
    return false;
  }
  if (v1.getElementType() != v2.getElementType()) {
    return false;
  }
  for (int i = 1; i < v1.getRank(); ++i) {
    if (v1.getDimSize(i) != v2.getDimSize(i)) {
      return false;
    }
  }
  return true;
}

OpGenerator vectorShuffleGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operand0Candidates = std::vector<mlir::Value>();
    mb->search(vector_filter, &operand0Candidates);
    if (operand0Candidates.empty()) {
      operand0Candidates.push_back(
        generateVector(builder, loc, randomVectorType(builder.getContext()), mb));
    }
    auto operand0 = operand0Candidates[rollIdx(operand0Candidates.size())];
    auto vTy0 = operand0.getType().dyn_cast<VectorType>();

    std::vector<mlir::Value> operand1Candidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!vector_filter(s)) return false;
        return hasSameTrailingShape(v.getType().dyn_cast<VectorType>(), vTy0);
      },
      &operand1Candidates
    );
    if (operand1Candidates.empty()) {
      operand1Candidates.push_back(generateVector(builder, loc, vTy0, mb));
    }
    auto operand1 = operand1Candidates[rollIdx(operand1Candidates.size())];
    auto vTy1 = operand1.getType().dyn_cast<VectorType>();

    SmallVector<int64_t> mask;
    if (vTy0.getRank() == 0) {
      mask.push_back(0);
      mask.push_back(1);
    } else {
      for (int i = 0; i < vTy0.getDimSize(0) + vTy1.getDimSize(0); ++i) {
        if (rollIdx(2)) {
          mask.push_back(i);
        }
      }
      // vector types must have positive constant sizes but got 0
      if (mask.empty()) {
        mask.push_back(0);
      }
    }
    SmallVector<int64_t> resShape;
    resShape.push_back(mask.size());
    for (int i = 1; i < vTy1.getRank(); ++i) {
      resShape.push_back(vTy1.getDimSize(i));
    }
    auto resTy = VectorType::get(resShape, vTy1.getElementType());

    mlir::Value res = builder.create<vector::ShuffleOp>(loc, operand0, operand1, mask);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorSplatGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> inputCandidates = std::vector<mlir::Value>();
    mb->search(int_float_index_filter, &inputCandidates);
    if (inputCandidates.empty()) {
      inputCandidates.push_back(
        generateTypedValue(builder, loc, randomElementaryOrIndexType(builder.getContext()), mb));
    }
    auto input = inputCandidates[rollIdx(inputCandidates.size())];
    auto vTy = randomVectorType(input.getType());

    mlir::Value res = builder.create<vector::SplatOp>(loc, input, vTy);
    mb->add2Pool(res);
    return true;
  };
}

OpGenerator vectorTransposeGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> vecCandidates = std::vector<mlir::Value>();
    mb->search(vector_filter, &vecCandidates);
    if (vecCandidates.empty()) {
      vecCandidates.push_back(
        generateVector(builder, loc, randomVectorType(builder.getContext()), mb));
    }
    auto vec = vecCandidates[rollIdx(vecCandidates.size())];
    auto vTy = vec.getType().dyn_cast<VectorType>();

    SmallVector<int64_t> transp;
    for (int i = 0; i < vTy.getRank(); ++i) {
      transp.insert(transp.begin() + rollIdx(transp.size() + 1), i);
    }
    SmallVector<int64_t> transposedShape;
    for (int i = 0; i < vTy.getRank(); ++i) {
      transposedShape.push_back(vTy.getDimSize(transp[i]));
    }

    builder.create<vector::TransposeOp>(loc, vec, transp);
    return true;
  };
}

OpGenerator vectorTransferReadGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> srcCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_memref_tensor_filter(s, v)) return false;
        auto shapedTy = v.getType().dyn_cast<ShapedType>();
        return shapedTy.getRank() > 0 &&
                shapedTy.getElementType().isIntOrFloat();
      },
      &srcCandidates
    );
    if (srcCandidates.empty()) {
      srcCandidates.push_back(
        generateStaticShapedMemref(
          builder, 
          loc, 
          randomStaticShapedMemrefType(builder.getContext())));
    }
    auto src = srcCandidates[rollIdx(srcCandidates.size())];

    auto srcRank = src.getType().dyn_cast<ShapedType>().getRank();
    auto srcElemTy = src.getType().dyn_cast<ShapedType>().getElementType();
    std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
    mb->search(index_filter, &indexCandidates);
    if (indexCandidates.empty()) {
      indexCandidates.push_back(generateIndex(builder, loc, 1));
    }
    SmallVector<Value> indices;
    for (int i = 0; i < srcRank; ++i) {
      indices.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
    }
    assert(srcRank > 0);
    auto vectorRank = rollIdx(srcRank);
    SmallVector<int64_t> vectorShape;
    for (int i = 0; i < vectorRank; ++i) {
      vectorShape.push_back(rollIdx(dim_ub)+1);
    }
    auto vecTy = VectorType::get(
      vectorShape, src.getType().dyn_cast<ShapedType>().getElementType());

    SmallVector<AffineExpr> results;
    for (int i = 0; i < vectorRank; ++i) {
      if (rollIdx(2)) {
        results.push_back(builder.getAffineDimExpr(i));
      } else {
        results.push_back(builder.getAffineConstantExpr(0));
      }
    }
    auto map = AffineMap::get(srcRank, 0, results, builder.getContext());
    auto permutationMapAttr = AffineMapAttr::get(map);
    
    std::string srcElemTyStr = getValueTypeStr(srcElemTy);
    std::vector<mlir::Value> paddingCandidates = std::vector<mlir::Value>();
    if (mb->valuePool.pool.count(srcElemTyStr)) {
      paddingCandidates.insert(
        paddingCandidates.end(), 
        mb->valuePool.pool[srcElemTyStr].begin(), 
        mb->valuePool.pool[srcElemTyStr].end());
    }
    if (paddingCandidates.empty()) {
      if (srcElemTy.isa<IntegerType>()) {
        paddingCandidates.push_back(
          generateInteger(builder, loc, srcElemTy.dyn_cast<IntegerType>()));
      } else {
        paddingCandidates.push_back(
          generateFloat(builder, loc, srcElemTy.dyn_cast<FloatType>()));
      }
    }
    auto padding = paddingCandidates[rollIdx(paddingCandidates.size())];
    if (rollIdx(2)) {
      builder.create<vector::TransferReadOp>(
        loc, vecTy, src, indices, permutationMapAttr, nullptr);
    } else {
      builder.create<vector::TransferReadOp>(
        loc, vecTy, src, indices, padding, std::optional<ArrayRef<bool>>());
    }
    return true;
  };
}

OpGenerator vectorTransferWriteGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> destCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        auto shapedTy = v.getType().dyn_cast<ShapedType>();
        return shapedTy.getRank() > 0 &&
                shapedTy.getElementType().isIntOrFloat();
      },
      &destCandidates
    );
    if (destCandidates.empty()) {
      destCandidates.push_back(
        generateStaticShapedMemref(
          builder, 
          loc, 
          randomStaticShapedMemrefType(builder.getContext())));
    }
    auto dest = destCandidates[rollIdx(destCandidates.size())];
    
    auto destRank = dest.getType().dyn_cast<ShapedType>().getRank();
    auto destElemTy = dest.getType().dyn_cast<ShapedType>().getElementType();

    std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
    mb->search(index_filter, &indexCandidates);
    if (indexCandidates.empty()) {
      indexCandidates.push_back(generateIndex(builder, loc, 1));
    }
    SmallVector<Value> indices;
    for (int i = 0; i < destRank; ++i) {
      indices.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
    }
    assert(destRank > 0);
    auto vectorRank = rollIdx(destRank);
    SmallVector<int64_t> vectorShape;
    for (int i = 0; i < vectorRank; ++i) {
      vectorShape.push_back(rollIdx(dim_ub)+1);
    }
    auto vecTy = VectorType::get(
      vectorShape, dest.getType().dyn_cast<ShapedType>().getElementType());
    std::vector<mlir::Value> vectorCandidates = std::vector<mlir::Value>();
    std::string vecTyStr = getValueTypeStr(vecTy);
    if (mb->valuePool.pool.count(vecTyStr)) {
      vectorCandidates.insert(
        vectorCandidates.begin(), 
        mb->valuePool.pool[vecTyStr].begin(), 
        mb->valuePool.pool[vecTyStr].end());
    }
    if (vectorCandidates.empty()) {
      vectorCandidates.push_back(generateVector(builder, loc, vecTy, mb));
    }
    auto vector = vectorCandidates[rollIdx(vectorCandidates.size())];

    SmallVector<AffineExpr> results;
    for (int i = 0; i < vectorRank; ++i) {
      results.push_back(builder.getAffineDimExpr(i));
    }
    auto map = AffineMap::get(destRank, 0, results, builder.getContext());
    auto permutationMapAttr = AffineMapAttr::get(map);

    builder.create<vector::TransferWriteOp>(
      loc, vector, dest, indices, permutationMapAttr, nullptr, nullptr);
    return true;
  };
}

OpGenerator vectorWarpExecuteOnLane0Op() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    auto laneId = generateIndex(builder, loc, 0);
    int warpSize = 32;

    auto op = builder.create<vector::WarpExecuteOnLane0Op>(
      loc, TypeRange({}), laneId, warpSize);

    auto point = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&op.getRegion().front());

    MutationBlock* childMb = new MutationBlock(*mb);
    unsigned statementNum = rollIdx(8);
    for (unsigned stIdx = 0 ; stIdx < statementNum ; ++stIdx) {
      OpGenerator mutator = generators[rollIdx(generators.size())];
      mutator(builder, loc, childMb);
    }
    builder.create<vector::YieldOp>(loc);

    builder.restoreInsertionPoint(point);
    return true;
  };
}