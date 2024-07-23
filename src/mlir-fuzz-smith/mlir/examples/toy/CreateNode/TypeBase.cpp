#include "TypeBase.h"

//-----------------------------------------------------------------------
// Functions for generate simple type
std::vector<mlir::Type> getSupportedIntTypes(MLIRContext *ctx) {
  std::vector<mlir::Type> res = {
      IntegerType::get(ctx, 1), IntegerType::get(ctx, 16), // an unusual type
      IntegerType::get(ctx, 32), IntegerType::get(ctx, 64)};
  return res;
}

std::vector<mlir::Type> getSupportedFloatTypes(MLIRContext *ctx) {
  std::vector<mlir::Type> res = {FloatType::getF16(ctx), FloatType::getF32(ctx)
//                              ,   FloatType::getF80(ctx),
//                                 FloatType::getF128(ctx)
  };
  return res;
}

std::vector<mlir::Type> getSupportedIntOrFloatTypes(MLIRContext *ctx) {
//  auto ctx = builder.getContext();
  std::vector<mlir::Type> supportedElementaryTypes;
  auto intTys = getSupportedIntTypes(ctx);
  auto floatTys = getSupportedFloatTypes(ctx);
  supportedElementaryTypes.insert(supportedElementaryTypes.end(),
                                  intTys.begin(), intTys.end());
  supportedElementaryTypes.insert(supportedElementaryTypes.end(),
                                  floatTys.begin(), floatTys.end());
  return supportedElementaryTypes;
}

//-----------------------------------------------------------------------
// Functions for generate value of specific type.

mlir::Value 
generateIndex(mlir::OpBuilder &builder, mlir::Location loc) {
  mlir::Value res = builder.create<arith::ConstantIndexOp>(loc, 0);
  return res;
}

mlir::Value 
generateIndex(mlir::OpBuilder &builder, mlir::Location loc, int q) {
  mlir::Value res = builder.create<arith::ConstantIndexOp>(loc, q);
  return res;
}

mlir::Value 
generateInteger(OpBuilder &builder, Location loc, IntegerType type) {
  mlir::Value res = builder.create<arith::ConstantIntOp>(loc, rollIdx(2), type);
  return res;
}

mlir::Value 
generateFloat(OpBuilder &builder, Location loc, FloatType type) {
  auto one = APFloat(type.getFloatSemantics(), 1);
  mlir::Value res = builder.create<arith::ConstantFloatOp>(loc, one, type);
  return res;
}

mlir::Value 
generateElement(OpBuilder &builder, Location loc, Type type) {
  auto iTy = type.dyn_cast<IntegerType>();
  auto fTy = type.dyn_cast<FloatType>();
  auto idxTy = type.dyn_cast<IndexType>();
  assert(iTy || fTy || idxTy);
  if (iTy) {
    return generateInteger(builder, loc, iTy);
  } else if (fTy) {
    return generateFloat(builder, loc, fTy);
  } else {
    return generateIndex(builder, loc);
  }
}

mlir::Value 
generateStaticShapedMemref(OpBuilder &builder, Location loc, MemRefType type) {
  mlir::Value memref = builder.create<memref::AllocOp>(loc, type);
  return memref;
}

mlir::Value 
generateDynamicShapedMemref(OpBuilder &builder, Location loc, MemRefType type, MutationBlock* mb) {
  auto dyNum = type.getNumDynamicDims();
  SmallVector<Value> dyDims;
  // Get index candidates
  std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
  mb->search(index_filter, &idxCandidates);
  if (idxCandidates.empty()) {
    idxCandidates.push_back(builder.create<arith::ConstantIndexOp>(loc, 1)); 
  }
  // Construct dims
  for (int i = 0; i < dyNum; ++i) {
    dyDims.push_back(idxCandidates[rollIdx(idxCandidates.size())]);
  }
  mlir::Value memref = builder.create<memref::AllocOp>(loc, type, dyDims);
  return memref;
}

mlir::Value 
generateVector(OpBuilder &builder, Location loc, VectorType type, MutationBlock* mb) {
  auto shapeTy = type.dyn_cast<ShapedType>();
  auto elemTy = shapeTy.getElementType();
  std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
  if (!mb->valuePool.pool.count(getValueTypeStr(elemTy))) {
    if (elemTy.dyn_cast<IndexType>()) {
      candidates.push_back(generateIndex(builder, loc));
    } else {
      candidates.push_back(generateElement(builder, loc, elemTy));
    }
  } else {
    candidates.insert(
      candidates.end(), 
      mb->valuePool.pool[getValueTypeStr(elemTy)].begin(), 
      mb->valuePool.pool[getValueTypeStr(elemTy)].end());
  }
  mlir::Value elem = candidates[rollIdx(candidates.size())];
  mlir::Value vec = builder.create<vector::BroadcastOp>(loc, type, elem);
  return vec;
}

mlir::Value 
generateStaticShapedTensor(OpBuilder &builder, Location loc, RankedTensorType type) {
  assert(type.hasStaticShape());
  auto shapedType = type.dyn_cast<ShapedType>();
  mlir::Value t = builder.create<tensor::EmptyOp>(loc, shapedType.getShape(), shapedType.getElementType());
  return t;
}

// generate an empty d tensor
mlir::Value 
generateDynamicShapedTensor(OpBuilder &builder, Location loc, RankedTensorType type, MutationBlock* mb) {
  auto dynDimNum = type.getNumDynamicDims();
  SmallVector<Value> dynDims;
  
  std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
  mb->search(index_filter, &candidates);
  if (candidates.empty()) {
    candidates.push_back(generateIndex(builder, loc));
  }
  for (int i = 0; i < dynDimNum; ++i) {
    dynDims.push_back(candidates[rollIdx(candidates.size())]);
  }
  mlir::Value t = builder.create<tensor::EmptyOp>(loc, type, ValueRange(dynDims));
  return t;
}

mlir::Value generateRankedTensor(mlir::OpBuilder &builder, mlir::Location loc, mlir::RankedTensorType rtTy, MutationBlock* mb) {
  assert(rtTy.hasRank());
  if (rtTy.hasStaticShape()) {
    return generateStaticShapedTensor(builder, loc, rtTy);
  } else {
    return generateDynamicShapedTensor(builder, loc, rtTy, mb);
  }
}

mlir::Value generateRankedMemref(mlir::OpBuilder &builder, mlir::Location loc, mlir::MemRefType type, MutationBlock* mb) {
  if (type.hasStaticShape()) {
    return generateStaticShapedMemref(builder, loc, type);
  } else {
    return generateDynamicShapedMemref(builder, loc, type, mb);
  }
}

mlir::Value createNewValue(OpBuilder &builder, Location loc, Type type) {
  if (type.dyn_cast<IntegerType>()) {
    auto attr = builder.getIntegerAttr(type, 1);
    Value i = builder.create<arith::ConstantOp>(loc, attr);
    return i;
  } else if (type.dyn_cast<FloatType>()) {
    auto attr = builder.getFloatAttr(type, 1.0);
    Value f = builder.create<arith::ConstantOp>(loc, attr);
    return f;
  } else if (type.isa<MemRefType>()) {
    auto ty = type.dyn_cast<MemRefType>();
    if (ty.hasStaticShape()) {
      auto m = builder.create<memref::AllocOp>(loc, type.dyn_cast<MemRefType>());
      return m;
    } else {
      assert(false);
    }
  } else if (type.dyn_cast<RankedTensorType>()) {
    auto shapeTy = type.dyn_cast<ShapedType>();
    auto t = builder.create<tensor::EmptyOp>(
      loc, 
      shapeTy.getShape(), 
      shapeTy.getElementType());
    return t;
  } else if (type.dyn_cast<VectorType>()) {
    assert(false);
  } else {
    llvm::outs() << "unsupported type for new value generation\n";
    exit(-1);
  }
}

mlir::Value generateTypedValue(OpBuilder &builder, Location loc, Type type, MutationBlock* mb) {
  if (type.isa<IntegerType>()) {
    return generateInteger(builder, loc, type.dyn_cast<IntegerType>());
  } else if (type.isa<FloatType>()) {
    return generateFloat(builder, loc, type.dyn_cast<FloatType>());
  } else if (type.isa<MemRefType>()) {
    auto memTy = type.dyn_cast<MemRefType>();
    if (memTy.hasStaticShape()) {
      return generateStaticShapedMemref(builder, loc, memTy);
    } else {
      return generateDynamicShapedMemref(builder, loc, memTy, mb);
    }
  } else if (type.isa<RankedTensorType>()) {
    auto rtTy = type.dyn_cast<RankedTensorType>();
    if (rtTy.hasStaticShape()) {
      return generateStaticShapedTensor(builder, loc, rtTy);
    } else {
      return generateDynamicShapedTensor(builder, loc, rtTy, mb);
    }
  } else if (type.isa<VectorType>()) {
    return generateVector(builder, loc, type.dyn_cast<VectorType>(), mb);
  } else if (type.isa<IndexType>()) {
    return generateIndex(builder, loc);
  } else {
    llvm::outs() << "unsupported type for new value generation\n";
    exit(-1);
  }
}

// inline VectorType randomVectorType(MLIRContext *ctx) {
//   auto elemTy = randomIntOrFloatType(ctx);
//   auto shape = getRandomShape();
//   return VectorType::get(shape, elemTy);
// }

//-----------------------------------------------------------------------
// Other approaches...
bool isTheSameShapedType(ShapedType typeA, ShapedType typeB) {
  if (!typeA || !typeB) {
    return false;
  }
  if (typeA.getElementType() != typeB.getElementType()) {
    return false;
  }
  auto shapeA = typeA.getShape();
  auto shapeB = typeB.getShape();
  if (shapeA.size() != shapeB.size()) {
    return false;
  }
  for (size_t i = 0; i < shapeA.size(); i++) {
    if (shapeA[i] != shapeB[i]) {
      return false;
    }
  }
  return true;
}

std::vector<mlir::Value>
searchShapedInputFrom(ShapedType type, const std::vector<mlir::Value> &pool) {
  std::vector<mlir::Value> candidates;
  for (auto tensor : pool) {
    ShapedType t = tensor.getType().dyn_cast<ShapedType>();
    if (isTheSameShapedType(type, t)) {
      candidates.push_back(tensor);
    }
  }
  return candidates;
}

SmallVector<mlir::Value> 
randomIndicesForShapedType(ShapedType shapedType, OpBuilder &builder, Location loc) {
  SmallVector<Value, 8> indices;
  for (size_t dim = 0; dim < shapedType.getRank(); ++dim) {
    if (shapedType.isDynamicDim(dim)) {
      indices.push_back(generateIndex(builder, loc, 0));
    } else {
      auto idx = rollIdx(shapedType.getDimSize(dim));
      auto idxVal = generateIndex(builder, loc, idx);
      indices.push_back(idxVal);
    }
  }
  return indices;
}

