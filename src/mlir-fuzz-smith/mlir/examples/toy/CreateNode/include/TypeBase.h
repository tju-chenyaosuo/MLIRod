#ifndef TYPE_BASE_H
#define TYPE_BASE_H

#include "MutationUtil.h"
#include "ValuePool.h"

std::vector<mlir::Type> getSupportedIntTypes(MLIRContext *ctx);
std::vector<mlir::Type> getSupportedFloatTypes(MLIRContext *ctx);
std::vector<mlir::Type> getSupportedIntOrFloatTypes(MLIRContext *ctx);

inline std::vector<int64_t> getRandomShape() {
  std::vector<int64_t> randomShape;
  for (int j = 0; j < rollIdx(3)+1; j++) {
		randomShape.push_back(rollIdx(32)+1);
	}
  return randomShape;
}

inline Type randomIntOrFloatType(MLIRContext *ctx) {
  auto supportedTypes = getSupportedIntOrFloatTypes(ctx);
  return supportedTypes[rollIdx(supportedTypes.size())];
}

inline Type randomIntType(MLIRContext *ctx) {
  auto supportedTypes = getSupportedIntTypes(ctx);
  return supportedTypes[rollIdx(supportedTypes.size())];
}

//-----------------------------------------------------------------------
// Functions for generate memref type

// memref can be static ranked, dynamically ranked and unranked.
inline MemRefType randomStaticShapedMemrefType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  // generate random shape
	std::vector<int64_t> randomShape;
	for (int j = 0; j < rollIdx(3)+1; j++) {
		randomShape.push_back(rollIdx(32)+1);
	}
  return MemRefType::get(randomShape, elemType);
}

inline MemRefType randomStaticShapedMemrefType(Type elemTy) {
	std::vector<int64_t> randomShape;
	for (int j = 0; j < rollIdx(3)+1; j++) {
		randomShape.push_back(rollIdx(32)+1);
	}
  return MemRefType::get(randomShape, elemTy);
}

inline MemRefType randomDynamicShapedMemrefType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  std::vector<int64_t> randomShape;
	for (int j = 0; j < rollIdx(3)+1; j++) {
		randomShape.push_back(rollIdx(32)+1);
	}
  auto num = rollIdx(randomShape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    randomShape[i] = ShapedType::kDynamic;
  }
  return MemRefType::get(randomShape, elemType);
}

inline MemRefType randomDynamicShapedMemrefType(Type elemType) {
  std::vector<int64_t> randomShape;
	for (int j = 0; j < rollIdx(3)+1; j++) {
		randomShape.push_back(rollIdx(32)+1);
	}
  auto num = rollIdx(randomShape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    randomShape[i] = ShapedType::kDynamic;
  }
  return MemRefType::get(randomShape, elemType);
}

inline MemRefType randomRankedMemrefType(Type ty) {
  if (rollIdx(2)) {
    return randomStaticShapedMemrefType(ty);
  } else {
    return randomDynamicShapedMemrefType(ty);
  }
}

inline MemRefType randomRankedMemrefType(MLIRContext *ctx) {
  if (rollIdx(2)) {
    return randomStaticShapedMemrefType(ctx);
  } else {
    return randomDynamicShapedMemrefType(ctx);
  }
}

//-----------------------------------------------------------------------
// Functions for generate tensor type
inline Type randomNonTensorType(MLIRContext *ctx) {
  std::vector<Type> candidates = {randomIntOrFloatType(ctx),
                                  randomStaticShapedMemrefType(ctx)};
  return candidates[rollIdx(candidates.size())];
}

inline RankedTensorType randomStaticShapedTensorType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  std::vector<int64_t> shape = getRandomShape();
  return RankedTensorType::get(shape, elemType);
}

inline RankedTensorType randomStaticShapedTensorType(Type elemTy) {
  std::vector<int64_t> shape = getRandomShape();
  return RankedTensorType::get(shape, elemTy);
}

inline RankedTensorType randomDynamicShapedTensorType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  std::vector<int64_t> shape = getRandomShape();
  auto num = rollIdx(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = ShapedType::kDynamic;
  }
  return RankedTensorType::get(shape, elemType);
}

inline RankedTensorType randomDynamicShapedTensorType(Type elemTy) {
  auto shape = getRandomShape();
  auto num = rollIdx(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = ShapedType::kDynamic;
  }
  return RankedTensorType::get(shape, elemTy);
}

inline RankedTensorType randomRankedTensorType(MLIRContext *ctx) {
  if (rollIdx(2)) {
    return randomStaticShapedTensorType(ctx);
  } else {
    return randomDynamicShapedTensorType(ctx);
  }
}

//-----------------------------------------------------------------------
// Functions for generate vector type
inline VectorType random1DVectorType(Type elemTy) {
  const int dim_ub = 32;
  SmallVector<int64_t> shape;
  shape.push_back(rollIdx(dim_ub)+1);
  return VectorType::get(shape, elemTy);
}

inline VectorType random1DVectorType(MLIRContext *ctx) {
  const int dim_ub = 32;
  SmallVector<int64_t> shape;
  shape.push_back(rollIdx(dim_ub)+1);
  return VectorType::get(shape, randomIntOrFloatType(ctx));
}

//-----------------------------------------------------------------------
// Functions for generate random complex type
inline Type randomMemrefOrRankedTensorType(MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  auto shape = getRandomShape();
  if (rollIdx(2)) {
    return RankedTensorType::get(shape, elemTy);
  } else {
    return MemRefType::get(shape, elemTy);
  }
}

inline Type randomMemrefOrRankedTensorType(ShapedType t) {
  auto shape = t.getShape();
  auto elemTy = t.getElementType();
  if (rollIdx(2)) {
    return RankedTensorType::get(shape, elemTy);
  } else {
    return MemRefType::get(shape, elemTy);
  }
}

inline Type randomFloatType(MLIRContext *ctx) {
  std::vector<Type> candidates;
  auto supportedTypes = getSupportedFloatTypes(ctx);
  auto elemTy = supportedTypes[rollIdx(supportedTypes.size())];
  candidates.push_back(elemTy);
  candidates.push_back(randomStaticShapedMemrefType(elemTy));
  candidates.push_back(randomStaticShapedTensorType(elemTy));
  return candidates[rollIdx(candidates.size())];
}

inline VectorType randomVectorType(MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  auto shape = getRandomShape();
  return VectorType::get(shape, elemTy);
}

inline VectorType randomVectorType(Type elemTy) {
  auto shape = getRandomShape();
  return VectorType::get(shape, elemTy);
}

inline Type randomType(MLIRContext *ctx) {
  std::vector<Type> candidates = {IndexType::get(ctx),
                                  randomIntOrFloatType(ctx),
                                  randomStaticShapedMemrefType(ctx),
                                  randomStaticShapedTensorType(ctx),
                                  randomDynamicShapedTensorType(ctx),
                                  randomDynamicShapedMemrefType(ctx),
                                  randomVectorType(ctx)};
  return candidates[rollIdx(candidates.size())];
}

inline Type randomElementaryOrIndexType(MLIRContext *ctx) {
  auto types = getSupportedIntOrFloatTypes(ctx);
  types.push_back(IndexType::get(ctx));
  return types[rollIdx(types.size())];
}

mlir::Value generateIndex(mlir::OpBuilder &builder, mlir::Location loc);
mlir::Value generateIndex(mlir::OpBuilder &builder, mlir::Location loc, int q);
mlir::Value generateInteger(OpBuilder &builder, Location loc, IntegerType type);
mlir::Value generateFloat(OpBuilder &builder, Location loc, FloatType type);
mlir::Value generateElement(OpBuilder &builder, Location loc, Type type);
mlir::Value generateStaticShapedMemref(OpBuilder &builder, Location loc, MemRefType type);
mlir::Value generateDynamicShapedMemref(OpBuilder &builder, Location loc, MemRefType type, MutationBlock* mb);
mlir::Value generateVector(OpBuilder &builder, Location loc, VectorType type, MutationBlock* mb);
mlir::Value generateStaticShapedTensor(OpBuilder &builder, Location loc, RankedTensorType type);
mlir::Value generateDynamicShapedTensor(OpBuilder &builder, Location loc, RankedTensorType type, MutationBlock* mb);
mlir::Value generateRankedTensor(mlir::OpBuilder &builder, mlir::Location loc, mlir::RankedTensorType rtTy, MutationBlock* mb);
mlir::Value generateRankedMemref(mlir::OpBuilder &builder, mlir::Location loc, mlir::MemRefType type, MutationBlock* mb);
mlir::Value createNewValue(OpBuilder &builder, Location loc, Type type);
mlir::Value generateTypedValue(OpBuilder &builder, Location loc, Type type, MutationBlock* mb);
bool isTheSameShapedType(ShapedType typeA, ShapedType typeB);
std::vector<mlir::Value> searchShapedInputFrom(ShapedType type, const std::vector<mlir::Value> &pool);
SmallVector<mlir::Value> randomIndicesForShapedType(ShapedType shapedType, OpBuilder &builder, Location loc);

inline bool isLLVMIRIntegerBitWidth(int intBitWidth) {
  return intBitWidth == 8 || intBitWidth == 16 || intBitWidth == 32 ||
         intBitWidth == 64;
}
#endif

