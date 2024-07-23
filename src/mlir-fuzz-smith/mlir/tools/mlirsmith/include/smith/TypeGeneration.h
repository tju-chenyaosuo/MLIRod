//
// Created by Stan Wang on 2022/10/31.
//

#ifndef MLIR_FUZZ_TYPEGENERATION_H
#define MLIR_FUZZ_TYPEGENERATION_H
#pragma once

#include "Util.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

using namespace mlir;

// TODO- Migrate to Conf
const int rank_ub = 3;
const int dim_ub = 32;
const int idx_ub = dim_ub;
const int func_arg_ub = 4;
const int func_ret_ub = 2; // the fun either have one return value or none.
const int dimPoolSize = 4;
const int shapePoolSize = 4;

inline std::vector<int64_t> dimPool;
inline std::vector<std::vector<int64_t>> shapePool; // non-zero-ranked Shapes

inline long long random(long long mod) {
  long long rd = ((long long)rand() << 32) | rand();
  return rd % mod;
}

inline int randomRank() { return random(rank_ub) + 1; }

inline void initType() {
  for (int i = 0; i < dimPoolSize; i++) {
    dimPool.push_back(random(dim_ub) + 1);
  }
  // TODO-support dynamic shape
  // dimPool.push_back(ShapedType::kDynamicSize);

  // init shape pool
  for (int i = 0; i < shapePoolSize; i++) {
    std::vector<int64_t> dims;
    for (int j = 0; j < randomRank(); j++) {
      dims.push_back(dimPool[random(dimPoolSize)]);
    }
    shapePool.push_back(dims);
  }
}

inline Type randomIntOrFloatType(MLIRContext *ctx) {
  auto supportedTypes = getSupportedIntOrFloatTypes(ctx);
  return supportedTypes[random(supportedTypes.size())];
}

inline Type randomElementaryOrIndexType(MLIRContext *ctx) {
  auto types = getSupportedIntOrFloatTypes(ctx);
  types.push_back(IndexType::get(ctx));
  return types[random(types.size())];
}

inline VectorType randomVectorType(MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  return VectorType::get(shape, elemTy);
}

inline VectorType randomVectorType(Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  return VectorType::get(shape, elemTy);
}

inline VectorType random1DVectorType(Type elemTy) {
  SmallVector<int64_t> shape;
  shape.push_back(dimPool[random(dimPool.size())]);
  return VectorType::get(shape, elemTy);
}

inline VectorType random1DVectorType(MLIRContext *ctx) {
  SmallVector<int64_t> shape;
  shape.push_back(dimPool[random(dimPool.size())]);
  return VectorType::get(shape, randomIntOrFloatType(ctx));
}

// memref can be static ranked, dynamically ranked and unranked.
inline MemRefType randomStaticShapedMemrefType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  return MemRefType::get(shape, elemType);
}

inline MemRefType randomStaticShapedMemrefType(Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  return MemRefType::get(shape, elemTy);
}

inline MemRefType randomDynamicShapedMemrefType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = ShapedType::kDynamic;
  }
  return MemRefType::get(shape, elemType);
}

inline MemRefType randomDynamicShapedMemrefType(Type elemType) {
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = ShapedType::kDynamic;
  }
  return MemRefType::get(shape, elemType);
}

inline MemRefType randomRankedMemrefType(MLIRContext *ctx) {
  if (UR(2)) {
    return randomStaticShapedMemrefType(ctx);
  } else {
    return randomDynamicShapedMemrefType(ctx);
  }
}

inline MemRefType randomRankedMemrefType(Type ty) {
  if (UR(2)) {
    return randomStaticShapedMemrefType(ty);
  } else {
    return randomDynamicShapedMemrefType(ty);
  }
}

inline RankedTensorType randomStaticShapedTensorType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  return RankedTensorType::get(shape, elemType);
}

inline RankedTensorType randomStaticShapedTensorType(Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  return RankedTensorType::get(shape, elemTy);
}

inline RankedTensorType randomDynamicShapedTensorType(MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = ShapedType::kDynamic;
  }
  return RankedTensorType::get(shape, elemType);
}

inline RankedTensorType randomDynamicShapedTensorType(Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = ShapedType::kDynamic;
  }
  return RankedTensorType::get(shape, elemTy);
}

inline RankedTensorType randomRankedTensorType(MLIRContext *ctx) {
  if (UR(2)) {
    return randomStaticShapedTensorType(ctx);
  } else {
    return randomDynamicShapedTensorType(ctx);
  }
}

inline ShapedType randomStaticShapeType(Type elemTy) {
  switch (random(3)) {
  case 0:
    return randomStaticShapedMemrefType(elemTy);
  case 1:
    return randomStaticShapedTensorType(elemTy);
  default:
    return randomVectorType(elemTy);
  }
}

inline ShapedType randomStaticShapeType(MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  return randomStaticShapeType(elemTy);
}

inline Type randomIntegerType(MLIRContext *ctx) {
  std::vector<Type> candidates;
  auto supportedTypes = getSupportedIntTypes(ctx);
  auto elemTy = supportedTypes[random(supportedTypes.size())];
  candidates.push_back(elemTy);
  candidates.push_back(randomStaticShapedMemrefType(elemTy));
  candidates.push_back(randomStaticShapedTensorType(elemTy));
  return candidates[random(candidates.size())];
}

inline Type randomFloatType(MLIRContext *ctx) {
  std::vector<Type> candidates;
  auto supportedTypes = getSupportedFloatTypes(ctx);
  auto elemTy = supportedTypes[random(supportedTypes.size())];
  candidates.push_back(elemTy);
  candidates.push_back(randomStaticShapedMemrefType(elemTy));
  candidates.push_back(randomStaticShapedTensorType(elemTy));
  return candidates[random(candidates.size())];
}

inline Type randomType(MLIRContext *ctx) {
  std::vector<Type> candidates = {IndexType::get(ctx),
                                  randomIntOrFloatType(ctx),
                                  randomStaticShapedMemrefType(ctx),
                                  randomStaticShapedTensorType(ctx),
                                  randomDynamicShapedTensorType(ctx),
                                  randomDynamicShapedMemrefType(ctx),
                                  randomVectorType(ctx)};
  return candidates[random(candidates.size())];
}

inline Type randomNonTensorType(MLIRContext *ctx) {
  std::vector<Type> candidates = {randomIntOrFloatType(ctx),
                                  randomStaticShapedMemrefType(ctx)};
  return candidates[random(candidates.size())];
}

inline FunctionType randomFunctionType(MLIRContext *ctx) {
  int func_arg_num = random(func_arg_ub);
  int func_ret_num = random(func_ret_ub);
  SmallVector<Type> argTypes;
  SmallVector<Type> retTypes;
  for (int i = 0; i < func_arg_num; ++i) {
    argTypes.push_back(randomType(ctx));
  }

  for (int i = 0; i < func_ret_num; ++i) {
    retTypes.push_back(randomType(ctx));
  }
  auto funcType =
      FunctionType::get(ctx, TypeRange(argTypes), TypeRange(retTypes));

  return funcType;
}

inline Type randomMemrefOrRankedTensorType(MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  if (random(2)) {
    return RankedTensorType::get(shape, elemTy);
  } else {
    return MemRefType::get(shape, elemTy);
  }
}

inline Type randomMemrefOrRankedTensorType(ShapedType t) {
  auto shape = t.getShape();
  auto elemTy = t.getElementType();
  if (random(2)) {
    return RankedTensorType::get(shape, elemTy);
  } else {
    return MemRefType::get(shape, elemTy);
  }
}

#endif // MLIR_FUZZ_TYPEGENERATION_H
