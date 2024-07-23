//
// Created by Stan Wang on 2022/9/13.
//
/*
 *myheader.h
 */
#include "smith/Util.h"
#include "smith/DiversityCriteria.h"
#include "smith/ExpSetting.h"
#include "smith/TypeGeneration.h"
#include "smith/ValueGeneration.h"

int min(int i1, int i2) { return i1 < i2 ? i1 : i2; }
int max(int i1, int i2) { return i1 < i2 ? i2 : i1; }

Value getConstant(Location loc, OpBuilder &b, Type type, long long val) {
  if (type.dyn_cast<IntegerType>()) {
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(type, val));
  } else if (type.dyn_cast<FloatType>()) {
    return b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(type, static_cast<double>(val)));
  } else {
    if (conf.debug) {
      std::cout << "currently only support integer and float type" << std::endl;
    }
    return b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(type, static_cast<double>(val)));
  }
}

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

long long UR(long long mod) {
  long long rd = ((long long)rand() << 32) | rand();
  return rd % mod;
}

std::vector<TypeValue>
searchShapedInputFrom(ShapedType type, const std::vector<TypeValue> &pool) {

  std::vector<TypeValue> candidates;
  for (auto tensor : pool) {
    ShapedType t = tensor.type.dyn_cast<ShapedType>();
    if (isTheSameShapedType(type, t)) {
      candidates.push_back(tensor);
    }
  }
  return candidates;
}

TypeValue sampleTypedValueFrom(std::vector<TypeValue> candidates,
                               std::string opToBuild) {
  assert(!candidates.empty());
    return candidates[UR(candidates.size())];
}

std::vector<TypeValue>
searchIntegerCandidatesFrom(std::vector<TypeValue> pool) {
  std::vector<TypeValue> res;
  for (auto elem : pool) {
    RankedTensorType t = elem.type.dyn_cast<RankedTensorType>();
    if (t &&
        t.dyn_cast<ShapedType>().getElementType().dyn_cast<IntegerType>()) {
      res.push_back(elem);
    } else if (elem.type.dyn_cast<IntegerType>()) {
      res.push_back(elem);
    }
  }
  return res;
}

std::vector<TypeValue> searchFloatCandidatesFrom(std::vector<TypeValue> pool) {
  std::vector<TypeValue> res;
  for (auto elem : pool) {
    RankedTensorType t = elem.type.dyn_cast<RankedTensorType>();
    if (t && t.dyn_cast<ShapedType>().getElementType().dyn_cast<FloatType>()) {
      res.push_back(elem);
    } else if (elem.type.dyn_cast<FloatType>()) {
      res.push_back(elem);
    }
  }
  return res;
}

// search value with specified type
std::vector<TypeValue> searchTypedValueFrom(const std::vector<TypeValue> pool,
                                            Type type) {
  std::vector<TypeValue> res;
  for (auto elem : pool) {
    if (elem.type == type) {
      res.push_back(elem);
    }
  }
  return res;
}

TypeValue searchIntegerUnaryOperandFrom(OpBuilder &builder, Location loc,
                                        const std::vector<TypeValue> pool,
                                        std::string opToBuild) {
  auto candidates = searchIntegerCandidatesFrom(pool);
  if (candidates.empty()) {
    std::cout << "empty candidate unary int" << std::endl;
    auto iType = randomIntegerType(builder.getContext());
    auto tVal = createNewValue(builder, loc, iType);
    candidates.push_back(tVal);
  }
  return sampleTypedValueFrom(candidates, opToBuild);
}

TypeValue searchFloatUnaryOperandFrom(OpBuilder &builder, Location loc,
                                      const std::vector<TypeValue> pool,
                                      std::string opToBuild) {
  auto candidates = searchFloatCandidatesFrom(pool);
  if (candidates.empty()) {
    std::cout << "empty candidate unary float" << std::endl;
    auto fType = randomFloatType(builder.getContext());
    auto tVal = createNewValue(builder, loc, fType);
    candidates.push_back(tVal);
  }
  return sampleTypedValueFrom(candidates, opToBuild);
}

std::vector<TypeValue>
searchIntegerBinaryOperandsFrom(OpBuilder &builder, Location loc,
                                const std::vector<TypeValue> pool,
                                std::string opToBuild) {
  auto operandLeftCandidates = searchIntegerCandidatesFrom(pool);
  if (operandLeftCandidates.empty()) {
    std::cout << "empty candidate binary int l" << std::endl;
    auto iType = randomIntegerType(builder.getContext());
    auto tVal = createNewValue(builder, loc, iType);
    operandLeftCandidates.push_back(tVal);
  }
  auto operandLeft = sampleTypedValueFrom(operandLeftCandidates, opToBuild);
  auto typeLeft = operandLeft.type;
  auto operandRightCandidates = searchTypedValueFrom(pool, typeLeft);

  if (operandRightCandidates.empty()) {
    std::cout << "empty candidate binary int r" << std::endl;
    auto tVal = createNewValue(builder, loc, typeLeft);
    operandRightCandidates.push_back(tVal);
  }
  auto operandRight = sampleTypedValueFrom(operandRightCandidates, opToBuild);

  std::vector<TypeValue> res = {operandLeft, operandRight};
  return res;
}

std::vector<TypeValue>
searchFloatBinaryOperandsFrom(OpBuilder &builder, Location loc,
                              const std::vector<TypeValue> pool,
                              std::string opToBuild) {
  auto operandLeftCandidates = searchFloatCandidatesFrom(pool);
  if (operandLeftCandidates.empty()) {
    std::cout << "empty candidate binary float l" << std::endl;
    auto fType = randomFloatType(builder.getContext());
    auto tVal = createNewValue(builder, loc, fType);
    operandLeftCandidates.push_back(tVal);
  }
  auto operandLeft = sampleTypedValueFrom(operandLeftCandidates, opToBuild);
  auto typeLeft = operandLeft.type;
  auto operandRightCandidates = searchTypedValueFrom(pool, typeLeft);
  if (operandRightCandidates.empty()) {
    std::cout << "empty candidate binary float r" << std::endl;
    auto tVal = createNewValue(builder, loc, typeLeft);
    operandRightCandidates.push_back(tVal);
  }
  auto operandRight = sampleTypedValueFrom(operandRightCandidates, opToBuild);

  std::vector<TypeValue> res = {operandLeft, operandRight};
  return res;
}

std::vector<TypeValue>
searchFloatTernaryOperandsFrom(OpBuilder &builder, Location loc,
                               const std::vector<TypeValue> pool,
                               std::string opToBuild) {
  auto operandLeftCandidates = searchFloatCandidatesFrom(pool);
  if (operandLeftCandidates.empty()) {
    std::cout << "empty candidate ternary float l" << std::endl;
    auto fType = randomFloatType(builder.getContext());
    auto tVal = createNewValue(builder, loc, fType);
    operandLeftCandidates.push_back(tVal);
  }
  auto operand0 = sampleTypedValueFrom(operandLeftCandidates, opToBuild);
  auto type0 = operand0.type;

  auto operand12Candidates = searchTypedValueFrom(pool, type0);
  if (operand12Candidates.empty()) {
    std::cout << "empty candidate binary float 2,3" << std::endl;
    auto tVal = createNewValue(builder, loc, type0);
    operand12Candidates.push_back(tVal);
  }
  auto operand1 = sampleTypedValueFrom(operand12Candidates, opToBuild);
  auto operand2 = sampleTypedValueFrom(operand12Candidates, opToBuild);

  std::vector<TypeValue> res = {operand0, operand1, operand2};
  return res;
}

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

// 2D mat-mul
std::vector<TypeValue> search2DTensorsFrom(const std::vector<TypeValue> pool) {
  std::vector<TypeValue> res;
  for (auto tensor : pool) {
    if (tensor.type.dyn_cast<RankedTensorType>() &&
        tensor.type.dyn_cast<ShapedType>().getShape().size() == 2) {
      res.push_back(tensor);
    }
  }
  return res;
}

std::vector<TypeValue> search2thTensorsFor(const std::vector<TypeValue> pool,
                                           ShapedType shapedType) {
  std::vector<TypeValue> res;
  for (auto tensor : pool) {
    if (tensor.type.dyn_cast<RankedTensorType>() &&
        tensor.type.dyn_cast<ShapedType>().getElementType() ==
            shapedType.getElementType() &&
        tensor.type.dyn_cast<ShapedType>().getShape().size() == 2 &&
        tensor.type.dyn_cast<ShapedType>().getShape()[0] ==
            shapedType.getDimSize(1)) {
      res.push_back(tensor);
    }
  }
  return res;
}

TypeValue createNewValue(OpBuilder &builder, Location loc, Type type) {
  if (type.dyn_cast<IntegerType>()) {
    auto attr = builder.getIntegerAttr(type, 1);
    Value i = builder.create<arith::ConstantOp>(loc, attr);
    return TypeValue(type, i);
  } else if (type.dyn_cast<FloatType>()) {
    auto attr = builder.getFloatAttr(type, 1.0);
    Value f = builder.create<arith::ConstantOp>(loc, attr);
    return TypeValue(type, f);
  } else if (type.isa<MemRefType>()) {
    auto ty = type.dyn_cast<MemRefType>();
    if (ty.hasStaticShape()) {
      auto m =
          builder.create<memref::AllocOp>(loc, type.dyn_cast<MemRefType>());
      return TypeValue(type, m);
    } else {
      assert(false);
    }
  } else if (type.dyn_cast<RankedTensorType>()) {
    auto shapeTy = type.dyn_cast<ShapedType>();
    auto t = builder.create<tensor::EmptyOp>(loc, shapeTy.getShape(),
                                             shapeTy.getElementType());
    return TypeValue(type, t);
  } else if (type.dyn_cast<VectorType>()) {
    assert(false);
  } else {
    std::cout << "unsupported type for new value generation\n";
    exit(-1);
  }
}

size_t getWeightedRandomIndex(std::vector<float> weights) {
  std::discrete_distribution<int> dist(std::begin(weights), std::end(weights));
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd()); // if you want different results from different runs
  int a = dist(gen);
  return a;
}

std::string configJsonStr() {
  Conf c = getConfigTemplate();
  return c.toJson().dump(4);
}

using AffineExprGen =
    std::function<AffineExpr(std::vector<AffineExpr> &, std::vector<int64_t>)>;

std::vector<AffineExprGen> exprGens = {
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto expr1 = exprs[UR(exprs.size())];
      return expr0 + expr1;
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto expr1 = exprs[UR(exprs.size())];
      return expr0 - expr1;
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto constant = constants[UR(constants.size())];
      return expr0 + constant;
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto constant = constants[UR(constants.size())];
      return expr0 - constant;
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto constant = constants[UR(constants.size())];
      return expr0 * constant;
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto constant = constants[UR(constants.size())];
      return expr0.floorDiv(constant);
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto constant = constants[UR(constants.size())];
      return expr0.ceilDiv(constant);
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      auto constant = constants[UR(constants.size())];
      return expr0 % constant;
    },
    [](std::vector<AffineExpr> &exprs, std::vector<int64_t> constants) {
      auto expr0 = exprs[UR(exprs.size())];
      return -expr0;
    }};

std::vector<int64_t> constants = {1, 2, 4, 8, 16, 32, 64, 128};

std::vector<AffineExpr> randomAffineExprs(OpBuilder &builder, int dimNum,
                                          int symbolNum) {

  std::vector<AffineExpr> affineExprs;
  for (int j = 0; j < dimNum; ++j) {
    affineExprs.push_back(builder.getAffineDimExpr(j));
  }
  for (int j = 0; j < affine_expr_comb_ub; ++j) {
    auto exprGen = exprGens[UR(exprGens.size())];
    affineExprs.push_back(exprGen(affineExprs, constants));
  }
  return affineExprs;
}

void debugPrint(std::string str) {
  if (true) {
    std::cout << str << std::endl;
  }
}
