//
// Created by Stan Wang on 2022/10/28.
//

#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"

/*------------ math generator ------------*/

OpGenerator absFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.absf");
    auto value = builder.create<math::AbsFOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.absf");
    return value.getOperation();
  };
}

OpGenerator absIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<IntegerType>() ||
                 (tval.type.isa<ShapedType>() &&
                  tval.type.dyn_cast<ShapedType>()
                      .getElementType()
                      .isa<IntegerType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.absi");
    auto value = builder.create<math::AbsIOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.absi");
    return value.getOperation();
  };
}

OpGenerator atanGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.atan");
    auto value = builder.create<math::AtanOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.atan");
    return value.getOperation();
  };
}

OpGenerator atan2Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "math.atan2");
    auto value =
        builder.create<math::Atan2Op>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.atan2");
    return value.getOperation();
  };
}

OpGenerator ceilGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.ceil");
    auto value = builder.create<math::CeilOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.ceil");
    return value.getOperation();
  };
}

OpGenerator copySignGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands = searchFloatBinaryOperandsFrom(builder, loc, candidates,
                                                  "math.copySign");
    auto value =
        builder.create<math::CopySignOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.copySign");
    return value.getOperation();
  };
} // f b

OpGenerator cosGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.cos");
    auto value = builder.create<math::CosOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.cos");
    return value.getOperation();
  };
} // f u

OpGenerator sinGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.sin");
    auto value = builder.create<math::SinOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.sin");
    return value.getOperation();
  };
}

OpGenerator ctlzGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<IntegerType>() ||
                 (tval.type.isa<ShapedType>() &&
                  tval.type.dyn_cast<ShapedType>()
                      .getElementType()
                      .isa<IntegerType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.ctzl");
    auto value = builder.create<math::CountLeadingZerosOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.ctzl");
    return value.getOperation();
  };
}

OpGenerator cttzGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<IntegerType>() ||
                 (tval.type.isa<ShapedType>() &&
                  tval.type.dyn_cast<ShapedType>()
                      .getElementType()
                      .isa<IntegerType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.cttz");
    auto value = builder.create<math::CountTrailingZerosOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.cttz");
    return value.getOperation();
  };
}

OpGenerator ctpopGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<IntegerType>() ||
                 (tval.type.isa<ShapedType>() &&
                  tval.type.dyn_cast<ShapedType>()
                      .getElementType()
                      .isa<IntegerType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.ctpop");
    auto value = builder.create<math::CtPopOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.ctpop");
    return value.getOperation();
  };
}

// OpGenerator erfGenerator(); //f u // What's the semantic of this op?

OpGenerator expGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.exp");
    auto value = builder.create<math::ExpOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.exp");
    return value.getOperation();
  };
}

OpGenerator exp2Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.exp2");
    auto value = builder.create<math::Exp2Op>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.exp2");
    return value.getOperation();
  };
}

OpGenerator expm1Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.expm1");
    auto value = builder.create<math::ExpM1Op>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.expm1");
    return value.getOperation();
  };
}

OpGenerator floorGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.floor");
    auto value = builder.create<math::FloorOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.floor");
    return value.getOperation();
  };
}

OpGenerator fmaGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchFloatTernaryOperandsFrom(builder, loc, candidates, "math.fma");
    auto value = builder.create<math::FmaOp>(loc, operands[0].val,
                                             operands[1].val, operands[2].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.fma");
    return value.getOperation();
  };
} // f t

OpGenerator ipowiGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getI32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "math.ipowi");
    auto value =
        builder.create<math::IPowIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.ipowi");
    return value.getOperation();
  };
} // i b

OpGenerator logGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log");
    auto value = builder.create<math::LogOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log");
    return value.getOperation();
  };
} // f u
OpGenerator log10Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log10");
    auto value = builder.create<math::Log10Op>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log10");
    return value.getOperation();
  };
}
OpGenerator log1pGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log1p");
    auto value = builder.create<math::Log1pOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log1p");
    return value.getOperation();
  };
}
OpGenerator log2Generator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.log2");
    auto value = builder.create<math::Log2Op>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.log2");
    return value.getOperation();
  };
}
OpGenerator powfGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "math.powF");
    auto value =
        builder.create<math::PowFOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "math.powF");
    return value.getOperation();
  };
} // f b

OpGenerator rsqrtGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.rqst");
    auto value = builder.create<math::RsqrtOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.rsqrt");
    return value.getOperation();
  };
} // f u
OpGenerator sqrtGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.sqrt");
    auto value = builder.create<math::SqrtOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.sqrt");
    return value.getOperation();
  };
}
OpGenerator tanGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.absf");
    auto value = builder.create<math::TanOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.tan");
    return value.getOperation();
  };
}
OpGenerator tanhGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.tanh");
    auto value = builder.create<math::TanhOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.tanh");
    return value.getOperation();
  };
}
OpGenerator roundEvenGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.roundeven");
    auto value = builder.create<math::RoundEvenOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.roundeven");
    return value.getOperation();
  };
}

OpGenerator roundGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();

    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.round");
    auto value = builder.create<math::RoundOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.round");
    return value.getOperation();
  };
}

OpGenerator truncGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto operandCandidates = typedValuePool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::DynamicShapedTensor,
         PoolType::StaticShapedTensor},
        [&](TypeValue tval) {
          return tval.type.isa<FloatType>() || (tval.type.isa<ShapedType>() &&
                                                tval.type.dyn_cast<ShapedType>()
                                                    .getElementType()
                                                    .isa<FloatType>());
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(
          typedValuePool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(operandCandidates, "math.trunc");
    auto value = builder.create<math::RoundOp>(loc, operand.val);
    auto tVal = TypeValue(operand.type, value);
    typedValuePool.addTypeValue(tVal, "math.trunc");
    return value.getOperation();
  };
}

OpGenerator fpowiGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto elemTy = builder.getF32Type();
    auto candidates =
        typedValuePool.getCandidatesFromIntOrFloats(builder, loc, elemTy);
    auto tensorCandidates = typedValuePool.getCandidatesFromStaticShapedTensor(
        builder, loc, randomStaticShapedTensorType(elemTy));
    candidates.insert(candidates.end(), tensorCandidates.begin(),
                      tensorCandidates.end());
    auto operand0 =
        searchFloatUnaryOperandFrom(builder, loc, candidates, "math.fpowi");
    auto ctx = builder.getContext();
    auto i32 = IntegerType::get(ctx, 32);
    std::vector<TypeValue> candidates2;

    if (operand0.type.dyn_cast<FloatType>()) {
      for (auto ele : typedValuePool.getCandidatesFromIntOrFloats(
               builder, loc, builder.getI32Type())) {
        if (ele.type == i32) {
          candidates2.push_back(ele);
        }
      }
    } else { // otherwise it should be Tensor Type
      auto t = RankedTensorType::get(
          operand0.type.dyn_cast<ShapedType>().getShape(), i32);
      candidates2 = searchShapedInputFrom(
          t,
          typedValuePool.getCandidatesFromStaticShapedTensor(builder, loc, t));
    }
    if (candidates2.empty()) {
      Type ty;
      if (operand0.type.isa<FloatType>()) {
        ty = builder.getI32Type();
      } else {
        ty = RankedTensorType::get(
            operand0.type.dyn_cast<ShapedType>().getShape(), i32);
      }
      candidates2.push_back(region.pool.generateTypedValue(builder, loc, ty));
    }
    auto operand1 = sampleTypedValueFrom(candidates2, "math.fpowi");
    auto value = builder.create<math::FPowIOp>(loc, operand0.val, operand1.val);
    auto tVal = TypeValue(operand0.type, value);
    typedValuePool.addTypeValue(tVal, "math.fpowi");
    return value.getOperation();
  };
} // f,i
