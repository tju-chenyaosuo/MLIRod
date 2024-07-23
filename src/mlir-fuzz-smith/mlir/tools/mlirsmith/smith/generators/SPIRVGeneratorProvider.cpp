//
// Created by Stan Wang on 2023/8/17.
//

#include "include/smith/RegionGeneration.h"
#include "include/smith/TypeGeneration.h"
#include "include/smith/generators/OpGeneration.h"

void registerSPIRVGenerators() {
  std::vector<OpGen> spirvGens = {
      spirvBitCountGenerator(),
      spirvBitReverseGenerator(),
      spirvFNegateGenerator(),
      spirvIsInfGenerator(),
      spirvIsNanGenerator(),
      spirvLogicalNotGenerator(),
      spirvNotGenerator(),
      spirvBitFieldInsertGenerator(),
      spirvBitFieldSExtractGenerator(),
      spirvBitFieldUExtractGenerator(),
      spirvBitwiseAndGenerator(),
      spirvBitwiseOrGenerator(),
      spirvBitwiseXorGenerator(),
      spirvCLCeilGenerator(),
      spirvCLCosGenerator(),
      spirvCLErfGenerator(),
      spirvCLExpGenerator(),
      spirvCLFAbsGenerator(),
      spirvCLFloorGenerator(),
      spirvCLLogGenerator(),
      spirvCLRintGenerator(),
      spirvCLRoundGenerator(),
      spirvCLRsqrtGenerator(),
      spirvCLSinGenerator(),
      spirvCLSqrtGenerator(),
      spirvCLTanhGenerator(),
      spirvBitwiseOrGenerator(),
      spirvBitwiseXorGenerator(),
      spirvCLCeilGenerator(),
      spirvCLCosGenerator(),
      spirvCLErfGenerator(),
      spirvCLExpGenerator(),
      spirvCLFAbsGenerator(),
      spirvCLFloorGenerator(),
      spirvCLLogGenerator(),
      spirvCLRintGenerator(),
      spirvCLRoundGenerator(),
      spirvCLRsqrtGenerator(),
      spirvCLSinGenerator(),
      spirvCLSqrtGenerator(),
      spirvCLTanhGenerator(),
      spirvFOrdEqualGenerator(),
      spirvFOrdGreaterThanEqualGenerator(),
      spirvFOrdGreaterThanGenerator(),
      spirvFOrdLessThanEqualGenerator(),
      spirvFOrdLessThanGenerator(),
      spirvFOrdNotEqualGenerator(),
      spirvFUnordEqualGenerator(),
      spirvFUnordGreaterThanEqualGenerator(),
      spirvFUnordGreaterThanGenerator(),
      spirvFUnordLessThanEqualGenerator(),
      spirvFUnordLessThanGenerator(),
      spirvFUnordNotEqualGenerator(),
      spirvIEqualGenerator(),
      spirvINotEqualGenerator(),
      spirvLogicalAndGenerator(),
      spirvLogicalOrGenerator(),
      spirvLogicalEqualGenerator(),
      spirvLogicalNotEqualGenerator(),
      spirvSGreaterThanEqualGenerator(),
      spirvSGreaterThanGenerator(),
      spirvSLessThanEqualGenerator(),
      spirvSLessThanGenerator(),
      spirvUGreaterThanEqualGenerator(),
      spirvUGreaterThanGenerator(),
      spirvULessThanEqualGenerator(),
      spirvULessThanGenerator(),
      spirvUnorderedGenerator(),
      spirvGLAcosGenerator(),
      spirvGLAsinGenerator(),
      spirvGLAtanGenerator(),
      spirvGLCeilGenerator(),
      spirvGLCosGenerator(),
      spirvGLCoshGenerator(),
      spirvGLExpGenerator(),
      spirvGLFAbsGenerator(),
      spirvGLFSignGenerator(),
      spirvGLFloorGenerator(),
      spirvGLInverseSqrtGenerator(),
      spirvGLLogGenerator(),
      spirvGLRoundEvenGenerator(),
      spirvGLRoundGenerator(),
      spirvGLSinGenerator(),
      spirvGLSinhGenerator(),
      spirvGLSqrtGenerator(),
      spirvGLTanGenerator(),
      spirvGLTanhGenerator(),
      spirvGLFClampGenerator(),
      spirvGLFMaxGenerator(),
      spirvGLFMinGenerator(),
      spirvGLFMixGenerator(), // spirvGLFindUMsbGenerator(), //TODO
      spirvGLFmaGenerator(),
      spirvGLLdexpGenerator(),
      spirvGLPowGenerator(),
      spirvGLSAbsGenerator(),
      spirvGLSClampGenerator(),
      spirvGLSMaxGenerator(),
      spirvGLSMinGenerator(),
      spirvGLSSignGenerator(),
      spirvGLUClampGenerator(),
      spirvGLUMaxGenerator(),
      spirvGLUMinGenerator(),
      spirvCLFMaxGenerator(),
      spirvCLFMinGenerator(),
      spirvCLFmaGenerator(),
      spirvCLPowGenerator(),
      spirvCLSAbsGenerator(),
      spirvCLSMaxGenerator(),
      spirvCLSMinGenerator(),
      spirvCLUMaxGenerator(),
      spirvCLUMinGenerator()};

  for (auto gen : spirvGens) {
    operators.insert(std::make_pair(gen.opName, gen));
    opsForFunc.insert(gen.opName);
    auto pos = gen.opName.find(".");
    auto dialect = gen.opName.substr(0, pos);
    auto id = gen.opName.substr(pos + 1, gen.opName.size());
    if (is_default_config) {
      conf.supported_ops.push_back(OpConf(dialect, id, UR(priority_base), {}));
    }
  }
}

bool isBoolOrBoolVector(TypeValue t, std::vector<uint64_t> validDims) {
  auto i = t.type.isInteger(1);
  auto v = t.type.dyn_cast<VectorType>();
  return i || (v && v.getElementType().isInteger(1) && v.getRank() == 1 &&
               std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
                   validDims.end());
}

bool isIntOrIntVector(TypeValue t, std::vector<uint64_t> validWidths,
                      std::vector<uint64_t> validDims) {
  auto i = t.val.getType().dyn_cast<IntegerType>();
  auto v = t.val.getType().dyn_cast<VectorType>();
  return (i && std::find(validWidths.begin(), validWidths.end(),
                         i.getWidth()) != validWidths.end()) ||
         (v && v.getElementType().dyn_cast<IntegerType>() && v.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
              validDims.end());
}

bool isFloatOrFloatVector(TypeValue t, std::vector<uint64_t> validWidths,
                          std::vector<uint64_t> validDims) {
  auto f = t.val.getType().dyn_cast<FloatType>();
  auto v = t.val.getType().dyn_cast<VectorType>();
  return (f && std::find(validWidths.begin(), validWidths.end(),
                         f.getWidth()) != validWidths.end()) ||
         (v && v.getElementType().dyn_cast<FloatType>() && v.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
              validDims.end());
}

// Note： T should be either Integer or Float type.
template <typename T>
bool isValidUnaryOperand(TypeValue t, std::vector<uint64_t> validWidths,
                         std::vector<uint64_t> validDims) {
  auto intTy = t.type.dyn_cast<T>();
  auto vecTy = t.type.dyn_cast<VectorType>();
  return (intTy && std::find(validWidths.begin(), validWidths.end(),
                             intTy.getWidth()) != validWidths.end()) ||
         (vecTy && vecTy.getElementType().dyn_cast<T>() &&
          std::find(validWidths.begin(), validWidths.end(),
                    vecTy.getElementType().getIntOrFloatBitWidth()) !=
              validWidths.end() &&
          vecTy.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), vecTy.getDimSize(0)) !=
              validDims.end());
}

template <typename T>
OpGen getSPIRVIntUnaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVFloatUnaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {8, 16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVFloat16Or32UnaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVIntBinaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op = builder.create<T>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVIntLogicalBinaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    Type resTy = builder.getI1Type();
    auto operandTy = operand.val.getType().template dyn_cast<VectorType>();
    if (operandTy) {
      resTy = VectorType::get(operandTy.getShape(), builder.getI1Type());
    }

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op =
        builder.create<T>(loc, resTy, operand.val,
                          sampleTypedValueFrom(operand2Candidates).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVFloatBinaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {8, 16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op = builder.create<T>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVBitBinaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vecCandidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [](TypeValue t) {
          if (t.val.getType().dyn_cast<VectorType>().getElementType().isInteger(
                  1)) {
            return false;
          }
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (vecCandidates.empty()) {
      vecCandidates.push_back(region.pool.generateVector(
          builder, loc, VectorType::get({2}, builder.getI32Type())));
    }
    auto operand1 = sampleTypedValueFrom(vecCandidates);

    auto operand2Candidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue t) {
          return t.type == operand1.val.getType();
        });
    auto operand2 = sampleTypedValueFrom(operand2Candidates);

    auto op = builder.create<T>(loc, operand1.val, operand2.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVBoolUnaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isBoolOrBoolVector(t, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI1Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVFUnaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Vector}, [](TypeValue t) {
          return isFloatOrFloatVector(t, {16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVIUnaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Vector}, [](TypeValue t) {
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen spirvBitFieldExtractGenerator(std::string uOrS) {
  auto opName = "spirv.BitField" + uOrS + "Extract";
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vecCandidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [](TypeValue t) {
          if (t.val.getType().dyn_cast<VectorType>().getElementType().isInteger(
                  1)) {
            return false;
          }
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (vecCandidates.empty()) {
      vecCandidates.push_back(region.pool.generateVector(
          builder, loc, VectorType::get({2}, builder.getI32Type())));
    }
    auto operand1 = sampleTypedValueFrom(vecCandidates);

    auto intCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat}, [](TypeValue t) {
          auto ty = t.val.getType().dyn_cast<IntegerType>();
          return ty && (ty.getWidth() == 8 || ty.getWidth() == 16 ||
                        ty.getWidth() == 32 || ty.getWidth() == 64);
        });
    if (intCandidates.empty()) {
      intCandidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand3 = sampleTypedValueFrom(intCandidates);
    auto operand4 = sampleTypedValueFrom(intCandidates);

    auto op = builder.create<T>(loc, operand1.val, operand3.val, operand4.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVFloatTriOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op =
        builder.create<T>(loc, operand.val.getType(), operand.val,
                          sampleTypedValueFrom(operand2Candidates).val,
                          sampleTypedValueFrom(operand2Candidates).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVIntTriOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op =
        builder.create<T>(loc, operand.val.getType(), operand.val,
                          sampleTypedValueFrom(operand2Candidates).val,
                          sampleTypedValueFrom(operand2Candidates).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

template <typename T>
OpGen getSPIRVBoolBinaryOpGenerator(std::string opName) {
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {1}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI1Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op = builder.create<T>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

OpGen spirvBitCountGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::BitCountOp>("spirv.BitCount");
}

OpGen spirvBitFieldInsertGenerator() {
  auto opName = "spirv.BitFieldInsert";
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vecCandidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [](TypeValue t) {
          if (t.val.getType().dyn_cast<VectorType>().getElementType().isInteger(
                  1)) {
            return false;
          }
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (vecCandidates.empty()) {
      vecCandidates.push_back(region.pool.generateVector(
          builder, loc, VectorType::get({2}, builder.getI32Type())));
    }
    auto operand1 = sampleTypedValueFrom(vecCandidates);
    auto operand2Candidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue t) {
          return t.type == operand1.val.getType();
        });
    auto operand2 = sampleTypedValueFrom(operand2Candidates);

    auto intCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat}, [](TypeValue t) {
          auto ty = t.val.getType().dyn_cast<IntegerType>();
          return ty && (ty.getWidth() == 8 || ty.getWidth() == 16 ||
                        ty.getWidth() == 32 || ty.getWidth() == 64);
        });
    if (intCandidates.empty()) {
      intCandidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand3 = sampleTypedValueFrom(intCandidates);
    auto operand4 = sampleTypedValueFrom(intCandidates);

    auto op = builder.create<spirv::BitFieldInsertOp>(
        loc, operand1.val, operand2.val, operand3.val, operand4.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

OpGen spirvBitFieldSExtractGenerator() {
  return spirvBitFieldExtractGenerator<spirv::BitFieldSExtractOp>("S");
}

OpGen spirvBitFieldUExtractGenerator() {
  return spirvBitFieldExtractGenerator<spirv::BitFieldUExtractOp>("U");
}

OpGen spirvBitReverseGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::BitReverseOp>("spirv.BitReverse");
}

OpGen spirvNotGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::NotOp>("spirv.Not");
}
OpGen spirvBitwiseAndGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseAndOp>("spirv.BitwiseAnd");
}

OpGen spirvBitwiseOrGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseOrOp>("spirv.BitwiseOr");
}

OpGen spirvBitwiseXorGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseXorOp>("spirv.BitwiseXor");
}

OpGen spirvCLCeilGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLCeilOp>("spirv.CL.ceil");
}

OpGen spirvCLCosGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLCosOp>("spirv.CL.cos");
}

OpGen spirvCLErfGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLErfOp>("spirv.CL.erf");
}

OpGen spirvCLExpGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLExpOp>("spirv.CL.exp");
}

OpGen spirvCLFAbsGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLFAbsOp>("spirv.CL.fabs");
}

OpGen spirvCLFloorGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLFloorOp>("spirv.CL.floor");
}

OpGen spirvCLLogGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLLogOp>("spirv.CL.log");
}

OpGen spirvCLRoundGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRoundOp>("spirv.CL.round");
}

OpGen spirvCLRintGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRintOp>("spirv.CL.rint");
}

OpGen spirvCLRsqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRsqrtOp>("spirv.CL.rsqrt");
}

OpGen spirvCLSinGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLSinOp>("spirv.CL.sin");
}

OpGen spirvCLSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLSqrtOp>("spirv.CL.sqrt");
}

OpGen spirvCLTanhGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLTanhOp>("spirv.CL.tanh");
}

OpGen spirvCLSAbsGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::CLSAbsOp>("spirv.CL.sabs");
}

OpGen spirvCLFMaxGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLFMaxOp>("spirv.CL.fmax");
}

OpGen spirvCLFMinGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLFMinOp>("spirv.CL.fmin");
}
OpGen spirvCLPowGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLPowOp>("spirv.CL.pow");
}

OpGen spirvCLSMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLSMaxOp>("spirv.CL.smax");
}

OpGen spirvCLSMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLSMinOp>("spirv.CL.smin");
}

OpGen spirvCLUMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLUMaxOp>("spirv.CL.umax");
}

OpGen spirvCLUMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLUMinOp>("spirv.CL.umin");
}

OpGen spirvCLFmaGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::CLFmaOp>("spriv.CL.fma");
}
// OpGen spirvCLPrintfGenerator(){};

// OpGen spirvBitcastGenerator(){};
// OpGen spirvBranchConditionalGenerator(){};
// OpGen spirvBranchGenerator(){};

// OpGen spirvCompositeConstructGenerator(){};
// OpGen spirvCompositeExtractGenerator(){};
// OpGen spirvCompositeInsertGenerator(){};
// OpGen spirvConstantGenerator(){};
// OpGen spirvControlBarrierGenerator(){};
// OpGen spirvConvertFToSGenerator(){};
// OpGen spirvConvertFToUGenerator(){};
// OpGen spirvConvertPtrToUGenerator(){};
// OpGen spirvConvertSToFGenerator(){};
// OpGen spirvConvertUToFGenerator(){};
// OpGen spirvConvertUToPtrGenerator(){};
// OpGen spirvCopyMemoryGenerator(){};
// OpGen spirvEXTAtomicFAddGenerator(){};
// OpGen spirvEntryPointGenerator(){};
// OpGen spirvExecutionModeGenerator(){};
// OpGen spirvFAddGenerator(){};
// OpGen spirvFConvertGenerator(){};
// OpGen spirvFDivGenerator(){};
// OpGen spirvFModGenerator(){};
// OpGen spirvFMulGenerator(){};

// TODO- matrix
OpGen spirvFNegateGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::FNegateOp>("spirv.FNegate");
}

OpGen spirvFOrdEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdEqualOp>("spirv.FOrdEqual");
}

OpGen spirvFOrdGreaterThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdGreaterThanEqualOp>(
      "spriv.FOrdGreaterThanEqual");
}

OpGen spirvFOrdGreaterThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdGreaterThanOp>(
      "spriv.FOrdGreaterThan");
}

OpGen spirvFOrdLessThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdLessThanEqualOp>(
      "spirv.FOrdLessThanEqual");
}

OpGen spirvFOrdLessThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdLessThanOp>(
      "spirv.FOrdLessThan");
}
OpGen spirvFOrdNotEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdNotEqualOp>(
      "spirv.FOrdNotEqual");
}

OpGen spirvFUnordEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordEqualOp>(
      "spirv.FUnordEqual");
}

OpGen spirvFUnordGreaterThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordGreaterThanEqualOp>(
      "spirv.FUnordGreaterThanEqual");
}

OpGen spirvFUnordGreaterThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordGreaterThanOp>(
      "spirv.FUnordGreaterThan");
}

OpGen spirvFUnordLessThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordLessThanEqualOp>(
      "spirv.FUnordLessThanEqual");
}

OpGen spirvFUnordLessThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordLessThanOp>(
      "spirv.FUnordLessThan");
}

OpGen spirvFUnordNotEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordNotEqualOp>(
      "spirv.FUnordNotEqual");
}

OpGen spirvIEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::IEqualOp>("spirv.IEqual");
}

OpGen spirvINotEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::INotEqualOp>(
      "spirv.INotEqual");
}

OpGen spirvLogicalEqualGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalEqualOp>(
      "spirv.LogicalEqual");
}

OpGen spirvLogicalNotGenerator() {
  return getSPIRVBoolUnaryOpGenerator<spirv::LogicalNotOp>("spirv.LogicalNot");
}

OpGen spirvLogicalNotEqualGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalNotEqualOp>(
      "spirv.LogicalNotEqual");
}

OpGen spirvLogicalAndGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalAndOp>("spirv.LogicalAnd");
}

OpGen spirvLogicalOrGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalOrOp>("spirv.LogicalOr");
}

OpGen spirvSGreaterThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SGreaterThanEqualOp>(
      "spirv.SGreaterThanEqual");
}

OpGen spirvSGreaterThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SGreaterThanOp>(
      "spirv.SGreaterThan");
}

OpGen spirvSLessThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SLessThanEqualOp>(
      "spirv.SLessThanEqual");
}

OpGen spirvSLessThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SLessThanOp>(
      "spirv.SLessEqual");
}

OpGen spirvUGreaterThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::UGreaterThanEqualOp>(
      "spirv.UGreaterThanEqual");
}

OpGen spirvUGreaterThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::UGreaterThanOp>(
      "spirv.UGreaterThan");
}

OpGen spirvULessThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::ULessThanEqualOp>(
      "spirv.ULessThanEqual");
}

OpGen spirvULessThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::ULessThanOp>(
      "spirv.ULessThan");
}

OpGen spirvUnorderedGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::UnorderedOp>("spirv.Unordered");
}

// OpGen spirvFRemGenerator(){};
// OpGen spirvFSubGenerator(){};
// OpGen spirvFuncGenerator(){};
// OpGen spirvFunctionCallGenerator(){};
OpGen spirvGLAcosGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAcosOp>("spirv.GL.Acos");
}

OpGen spirvGLAsinGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAsinOp>("spirv.GL.Asin");
}

OpGen spirvGLAtanGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAtanOp>("spirv.GL.Atan");
}

OpGen spirvGLCeilGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLCeilOp>("spirv.GL.Ceil");
}

OpGen spirvGLCosGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLCosOp>("spirv.GL.Cos");
}

OpGen spirvGLCoshGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLCoshOp>("spirv.GL.Cosh");
}

OpGen spirvGLExpGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLExpOp>("spirv.GL.Exp");
}

OpGen spirvGLFAbsGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFAbsOp>("spirv.GL.FAbs");
}

OpGen spirvGLFSignGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFSignOp>("spirv.GL.FSign");
}

OpGen spirvGLFloorGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFloorOp>("spirv.GL.Floor");
}

OpGen spirvGLInverseSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLInverseSqrtOp>(
      "spirv.GL.InverseSqrt");
}

OpGen spirvGLLogGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLLogOp>("spirv.GL.Log");
}

OpGen spirvGLRoundEvenGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLRoundEvenOp>(
      "spirv.GL.RoundEven");
}

OpGen spirvGLRoundGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLRoundOp>("spirv.GL.Round");
}

OpGen spirvGLSinGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLSinOp>("spirv.GL.Sin");
}

OpGen spirvGLSinhGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLSinhOp>("spirv.GL.Sinh");
}

OpGen spirvGLSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLSqrtOp>("spirv.GL.Sqrt");
}

OpGen spirvGLTanGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLTanOp>("spirv.GL.Tan");
}

OpGen spirvGLTanhGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLTanhOp>("spirv.GL.Tanh");
}

OpGen spirvGLFClampGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFClampOp>("spirv.GL.FClamp");
}

OpGen spirvGLSAbsGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::GLSAbsOp>("spirv.GL.SAbs");
}

OpGen spirvGLSSignGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::GLSSignOp>("spirv.GL.SSign");
}

OpGen spirvGLFMaxGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::GLFMaxOp>("spirv.GL.FMax");
}

OpGen spirvGLFMinGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::GLFMinOp>("spirv.GL.FMin");
}

OpGen spirvGLFMixGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFMixOp>("spirv.GL.FMix");
}

OpGen spirvGLFindUMsbGenerator() {
  auto opName = "spirv.GL.FindUMsb";
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {32}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates);
    auto op = builder.create<spirv::GLFindUMsbOp>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

OpGen spirvGLFmaGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFmaOp>("spriv.GL.fma");
}

OpGen spirvGLLdexpGenerator() {
  auto opName = "spirv.GL.Ldexp";
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates1 = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates1.empty()) {
      candidates1.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand1 = sampleTypedValueFrom(candidates1);
    auto isVec = false;
    VectorType ty;
    if (operand1.val.getType().dyn_cast<VectorType>()) {
      isVec = true;
      ty = VectorType::get(
          operand1.val.getType().dyn_cast<VectorType>().getShape(),
          builder.getI32Type());
    }
    auto candidates2 = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [&](TypeValue t) {
          auto validBitWidth = {8, 16, 32, 64};
          auto i = t.type.dyn_cast<IntegerType>();
          auto v = t.type.dyn_cast<VectorType>();
          return (!isVec && i &&
                  std::find(validBitWidth.begin(), validBitWidth.end(),
                            i.getWidth()) != validBitWidth.end()) ||
                 (isVec && v && v.getRank() == 1 &&
                  v.getDimSize(0) ==
                      operand1.val.getType().dyn_cast<VectorType>().getDimSize(
                          0) &&
                  v.getElementType().dyn_cast<IntegerType>() &&
                  std::find(
                      validBitWidth.begin(), validBitWidth.end(),
                      v.getElementType().dyn_cast<IntegerType>().getWidth()) !=
                      validBitWidth.end());
        });
    if (candidates2.empty()) {
      if (isVec) {

        candidates2.push_back(region.pool.generateVector(builder, loc, ty));

      } else {
        candidates2.push_back(
            region.pool.generateInteger(builder, loc, builder.getI32Type()));
      }
    }
    auto operand2 = sampleTypedValueFrom(candidates2);
    auto op = builder.create<spirv::GLLdexpOp>(loc, operand1.val, operand2.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

OpGen spirvGLPowGenerator() {
  auto opName = "spirv.GL.Pow";
  auto gen = [&](OpBuilder &builder, Location loc, OpRegion &region) {
    debugPrint("1");
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32}, {2, 3, 4, 8, 16});
        });
    debugPrint("2");
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    debugPrint("1");
    auto operand = sampleTypedValueFrom(candidates);

    debugPrint("2");
    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    debugPrint("1");
    auto op = builder.create<spirv::GLPowOp>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates).val);
    auto tval = TypeValue(op.getType(), op);
    debugPrint("2");
    region.pool.addTypeValue(tval);
    return op.getOperation();
  };
  return OpGen(opName, gen);
}

OpGen spirvGLSClampGenerator() {
  return getSPIRVIntTriOpGenerator<spirv::GLSClampOp>("spirv.GL.SClamp");
}

OpGen spirvGLSMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLSMaxOp>("spirv.GL.SMax");
}

OpGen spirvGLSMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLSMinOp>("spirv.GL.SMin");
}

OpGen spirvGLUMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLUMaxOp>("spirv.GL.UMax");
}

OpGen spirvGLUMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLUMaxOp>("spirv.GL.UMin");
}

OpGen spirvGLUClampGenerator() {
  return getSPIRVIntTriOpGenerator<spirv::GLUClampOp>("spirv.GL.UClamp");
}
//  OpGen spirvGenericCastToPtrExplicitGenerator(){};
//  OpGen spirvGenericCastToPtrGenerator(){};
//  OpGen spirvGlobalVariableGenerator(){};
//  OpGen spirvGroupBroadcastGenerator(){};
//  OpGen spirvGroupFAddGenerator(){};
//  OpGen spirvGroupFMaxGenerator(){};
//  OpGen spirvGroupFMinGenerator(){};
//  OpGen spirvGroupFMulKHRGenerator(){};
//  OpGen spirvGroupIAddGenerator(){};
//  OpGen spirvGroupIMulKHRGenerator(){};
//  OpGen spirvGroupNonUniformBallotGenerator(){};
//  OpGen spirvGroupNonUniformBroadcastGenerator(){};
//  OpGen spirvGroupNonUniformElectGenerator(){};
//  OpGen spirvGroupNonUniformFAddGenerator(){};
//  OpGen spirvGroupNonUniformFMaxGenerator(){};
//  OpGen spirvGroupNonUniformFMinGenerator(){};
//  OpGen spirvGroupNonUniformFMulGenerator(){};
//  OpGen spirvGroupNonUniformIAddGenerator(){};
//  OpGen spirvGroupNonUniformIMulGenerator(){};
//  OpGen spirvGroupNonUniformSMaxGenerator(){};
//  OpGen spirvGroupNonUniformSMinGenerator(){};
//  OpGen spirvGroupNonUniformShuffleDownGenerator(){};
//  OpGen spirvGroupNonUniformShuffleGenerator(){};
//  OpGen spirvGroupNonUniformShuffleUpGenerator(){};
//  OpGen spirvGroupNonUniformShuffleXorGenerator(){};
//  OpGen spirvGroupNonUniformUMaxGenerator(){};
//  OpGen spirvGroupNonUniformUMinGenerator(){};
//  OpGen spirvGroupSMaxGenerator(){};
//  OpGen spirvGroupSMinGenerator(){};
//  OpGen spirvGroupUMaxGenerator(){};
//  OpGen spirvGroupUMinGenerator(){};
//  OpGen spirvIAddCarryGenerator(){};
//  OpGen spirvIAddGenerator(){};
//  OpGen spirvIMulGenerator(){};
//  OpGen spirvINTELConvertBF16ToFGenerator(){};
//  OpGen spirvINTELConvertFToBF16Generator(){};
//  OpGen spirvINTELJointMatrixLoadGenerator(){};
//  OpGen spirvINTELJointMatrixMadGenerator(){};
//  OpGen spirvINTELJointMatrixStoreGenerator(){};
//  OpGen spirvINTELJointMatrixWorkItemLengthGenerator(){};
//  OpGen spirvINTELSubgroupBlockReadGenerator(){};
//  OpGen spirvINTELSubgroupBlockWriteGenerator(){};
//  OpGen spirvISubBorrowGenerator(){};
//  OpGen spirvISubGenerator(){};
//  OpGen spirvImageDrefGatherGenerator(){};
//  OpGen spirvImageGenerator(){};
//  OpGen spirvImageQuerySizeGenerator(){};
//  OpGen spirvInBoundsPtrAccessChainGenerator(){};

OpGen spirvIsInfGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::IsInfOp>("spirv.IsInf");
}

OpGen spirvIsNanGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::IsNanOp>("spirv.IsNan");
}

// OpGen spirvKHRAssumeTrueGenerator(){};
// OpGen spirvKHRCooperativeMatrixLengthGenerator(){};
// OpGen spirvKHRCooperativeMatrixLoadGenerator(){};
// OpGen spirvKHRCooperativeMatrixStoreGenerator(){};
// OpGen spirvKHRSubgroupBallotGenerator(){};
// OpGen spirvLoadGenerator(){};

// OpGen spirvLoopGenerator(){};
// OpGen spirvMatrixTimesMatrixGenerator(){};
// OpGen spirvMatrixTimesScalarGenerator(){};
// OpGen spirvMemoryBarrierGenerator(){};
// OpGen spirvMergeGenerator(){};
// OpGen spirvModuleGenerator(){};
// OpGen spirvNVCooperativeMatrixLengthGenerator(){};
// OpGen spirvNVCooperativeMatrixLoadGenerator(){};
// OpGen spirvNVCooperativeMatrixMulAddGenerator(){};
// OpGen spirvNVCooperativeMatrixStoreGenerator(){};

// OpGen spirvOrderedGenerator(){};
// OpGen spirvPtrAccessChainGenerator(){};
// OpGen spirvPtrCastToGenericGenerator(){};
// OpGen spirvReferenceOfGenerator(){};
// OpGen spirvReturnGenerator(){};
// OpGen spirvReturnValueGenerator(){};
// OpGen spirvSConvertGenerator(){};
// OpGen spirvSDivGenerator(){};
// OpGen spirvSDotAccSatGenerator(){};
// OpGen spirvSDotGenerator(){};
// OpGen spirvSModGenerator(){};
// OpGen spirvSMulExtendedGenerator(){};

// TODO-matrix
// OpGen spirvSNegateGenerator() {
//  return getSPIRVIntUnaryOpGenerator<spirv::SNegateOp>("spirv.SNegate");
//}
// TODO: need spirv.struct as result type.
// OpGen spirvGLFrexpStructGenerator() {
//  return
//  getSPIRVFloatUnaryOpGenerator<spirv::GLFrexpStructOp>("spirv.GL.Frexp");
//}

// OpGen spirvSRemGenerator(){};
// OpGen spirvSUDotAccSatGenerator(){};
// OpGen spirvSUDotGenerator(){};
// OpGen spirvSelectGenerator(){};
// OpGen spirvSelectionGenerator(){};
// OpGen spirvShiftLeftLogicalGenerator(){};
// OpGen spirvShiftRightArithmeticGenerator(){};
// OpGen spirvShiftRightLogicalGenerator(){};
// OpGen spirvSpecConstantCompositeGenerator(){};
// OpGen spirvSpecConstantGenerator(){};
// OpGen spirvSpecConstantOperationGenerator(){};
// OpGen spirvStoreGenerator(){};
// OpGen spirvTransposeGenerator(){};
// OpGen spirvUConvertGenerator(){};
// OpGen spirvUDivGenerator(){};
// OpGen spirvUDotAccSatGenerator(){};
// OpGen spirvUDotGenerator(){};

// OpGen spirvUModGenerator(){};
// OpGen spirvUMulExtendedGenerator(){};
// OpGen spirvUndefGenerator(){};
// OpGen spirvUnreachableGenerator(){};
// OpGen spirvVariableGenerator(){};
// OpGen spirvVectorExtractDynamicGenerator(){};
// OpGen spirvVectorInsertDynamicGenerator(){};
// OpGen spirvVectorShuffleGenerator(){};
// OpGen spirvVectorTimesScalarGenerator(){};
// OpGen spirvYieldGenerator(){};