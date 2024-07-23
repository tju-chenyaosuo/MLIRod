#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"


// Note： T should be either Integer or Float type.
template <typename T>
bool isValidUnaryOperand(mlir::Value t, std::vector<uint64_t> validWidths,
                         std::vector<uint64_t> validDims) {
  auto intTy = t.getType().dyn_cast<T>();
  auto vecTy = t.getType().dyn_cast<VectorType>();
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

bool isBoolOrBoolVector(mlir::Value t, std::vector<uint64_t> validDims) {
  auto i = t.getType().isInteger(1);
  auto v = t.getType().dyn_cast<VectorType>();
  return i || (v && v.getElementType().isInteger(1) && v.getRank() == 1 &&
               std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
                   validDims.end());
}

bool isIntOrIntVector(mlir::Value t, std::vector<uint64_t> validWidths,
                      std::vector<uint64_t> validDims) {
  auto i = t.getType().dyn_cast<IntegerType>();
  auto v = t.getType().dyn_cast<VectorType>();
  return (i && std::find(validWidths.begin(), validWidths.end(),
                         i.getWidth()) != validWidths.end()) ||
         (v && v.getElementType().dyn_cast<IntegerType>() && v.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
              validDims.end());
}

bool isFloatOrFloatVector(mlir::Value t, std::vector<uint64_t> validWidths,
                          std::vector<uint64_t> validDims) {
  auto f = t.getType().dyn_cast<FloatType>();
  auto v = t.getType().dyn_cast<VectorType>();
  return (f && std::find(validWidths.begin(), validWidths.end(),
                         f.getWidth()) != validWidths.end()) ||
         (v && v.getElementType().dyn_cast<FloatType>() && v.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
              validDims.end());
}

template <typename T>
OpGenerator getSPIRVIntUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<IntegerType>(v, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }

		auto operand = candidates[rollIdx(candidates.size())];
		
    mlir::Value op = builder.create<T>(loc, operand);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVFloatUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<FloatType>(v, {8, 16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];

    mlir::Value op = builder.create<T>(loc, operand);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVFloat16Or32UnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<FloatType>(v, {16, 32}, {2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];
		mlir::Value op = builder.create<T>(loc, operand);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVIntBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<IntegerType>(v, {8, 16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandStr = getValueTypeStr(operand);
		if (mb->valuePool.pool.count(operandStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandStr].begin(), 
				mb->valuePool.pool[operandStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand.getType(), mb));
		}
		auto operand2 = operand2Candidates[rollIdx(operand2Candidates.size())];

    mlir::Value op = builder.create<T>(loc, operand, operand2);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVIntLogicalBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<IntegerType>(v, {8, 16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];
		Type resTy = builder.getI1Type();
    auto operandTy = operand.getType().template dyn_cast<VectorType>();
    if (operandTy) {
      resTy = VectorType::get(operandTy.getShape(), builder.getI1Type());
    }

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand.getType(), mb));
		}

		auto operand2 = operand2Candidates[rollIdx(operand2Candidates.size())];

		mlir::Value op = builder.create<T>(loc, resTy, operand, operand2);
		mb->add2Pool(op);
		return true;
  };
}
template <typename T>
OpGenerator getSPIRVFloatBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<FloatType>(v, {8, 16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand.getType(), mb));
		}
		auto operand2 = operand2Candidates[rollIdx(operand2Candidates.size())];

		mlir::Value op = builder.create<T>(loc, operand, operand2);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVBoolUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isBoolOrBoolVector(v, {2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI1Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];
		mlir::Value op = builder.create<T>(loc, operand);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVFUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isFloatOrFloatVector(v, {16, 32, 64}, {2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];
		mlir::Value op = builder.create<T>(loc, operand);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVIUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isIntOrIntVector(v, {8, 16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];
		mlir::Value op = builder.create<T>(loc, operand);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator spirvBitCountGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::BitCountOp>("spirv.BitCount");
}

OpGenerator spirvBitFieldInsertGenerator() {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> vecCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
				bool cond = v.getType().dyn_cast<VectorType>().getElementType().isInteger(1);
				if (!cond) return false;
				return isIntOrIntVector(v, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
			}, 
			&vecCandidates
		);
		if (vecCandidates.empty()) {
      vecCandidates.push_back(
				generateVector(builder, loc, VectorType::get({2}, builder.getI32Type()), mb));
    }
		auto operand1 = vecCandidates[rollIdx(vecCandidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand1);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand1.getType(), mb));
		}
		auto operand2 = operand2Candidates[rollIdx(operand2Candidates.size())];

		std::vector<mlir::Value> intCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!int_float_filter(s)) return false;
				auto ty = v.getType().dyn_cast<IntegerType>();
				return ty && (ty.getWidth() == 8 || ty.getWidth() == 16 ||
											ty.getWidth() == 32 || ty.getWidth() == 64);
			}, 
			&intCandidates
		);
		if (intCandidates.empty()) {
      intCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand3 = intCandidates[rollIdx(intCandidates.size())];
		auto operand4 = intCandidates[rollIdx(intCandidates.size())];

		mlir::Value op = builder.create<spirv::BitFieldInsertOp>(
			loc, operand1, operand2, operand3, operand4);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator spirvBitFieldExtractGenerator(std::string uOrS) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> vecCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
				if (v.getType().dyn_cast<VectorType>().getElementType().isInteger(1)) {
					return false;
				}
				return isIntOrIntVector(v, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
			}, 
			&vecCandidates
		);
		if (vecCandidates.empty()) {
      vecCandidates.push_back(
				generateVector(builder, loc, VectorType::get({2}, builder.getI32Type()), mb));
    }
		auto operand1 = vecCandidates[rollIdx(vecCandidates.size())];

		std::vector<mlir::Value> intCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!int_float_filter(s)) return false;
				auto ty = v.getType().dyn_cast<IntegerType>();
				return ty && (ty.getWidth() == 8 || ty.getWidth() == 16 ||
											ty.getWidth() == 32 || ty.getWidth() == 64);
			}, 
			&intCandidates
		);
		if (intCandidates.empty()) {
      intCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand3 = intCandidates[rollIdx(intCandidates.size())];
		auto operand4 = intCandidates[rollIdx(intCandidates.size())];

		mlir::Value op = builder.create<T>(loc, operand1, operand3, operand4);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator spirvBitFieldSExtractGenerator() {
  return spirvBitFieldExtractGenerator<spirv::BitFieldSExtractOp>("S");
}

OpGenerator spirvBitFieldUExtractGenerator() {
  return spirvBitFieldExtractGenerator<spirv::BitFieldUExtractOp>("U");
}

OpGenerator spirvBitReverseGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::BitReverseOp>("spirv.BitReverse");
}

OpGenerator spirvNotGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::NotOp>("spirv.Not");
}

template <typename T>
OpGenerator getSPIRVBitBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> vecCandidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				if (!vector_filter(s)) return false;
				if (v.getType().dyn_cast<VectorType>().getElementType().isInteger(1)) {
            return false;
          }
          return isIntOrIntVector(v, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
			}, 
			&vecCandidates
		);
		if (vecCandidates.empty()) {
      vecCandidates.push_back(
				generateVector(builder, loc, VectorType::get({2}, builder.getI32Type()), mb));
    }
		auto operand1 = vecCandidates[rollIdx(vecCandidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand1);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand1.getType(), mb));
		}
		auto operand2 = operand2Candidates[rollIdx(operand2Candidates.size())];
		mlir::Value op = builder.create<T>(loc, operand1, operand2);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator spirvBitwiseAndGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseAndOp>("spirv.BitwiseAnd");
}

OpGenerator spirvBitwiseOrGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseOrOp>("spirv.BitwiseOr");
}

OpGenerator spirvBitwiseXorGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseXorOp>("spirv.BitwiseXor");
}

OpGenerator spirvCLCeilGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLCeilOp>("spirv.CL.ceil");
}

OpGenerator spirvCLCosGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLCosOp>("spirv.CL.cos");
}

OpGenerator spirvCLErfGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLErfOp>("spirv.CL.erf");
}

OpGenerator spirvCLExpGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLExpOp>("spirv.CL.exp");
}

OpGenerator spirvCLFAbsGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLFAbsOp>("spirv.CL.fabs");
}

OpGenerator spirvCLFloorGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLFloorOp>("spirv.CL.floor");
}

OpGenerator spirvCLLogGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLLogOp>("spirv.CL.log");
}

OpGenerator spirvCLRoundGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRoundOp>("spirv.CL.round");
}

OpGenerator spirvCLRintGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRintOp>("spirv.CL.rint");
}

OpGenerator spirvCLRsqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRsqrtOp>("spirv.CL.rsqrt");
}

OpGenerator spirvCLSinGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLSinOp>("spirv.CL.sin");
}

OpGenerator spirvCLSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLSqrtOp>("spirv.CL.sqrt");
}

OpGenerator spirvCLTanhGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLTanhOp>("spirv.CL.tanh");
}

OpGenerator spirvCLSAbsGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::CLSAbsOp>("spirv.CL.sabs");
}

OpGenerator spirvCLFMaxGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLFMaxOp>("spirv.CL.fmax");
}

OpGenerator spirvCLFMinGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLFMinOp>("spirv.CL.fmin");
}
OpGenerator spirvCLPowGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLPowOp>("spirv.CL.pow");
}

OpGenerator spirvCLSMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLSMaxOp>("spirv.CL.smax");
}

OpGenerator spirvCLSMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLSMinOp>("spirv.CL.smin");
}

OpGenerator spirvCLUMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLUMaxOp>("spirv.CL.umax");
}

OpGenerator spirvCLUMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLUMinOp>("spirv.CL.umin");
}

template <typename T>
OpGenerator getSPIRVFloatTriOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<FloatType>(v, {16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand.getType(), mb));
		}

		mlir::Value op = builder.create<T>(loc, operand.getType(), operand,
                          operand2Candidates[rollIdx(operand2Candidates.size())],
                          operand2Candidates[rollIdx(operand2Candidates.size())]);
		mb->add2Pool(op);
		return true;
  };
}

template <typename T>
OpGenerator getSPIRVIntTriOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<IntegerType>(v, {8, 16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand.getType(), mb));
		}

		mlir::Value op = builder.create<T>(loc, operand.getType(), operand,
                          operand2Candidates[rollIdx(operand2Candidates.size())],
                          operand2Candidates[rollIdx(operand2Candidates.size())]);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator spirvCLFmaGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::CLFmaOp>("spriv.CL.fma");
}

// TODO- matrix
OpGenerator spirvFNegateGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::FNegateOp>("spirv.FNegate");
}

OpGenerator spirvFOrdEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdEqualOp>("spirv.FOrdEqual");
}

OpGenerator spirvFOrdGreaterThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdGreaterThanEqualOp>(
      "spriv.FOrdGreaterThanEqual");
}

OpGenerator spirvFOrdGreaterThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdGreaterThanOp>(
      "spriv.FOrdGreaterThan");
}

OpGenerator spirvFOrdLessThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdLessThanEqualOp>(
      "spirv.FOrdLessThanEqual");
}

OpGenerator spirvFOrdLessThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdLessThanOp>(
      "spirv.FOrdLessThan");
}
OpGenerator spirvFOrdNotEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdNotEqualOp>(
      "spirv.FOrdNotEqual");
}

OpGenerator spirvFUnordEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordEqualOp>(
      "spirv.FUnordEqual");
}

OpGenerator spirvFUnordGreaterThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordGreaterThanEqualOp>(
      "spirv.FUnordGreaterThanEqual");
}

OpGenerator spirvFUnordGreaterThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordGreaterThanOp>(
      "spirv.FUnordGreaterThan");
}

OpGenerator spirvFUnordLessThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordLessThanEqualOp>(
      "spirv.FUnordLessThanEqual");
}

OpGenerator spirvFUnordLessThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordLessThanOp>(
      "spirv.FUnordLessThan");
}

OpGenerator spirvFUnordNotEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordNotEqualOp>(
      "spirv.FUnordNotEqual");
}

OpGenerator spirvIEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::IEqualOp>("spirv.IEqual");
}

OpGenerator spirvINotEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::INotEqualOp>(
      "spirv.INotEqual");
}

template <typename T>
OpGenerator getSPIRVBoolBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<IntegerType>(v, {1},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI1Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand.getType(), mb));
		}

		mlir::Value op = builder.create<T>(
			loc, operand, operand2Candidates[rollIdx(operand2Candidates.size())]);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator spirvLogicalEqualGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalEqualOp>(
      "spirv.LogicalEqual");
}

OpGenerator spirvLogicalNotGenerator() {
  return getSPIRVBoolUnaryOpGenerator<spirv::LogicalNotOp>("spirv.LogicalNot");
}

OpGenerator spirvLogicalNotEqualGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalNotEqualOp>(
      "spirv.LogicalNotEqual");
}

OpGenerator spirvLogicalAndGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalAndOp>("spirv.LogicalAnd");
}

OpGenerator spirvLogicalOrGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalOrOp>("spirv.LogicalOr");
}

OpGenerator spirvSGreaterThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SGreaterThanEqualOp>(
      "spirv.SGreaterThanEqual");
}

OpGenerator spirvSGreaterThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SGreaterThanOp>(
      "spirv.SGreaterThan");
}

OpGenerator spirvSLessThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SLessThanEqualOp>(
      "spirv.SLessThanEqual");
}

OpGenerator spirvSLessThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SLessThanOp>(
      "spirv.SLessEqual");
}

OpGenerator spirvUGreaterThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::UGreaterThanEqualOp>(
      "spirv.UGreaterThanEqual");
}

OpGenerator spirvUGreaterThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::UGreaterThanOp>(
      "spirv.UGreaterThan");
}

OpGenerator spirvULessThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::ULessThanEqualOp>(
      "spirv.ULessThanEqual");
}

OpGenerator spirvULessThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::ULessThanOp>(
      "spirv.ULessThan");
}

OpGenerator spirvUnorderedGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::UnorderedOp>("spirv.Unordered");
}

OpGenerator spirvGLAcosGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAcosOp>("spirv.GL.Acos");
}

OpGenerator spirvGLAsinGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAsinOp>("spirv.GL.Asin");
}

OpGenerator spirvGLAtanGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAtanOp>("spirv.GL.Atan");
}

OpGenerator spirvGLCeilGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLCeilOp>("spirv.GL.Ceil");
}

OpGenerator spirvGLCosGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLCosOp>("spirv.GL.Cos");
}

OpGenerator spirvGLCoshGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLCoshOp>("spirv.GL.Cosh");
}

OpGenerator spirvGLExpGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLExpOp>("spirv.GL.Exp");
}

OpGenerator spirvGLFAbsGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFAbsOp>("spirv.GL.FAbs");
}

OpGenerator spirvGLFSignGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFSignOp>("spirv.GL.FSign");
}

OpGenerator spirvGLFloorGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFloorOp>("spirv.GL.Floor");
}

OpGenerator spirvGLInverseSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLInverseSqrtOp>(
      "spirv.GL.InverseSqrt");
}

OpGenerator spirvGLLogGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLLogOp>("spirv.GL.Log");
}

OpGenerator spirvGLRoundEvenGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLRoundEvenOp>(
      "spirv.GL.RoundEven");
}

OpGenerator spirvGLRoundGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLRoundOp>("spirv.GL.Round");
}

OpGenerator spirvGLSinGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLSinOp>("spirv.GL.Sin");
}

OpGenerator spirvGLSinhGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLSinhOp>("spirv.GL.Sinh");
}

OpGenerator spirvGLSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLSqrtOp>("spirv.GL.Sqrt");
}

OpGenerator spirvGLTanGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLTanOp>("spirv.GL.Tan");
}

OpGenerator spirvGLTanhGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLTanhOp>("spirv.GL.Tanh");
}

OpGenerator spirvGLFClampGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFClampOp>("spirv.GL.FClamp");
}

OpGenerator spirvGLSAbsGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::GLSAbsOp>("spirv.GL.SAbs");
}

OpGenerator spirvGLSSignGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::GLSSignOp>("spirv.GL.SSign");
}

OpGenerator spirvGLFMaxGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::GLFMaxOp>("spirv.GL.FMax");
}

OpGenerator spirvGLFMinGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::GLFMinOp>("spirv.GL.FMin");
}

OpGenerator spirvGLFMixGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFMixOp>("spirv.GL.FMix");
}

OpGenerator spirvGLFindUMsbGenerator() {
  auto opName = "spirv.GL.FindUMsb";
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<IntegerType>(v, {32},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];
		mlir::Value op = builder.create<spirv::GLFindUMsbOp>(loc, operand);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator spirvGLFmaGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFmaOp>("spriv.GL.fma");
}

OpGenerator spirvGLLdexpGenerator() {
  auto opName = "spirv.GL.Ldexp";
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates1 = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<FloatType>(v, {16, 32, 64},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates1
		);
		if (candidates1.empty()) {
      candidates1.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand1 = candidates1[rollIdx(candidates1.size())];

		auto isVec = false;
    VectorType ty;
		if (operand1.getType().dyn_cast<VectorType>()) {
      isVec = true;
      ty = VectorType::get(
          operand1.getType().dyn_cast<VectorType>().getShape(),
          builder.getI32Type());
    }

		std::vector<mlir::Value> candidates2 = std::vector<mlir::Value>();
		mb->search(
			[&] (std::string s, mlir::Value t) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				auto validBitWidth = {8, 16, 32, 64};
          auto i = t.getType().dyn_cast<IntegerType>();
          auto v = t.getType().dyn_cast<VectorType>();
          return (!isVec && i &&
                  std::find(validBitWidth.begin(), validBitWidth.end(),
                            i.getWidth()) != validBitWidth.end()) ||
                 (isVec && v && v.getRank() == 1 &&
                  v.getDimSize(0) ==
                      operand1.getType().dyn_cast<VectorType>().getDimSize(
                          0) &&
                  v.getElementType().dyn_cast<IntegerType>() &&
                  std::find(
                      validBitWidth.begin(), validBitWidth.end(),
                      v.getElementType().dyn_cast<IntegerType>().getWidth()) !=
                      validBitWidth.end());
			}, 
			&candidates2
		);
		if (candidates2.empty()) {
      if (isVec){
         candidates2.push_back(generateVector(builder, loc, ty, mb));
      }else {
        candidates2.push_back(generateInteger(builder, loc, builder.getI32Type()));
      }    
		}
		auto operand2 = candidates2[rollIdx(candidates2.size())];
		mlir::Value op = builder.create<spirv::GLLdexpOp>(loc, operand1, operand2);
		mb->add2Pool(op);
		return true;
  };
}
// TODO-
OpGenerator spirvGLPowGenerator() {
  return [&](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(
			[] (std::string s, mlir::Value v) -> bool {
				bool cond1 = vector_filter(s);
				bool cond2 = int_float_filter(s);
				if (!cond1 && !cond2) return false;
				return isValidUnaryOperand<FloatType>(v, {16, 32},
																									{2, 3, 4, 8, 16});
			}, 
			&candidates
		);
		if (candidates.empty()) {
      candidates.push_back(generateFloat(builder, loc, builder.getF32Type()));
    }
		auto operand = candidates[rollIdx(candidates.size())];

		std::vector<mlir::Value> operand2Candidates = std::vector<mlir::Value>();
		std::string operandTyStr = getValueTypeStr(operand);
		if (mb->valuePool.pool.count(operandTyStr)) {
			operand2Candidates.insert(
				operand2Candidates.begin(), 
				mb->valuePool.pool[operandTyStr].begin(), 
				mb->valuePool.pool[operandTyStr].end());
		}
		if (operand2Candidates.empty()) {
			operand2Candidates.push_back(generateTypedValue(builder, loc, operand.getType(), mb));
		}
		mlir::Value op = builder.create<spirv::GLPowOp>(
			loc, operand, operand2Candidates[rollIdx(operand2Candidates.size())]);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator spirvGLSClampGenerator() {
  return getSPIRVIntTriOpGenerator<spirv::GLSClampOp>("spirv.GL.SClamp");
}

OpGenerator spirvGLSMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLSMaxOp>("spirv.GL.SMax");
}

OpGenerator spirvGLSMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLSMinOp>("spirv.GL.SMin");
}

OpGenerator spirvGLUMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLUMaxOp>("spirv.GL.UMax");
}

OpGenerator spirvGLUMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLUMaxOp>("spirv.GL.UMin");
}

OpGenerator spirvGLUClampGenerator() {
	return getSPIRVIntTriOpGenerator<spirv::GLUClampOp>("spirv.GL.UClamp");
}

OpGenerator spirvIsInfGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::IsInfOp>("spirv.IsInf");
}

OpGenerator spirvIsNanGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::IsNanOp>("spirv.IsNan");
}