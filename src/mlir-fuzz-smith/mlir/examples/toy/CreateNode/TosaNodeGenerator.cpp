#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

// Operator: tosa::AbsOp
// AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1)
OpGenerator tosaAbsGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &operandCandidates);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto src = operandCandidates[rollIdx(operandCandidates.size())];
		auto output = src.getType();
    auto input1 = src;
    mlir::Value res_tensor = builder.create<mlir::tosa::AbsOp>(loc, output, input1);
		mb->add2Pool(res_tensor);
		return true;
  };
}

// Operator: mlir::tosa::AddOp
// Documentation: 
// input1	tensor of number values
// input2	tensor of number values
// output	tensor of number values
// Builder Definition File: ./include/mlir/Dialect/Tosa/IR/TosaOps.td
// Selected Builder Function:
// void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, 
//                   ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2);
OpGenerator tosaAddGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &operandCandidates);
		if (operandCandidates.empty()) {
      operandCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

		std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
		std::string src1TyStr = getValueTypeStr(src1);
		if (mb->valuePool.pool.count(src1TyStr)) {
			src2Candidates.insert(
				src2Candidates.begin(), 
				mb->valuePool.pool[src1TyStr].begin(), 
				mb->valuePool.pool[src1TyStr].end());
		}
		if (src2Candidates.empty()) {
			src2Candidates.push_back(
				generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
		}
		auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

		auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();

		mlir::Value res_tensor = builder.create<mlir::tosa::AddOp>(loc, output, input1, input2);
		mb->add2Pool(res_tensor);
		return true;
  };
}

// Operator: mlir::tosa::ApplyScaleOp
// Documentation: 
// double_round	::mlir::BoolAttr	bool attribute
// value	signless-integer-like
// multiplier	signless-integer-like
// shift	signless-integer-8-bit-like
// output	signless-integer-like
// Builder Definition File: ./include/mlir/Dialect/Tosa/IR/TosaOps.td
// Selected Builder Function:
// void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, 
//                          ::mlir::Type output, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, bool double_round)
OpGenerator tosaApplyScaleGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(int_filter, &candidates);
		if (candidates.empty()) {
      candidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
    }
		auto value_vt = candidates[rollIdx(candidates.size())];
		auto multiplier_vt = candidates[rollIdx(candidates.size())];
		auto shift_vt = generateInteger(builder, loc, builder.getI8Type());
		auto value = value_vt;
    auto multiplier = multiplier_vt;
    auto shift = shift_vt;
    bool double_round = rollIdx(2);
    auto output = value_vt.getType().dyn_cast<IntegerType>();

		mlir::Value res = builder.create<mlir::tosa::ApplyScaleOp>(
			loc, output, value, multiplier, shift, double_round);
    mb->add2Pool(res);
    return true;
  };
}

// Operator:tosa.argmax (mlir::tosa::ArgMaxOp) 
//export PATH=/data/hqy/mlirsmith-dev/build/tools/llvm-symbolizer:$PATH

// axis	::mlir::IntegerAttr	64-bit signless integer attribute
// input	tensor of number values
// output	tensor of number values

//include/mlir/Dialect/Tosa/IR/TosaOps.td
//void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis)
//void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) 
OpGenerator tosaArgMaxGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		const int dim_ub = 32;

		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(
			[&] (std::string s, mlir::Value v) -> bool {
				if (!ranked_tensor_filter(s, v)) return false;
        if (auto t = v.getType().dyn_cast<RankedTensorType>()) {
          bool rankCond = t.getRank() > 0 && t.getRank() <= 4;
          bool elementTyCond = t.getElementType().isIntOrIndex();
          return rankCond && elementTyCond;
        }
				return false;
      },
			&operandCandidates
		);
		if(operandCandidates.empty()){
      SmallVector<int64_t> new_shape;
      new_shape.push_back(1);
      for(int i=0;i<rollIdx(3); i++){
        new_shape.push_back(rollIdx(dim_ub)+1);
      }
      operandCandidates.push_back(
				generateRankedTensor(builder, loc, RankedTensorType::get(new_shape, randomIntType(builder.getContext())), mb));
    }

		auto src = operandCandidates[rollIdx(operandCandidates.size())];
		auto input = src;
    uint64_t axis = rollIdx(input.getType().dyn_cast<RankedTensorType>().getRank());

		SmallVector<int64_t> shape;
    auto srcTy = src.getType().dyn_cast<RankedTensorType>();

		if (srcTy.hasStaticShape()) {
      for (uint64_t i = 0; i < srcTy.getRank(); ++i) {
        if(i==axis) continue;
        shape.push_back(ShapedType::kDynamic);
      }
    } else {
      for (uint64_t i = 0; i < srcTy.getRank(); ++i) {
        if(i==axis) continue;
        if (srcTy.isDynamicDim(i)) {
          shape.push_back(rollIdx(dim_ub)+1);
        } else {
          shape.push_back(srcTy.getDimSize(i));
        }
      }
    }

		auto output = RankedTensorType::get(shape ,srcTy.getElementType());

		mlir::Value res = builder.create<mlir::tosa::ArgMaxOp>(loc, output, input, axis);
		mb->add2Pool(res);
		return true;
  };
}

/*
tosa.arithmetic_right_shift (mlir::tosa::ArithmeticRightShiftOp) 
Elementwise arithmetic right shift of input1 by the amount specified in input2. Axis of size 1 will be broadcast, as necessary.

round	::mlir::BoolAttr	bool attribute
input1	tensor of number values
input2	tensor of number values
output	tensor of number values
void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, 
::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, bool round)
*/
OpGenerator tosaArithmeticRightShiftGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &operandCandidates);
		if(operandCandidates.empty()){
      operandCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
		
    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

		auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType().dyn_cast<RankedTensorType>();
    bool round = rollIdx(2);

		mlir::Value res = builder.create<mlir::tosa::ArithmeticRightShiftOp>(
			loc, output, input1, input2, round);
		mb->add2Pool(res);
    return true;
  };
}

/*tosa.avg_pool2d (mlir::tosa::AvgPool2dOp) ¶
Performs max pooling on the input.
This performs an average pooling over the given input tensor. A sliding window of size given by is passed over the input tensor, with the mean value being placed in the output tensor.

kernel	::mlir::DenseI64ArrayAttr	i64 dense array attribute with exactly 2 elements
stride	::mlir::DenseI64ArrayAttr	i64 dense array attribute with exactly 2 elements
pad	::mlir::DenseI64ArrayAttr	i64 dense array attribute with exactly 4 elements
acc_type	::mlir::TypeAttr	type attribute of 32-bit signless integer or 32-bit signed integer or 16-bit float or 32-bit float
quantization_info	mlir::tosa::UnaryOpQuantizationAttr	Attribute for UnaryOp quantization information.
input	unranked tensor of number values or 4D tensor of number values
output	unranked tensor of number values or 4D tensor of number values
void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputType, ::mlir::Value input, ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr s
tride, ::mlir::DenseI64ArrayAttr pad)
*/


/*tosa.bitwise_and (mlir::tosa::BitwiseAndOp) 
Bitwise AND operator
Elementwise bitwise AND of input1 and input2. Axis of size 1 will be broadcast as necessary.

input1	tensor of number values
input2	tensor of number values
output	tensor of number values

void BitwiseAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseAndGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &operandCandidates);
		if(operandCandidates.empty()){
      operandCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
		auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType().dyn_cast<RankedTensorType>();
    mlir::Value res = builder.create<mlir::tosa::BitwiseAndOp>(loc, output, input1, input2);
    mb->add2Pool(res);
    return true;
  };
}



/*tosa.bitwise_not (mlir::tosa::BitwiseNotOp) 

input1	tensor of number values
output	tensor of number values

*/
OpGenerator tosaBitwiseNotGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src;
    auto output = src.getType().dyn_cast<RankedTensorType>();
    mlir::Value res = builder.create<mlir::tosa::BitwiseNotOp>(loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}



/*tosa.bitwise_or (mlir::tosa::BitwiseOrOp) 
input1	tensor of number values
input2	tensor of number values
output	tensor of number values

void BitwiseOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseOrGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType().dyn_cast<RankedTensorType>();
    mlir::Value res = builder.create<mlir::tosa::BitwiseOrOp>(loc, output, input1, input2);
    mb->add2Pool(res);
    return true;
  };
}


/*tosa.bitwise_xor (mlir::tosa::BitwiseXorOp) 
input1	tensor of number values
input2	tensor of number values
output	tensor of number values
void BitwiseXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseXorGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];
    
    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType().dyn_cast<RankedTensorType>();
    mlir::Value res = builder.create<mlir::tosa::BitwiseXorOp>(loc, output, input1, input2);
    mb->add2Pool(res);
    return true;
  };
}

/*tosa.cast (mlir::tosa::CastOp)
void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) 
Performs a set of permissible cast operations

input:tensor of number_plus_f64 values
output	tensor of number_plus_f64 values
void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) 
*/

OpGenerator tosaCastGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    const int dim_ub = 32;

    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (auto t = v.getType().dyn_cast<TensorType>()) {
          return t.getElementType() == builder.getI8Type();
        }
        return false;
      }, 
      &operandCandidates
    );
    if(operandCandidates.empty()){
      SmallVector<int64_t> shape;
      
      for(int i=0;i<rollIdx(dim_ub)+1; i++){
        shape.push_back(rollIdx(dim_ub)+1);
      }
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, RankedTensorType::get(shape,builder.getI8Type()), mb));
    }
    auto src = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src;
    auto shape = src.getType().dyn_cast<RankedTensorType>().getShape();
    auto output = RankedTensorType::get(shape,builder.getI16Type());

    mlir::Value res = builder.create<mlir::tosa::CastOp>(loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}


/*tosa.ceil (mlir::tosa::CeilOp)
:Elementwise ceiling operation
input1	tensor of number values
output	tensor of number values
void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) 
*/
OpGenerator tosaCeilGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src;
    auto output = src.getType().dyn_cast<RankedTensorType>();
    mlir::Value res = builder.create<mlir::tosa::CeilOp>(loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}

//tosa.clamp (mlir::tosa::ClampOp)
//void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t min_int, uint64_t max_int, ::llvm::APFloat min
//_fp, ::llvm::APFloat max_fp) 
// min_int	::mlir::IntegerAttr	64-bit signless integer attribute
// max_int	::mlir::IntegerAttr	64-bit signless integer attribute
// min_fp	::mlir::FloatAttr	32-bit float attribute
// max_fp	::mlir::FloatAttr	32-bit float attribute

// OpGenerator tosaClampGenerator() {
//   return [](OpBuilder &builder, Location loc, OpRegion &region) {
//     auto operandCandidates = region.pool.searchCandidatesFrom(
//       {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
//       emptyFilter());
//     if (operandCandidates.empty()) {
//         operandCandidates.push_back(
//           region.pool.generateRankedTensor(builder, loc, 
//           randomRankedTensorType(builder.getContext())));
//     }
//     auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.clamp");

//     auto input = src1.val;
//     auto output = src1.type;
//     uint64_t max_int = ((unsigned long long)rand() << 32) | rand();
//     uint64_t min_int = (unsigned long long)UR(max_int+1);
//     // 创建一个随机数生成器
//     std::random_device rd;
//     std::mt19937 gen(rd());
  
//     // 定义随机数的分布范围
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);

//     // 生成随机数
//     ::llvm::APFloat min_fp = APFloat(::llvm::APFloat::IEEEsingle, dist(gen));
//     ::llvm::APFloat max_fp = APFloat(::llvm::APFloat::IEEEsingle, dist(gen));

//     auto res = builder.create<mlir::tosa::ClampOp>(loc, output, input,min_int, max_int, min_fp, max_fp);
//     region.pool.addRankedTensor(
//       TypeValue(output.dyn_cast<RankedTensorType>(), res), 
//       "tosa.clamp");
//     return res.getOperation();
//   };
// }

//  ::mlir::Type output, ::mlir::Value input1) {
// tosa.identity (mlir::tosa::IdentityOp)
OpGenerator tosaIdentityOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res = builder.create<mlir::tosa::IdentityOp>(loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}
// ::mlir::TypeRange output, ::mlir::Value cond, ::mlir::ValueRange inputs) {
// tosa.cond_if (mlir::tosa::IfOp) 

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.log (mlir::tosa::LogOp)
OpGenerator tosaLogOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res = builder.create<mlir::tosa::LogOp>(loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_and (mlir::tosa::LogicalAndOp) 
OpGenerator tosaLogicalAndOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!(ranked_tensor_filter(s, v))) return false;
        return v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI1Type();
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::LogicalAndOp>(loc, z, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_left_shift (mlir::tosa::LogicalLeftShiftOp)
OpGenerator tosaLogicalLeftShiftOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::LogicalLeftShiftOp>(loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}
// ::mlir::Type output, ::mlir::Value input1) {
// tosa.logical_not (mlir::tosa::LogicalNotOp)
OpGenerator tosaLogicalNotOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI1Type();    
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res = builder.create<mlir::tosa::LogicalNotOp>(loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_or (mlir::tosa::LogicalOrOp)
OpGenerator tosaLogicalOrOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI1Type();
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::LogicalOrOp>(loc, z, input1, input2);

    mb->add2Pool(res_tensor);
    return true;
  };
}
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_right_shift (mlir::tosa::LogicalRightShiftOp)
OpGenerator tosaLogicalRightShiftOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::LogicalRightShiftOp>(
      loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}
// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_xor (mlir::tosa::LogicalXorOp)
OpGenerator tosaLogicalXorOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI1Type();
      },
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::LogicalXorOp>(loc, z, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.matmul (mlir::tosa::MatMulOp) 
// Type outputType, Value a, Value b) {
OpGenerator tosaMatMulOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 3;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<3;i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    Type outputType = src1.getType();
    Value a = src1;
    Value b = src2;

    mlir::Value res = builder.create<mlir::tosa::MatMulOp>(loc, outputType, a, b);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.max_pool2d (mlir::tosa::MaxPool2dOp) 
// ::mlir::Type output, ::mlir::Value input, 
// ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr pad) {
OpGenerator tosaMaxPool2dOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<4;i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    SmallVector<int64_t> a, b, c;
    for(int i = 0;i<2;i++){
      a.push_back(1);
    }
    for(int i = 0;i<2;i++){
      b.push_back(1);
    }
    for(int i = 0;i<4;i++){
      c.push_back(1);
    }
    auto kernel = builder.getDenseI64ArrayAttr(a);
    auto stride = builder.getDenseI64ArrayAttr(b);
    auto pad = builder.getDenseI64ArrayAttr(c);

    mlir::Value res = builder.create<mlir::tosa::MaxPool2dOp>(
      loc, output, input, kernel, stride, pad);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.maximum (mlir::tosa::MaximumOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaMaximumOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::MaximumOp>(loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.minimum (mlir::tosa::MinimumOp)
OpGenerator tosaMinimumOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::MinimumOp>(loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.mul (mlir::tosa::MulOp) 
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr shift) {
OpGenerator tosaMulOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
    
    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto output = src1.getType();
    auto input1 = src1;
    auto input2 = src2;
    auto shift = builder.getIntegerAttr(
      builder.getI8Type(), ((long long)rand() << 8) | rand());

    mlir::Value res = builder.create<mlir::tosa::MulOp>(
      loc, output, input1, input2, shift);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.negate (mlir::tosa::NegateOp)
// Type outputType, Value input) {
OpGenerator tosaNegateOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    Type outputType = src1.getType();
    Value input = src1;

    mlir::Value res = builder.create<mlir::tosa::NegateOp>(loc, outputType, input);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.pad (mlir::tosa::PadOp)
// Type outputType, Value input, Value paddings) {
OpGenerator tosaPadOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> paddingsCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI64Type() ||
                  v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI32Type();
      }, 
      &paddingsCandidates
    );
    if (paddingsCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<rollIdx(5);i++){
        newShape.push_back(1);
      }
      paddingsCandidates.push_back(
        generateRankedTensor(builder, loc, RankedTensorType::get(newShape, builder.getI64Type()), mb));
    }
    auto src2 = paddingsCandidates[rollIdx(paddingsCandidates.size())];

    Type outputType = src1.getType();
    Value input = src1;
    Value paddings = src2;

    mlir::Value res = builder.create<mlir::tosa::PadOp>(loc, outputType, input, paddings);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.pow (mlir::tosa::PowOp)
// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaPowOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];
    
    auto input1 = src1;
    auto input2 = src2;
    auto z = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::PowOp>(loc, z, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// ::mlir::Type output_real, ::mlir::Type output_imag, ::mlir::Value input) {
// tosa.rfft2d (mlir::tosa::RFFT2dOp)
OpGenerator tosaRFFT2dOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 3;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<3;i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto output_real = src1.getType();
    auto output_imag = src2.getType();
    auto input = src2;
    
    auto res_tensor = builder.create<mlir::tosa::RFFT2dOp>(
      loc, output_real, output_imag, input);

    // mb->add2Pool(res_tensor);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.reciprocal (mlir::tosa::ReciprocalOp)
OpGenerator tosaReciprocalOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res = builder.create<mlir::tosa::ReciprocalOp>(loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}


//  ::mlir::Type output, ::mlir::Value input, uint64_t axis) 
// tosa.reduce_all (mlir::tosa::ReduceAllOp)
OpGenerator tosaReduceAllOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 &&
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;

      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for(int i = 0;i<rollIdx(3);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();


    uint64_t axis = rollIdx(src1.getType().cast<RankedTensorType>().getRank());
    mlir::Value res = builder.create<mlir::tosa::ReduceAllOp>(loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reduce_any (mlir::tosa::ReduceAnyOp)
// ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReduceAnyOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for(int i = 0;i<rollIdx(3);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
    
    auto input = src1;
    auto output = src1.getType();
    uint64_t axis = rollIdx(src1.getType().cast<RankedTensorType>().getRank());
    mlir::Value res = builder.create<mlir::tosa::ReduceAnyOp>(loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reduce_max (mlir::tosa::ReduceMaxOp)
//  ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReduceMaxOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for(int i = 0;i<rollIdx(3);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis = rollIdx(input.getType().dyn_cast<RankedTensorType>().getRank());
    mlir::Value res = builder.create<mlir::tosa::ReduceMaxOp>(loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reduce_min (mlir::tosa::ReduceMinOp)
// operand #0 must be unranked.tensor of number values or 1D/2D/3D/4D tensor of number values, but got 'tensor<f16>'
OpGenerator tosaReduceMinOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for(int i = 0;i<rollIdx(3);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis = rollIdx(input.getType().dyn_cast<RankedTensorType>().getRank());
    mlir::Value res = builder.create<mlir::tosa::ReduceMinOp>(loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}
// tosa.reduce_prod (mlir::tosa::ReduceProdOp)
OpGenerator tosaReduceProdOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for(int i = 0;i<rollIdx(4);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis = rollIdx(src1.getType().cast<RankedTensorType>().getRank());
    mlir::Value res = builder.create<mlir::tosa::ReduceProdOp>(loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}
// tosa.reduce_sum (mlir::tosa::ReduceSumOp) 
OpGenerator tosaReduceSumOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for(int i = 0;i<rollIdx(4);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();
    uint64_t axis = rollIdx(src1.getType().cast<RankedTensorType>().getRank());
    mlir::Value res = builder.create<mlir::tosa::ReduceSumOp>(loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.rescale (mlir::tosa::RescaleOp)
// ::mlir::Type output, ::mlir::Value input,
//  ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, :
//  ::mlir::DenseI32ArrayAttr multiplier, ::mlir::DenseI32ArrayAttr shift, 
//  ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr per_channel
OpGenerator tosaRescaleOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src1;
    auto output = src1.getType();

    auto input_zp = builder.getIntegerAttr(builder.getI32Type(), ((long long)rand() << 32) | rand());
    auto output_zp = builder.getIntegerAttr(builder.getI32Type(), ((long long)rand() << 32) | rand());

    SmallVector<int32_t> a, b;
    auto multiplier = builder.getDenseI32ArrayAttr(a);
    auto shift = builder.getDenseI32ArrayAttr(b);
    
    auto scale32 = builder.getBoolAttr(true);
    auto double_round = builder.getBoolAttr(true);
    auto per_channel = builder.getBoolAttr(true);

    mlir::Value res = builder.create<mlir::tosa::RescaleOp>(
      loc, 
      output, 
      input, 
      input_zp, 
      output_zp, 
      multiplier, 
      shift, 
      scale32, 
      double_round, 
      per_channel);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.reshape (mlir::tosa::ReshapeOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr new_shape) {
OpGenerator tosaReshapeOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
  
    auto input1 = src1;
    auto output = src1.getType();

    SmallVector<int64_t> a;
    auto new_shape = builder.getDenseI64ArrayAttr(a);

    mlir::Value res = builder.create<mlir::tosa::ReshapeOp>(loc, output, input1, new_shape);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.resize (mlir::tosa::ResizeOp)
// ::mlir::Type output, ::mlir::Value input, 
// ::mlir::DenseI64ArrayAttr scale, ::mlir::DenseI64ArrayAttr offset, ::mlir::DenseI64ArrayAttr border, 
// ::mlir::StringAttr mode) {
OpGenerator tosaResizeOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<4;i++){
        newShape.push_back(rollIdx(100)+10);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];
    auto input = src1;
    auto output = src1.getType();
    
    SmallVector<int64_t> a, b, c;
    for(int i = 0;i<4;i++){
      a.push_back(rollIdx(10));
    }
    for(int i = 0;i<2;i++){
      b.push_back(rollIdx(10));
    }
    for(int i = 0;i<2;i++){
      c.push_back(rollIdx(10));
    }
    auto scale = builder.getDenseI64ArrayAttr(a);
    auto offset = builder.getDenseI64ArrayAttr(b);
    auto border = builder.getDenseI64ArrayAttr(c);
    // 'tosa.resize' op attribute 'mode' failed to satisfy constraint: Supported resize/upsampling strategies
    
    // NearestNeighbor: 最近邻插值,简单快速但产生锯齿。
    // Linear: 双线性插值,较近邻产生更平滑结果但计算复杂度更高.
    // Area: 用平均值填充像素,保持像素总和不变但模糊原图。
    // Cubic: 三次样条插值,产生更光滑结果但计算开销最大。
    // Lanczos: 兰索斯窗口函数插值,在光滑度和计算效率上取得平衡。
    // MitchelNetravali: 一种二次Lanczos插值的变种算法。
    std::vector<std::string> modes = {
      "NearestNeighbor",  "Linear", "Area", "Cubic", "Lanczos", "MitchelNetravali"
    };

    auto mode = builder.getStringAttr(modes[rollIdx(modes.size())]);

    mlir::Value res = builder.create<mlir::tosa::ResizeOp>(
      loc, output, input, scale, offset, border, mode);
    mb->add2Pool(res);
    return true;
  };
}


// tosa.reverse (mlir::tosa::ReverseOp)
// ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReverseOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1); 
      for(int i = 0;i<rollIdx(3);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    uint64_t axis = rollIdx(src1.getType().dyn_cast<RankedTensorType>().getRank());
    mlir::Value res = builder.create<mlir::tosa::ReverseOp>(loc, output, input, axis);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.rsqrt (mlir::tosa::RsqrtOp)
OpGenerator tosaRsqrtOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input1 = src1;
    auto output = src1.getType();

    mlir::Value res = builder.create<mlir::tosa::RsqrtOp>(loc, output, input1);
    mb->add2Pool(res);
    return true;
  };
}


// tosa.scatter (mlir::tosa::ScatterOp)
// ::mlir::Type values_out, ::mlir::Value values_in, ::mlir::Value indices, ::mlir::Value input) {
OpGenerator tosaScatterOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> _3DCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 3;
      }, 
      &_3DCandidates
    );
    if (_3DCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<3;i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      _3DCandidates.push_back(dest);
    }
    auto src0 = _3DCandidates[rollIdx(_3DCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src0);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src0.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src1 = src2Candidates[rollIdx(src2Candidates.size())];

    std::vector<mlir::Value> _2D32bitCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 2 && 
          v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI32Type();
      }, 
      &_2D32bitCandidates
    );
    if (_2D32bitCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<2;i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, builder.getI32Type());
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      _2D32bitCandidates.push_back(dest);
    }
    auto src2 = _2D32bitCandidates[rollIdx(_2D32bitCandidates.size())];

    auto values_out = src0.getType();
    auto values_in = src0;
    auto indices = src2;
    auto input = src1;

    mlir::Value res = builder.create<mlir::tosa::ScatterOp>(
      loc, values_out, values_in, indices, input);
    mb->add2Pool(res);
    return true;
  };
}


// tosa.select (mlir::tosa::SelectOp)
// ::mlir::Type output, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value on_false) {
OpGenerator tosaSelectOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> predCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI1Type();
      }, 
      &predCandidates
    );
    if (predCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = generateRankedTensor(builder, loc, predTy, mb);
      predCandidates.push_back(dest);
    }
    auto src0 = predCandidates[rollIdx(predCandidates.size())];

    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto output = src1.getType();
    auto pred = src0;
    auto on_true = src1;
    auto on_false = src2;

    mlir::Value res_tensor = builder.create<mlir::tosa::SelectOp>(
      loc, output, pred, on_true, on_false);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.sigmoid (mlir::tosa::SigmoidOp) 
//  ::mlir::Type output, ::mlir::Value input) {
OpGenerator tosaSigmoidOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    mlir::Value res = builder.create<mlir::tosa::SigmoidOp>(loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.slice (mlir::tosa::SliceOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::DenseI64ArrayAttr start, ::mlir::DenseI64ArrayAttr size) {
OpGenerator tosaSliceOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 6;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
        operandCandidates.push_back(
          generateRankedTensor(
            builder, 
            loc, 
            randomRankedTensorType(builder.getContext()), 
            mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    SmallVector<int64_t> a, b;
    for (int i = 0 ; i < input.getType().cast<RankedTensorType>().getRank(); i++) {
      a.push_back(rollIdx(100));
      b.push_back(rollIdx(100));
    }
    auto start = builder.getDenseI64ArrayAttr(a);
    auto size = builder.getDenseI64ArrayAttr(b);

    mlir::Value res = builder.create<mlir::tosa::SliceOp>(
      loc, output, input, start, size);
    mb->add2Pool(res);
    return true;
  };
}


// tosa.sub (mlir::tosa::SubOp) 
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaSubOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> src2Candidates = std::vector<mlir::Value>();
    std::string src1TyStr = getValueTypeStr(src1);
    if (mb->valuePool.pool.count(src1TyStr)) {
      src2Candidates.insert(
        src2Candidates.begin(), 
        mb->valuePool.pool[src1TyStr].begin(), 
        mb->valuePool.pool[src1TyStr].end());
    }
    if (src2Candidates.empty()) {
      src2Candidates.push_back(
        generateRankedTensor(builder, loc, src1.getType().dyn_cast<RankedTensorType>(), mb));
    }
    auto src2 = src2Candidates[rollIdx(src2Candidates.size())];

    auto input1 = src1;
    auto input2 = src2;
    auto output = src1.getType();
    
    mlir::Value res_tensor = builder.create<mlir::tosa::SubOp>(
      loc, output, input1, input2);
    mb->add2Pool(res_tensor);
    return true;
  };
}

// tosa.table (mlir::tosa::TableOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::Value table) {
OpGenerator tosaTableOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for(int i = 0;i<rollIdx(3);i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      operandCandidates.push_back(dest);
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> tableCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 1;
      }, 
      &tableCandidates
    );
    if (tableCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      auto tableTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, tableTy, mb);
      tableCandidates.push_back(dest);
    }
    auto src2 = tableCandidates[rollIdx(tableCandidates.size())];

    auto output = src1.getType();
    auto input = src1;
    auto table = src2;

    mlir::Value res = builder.create<mlir::tosa::TableOp>(loc, output, input, table);
    mb->add2Pool(res);
    return true;
  };
}

// tosa.tanh (mlir::tosa::TanhOp)
OpGenerator tosaTanhOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(ranked_tensor_filter, &operandCandidates);
    if(operandCandidates.empty()){
      operandCandidates.push_back(
        generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    mlir::Value res = builder.create<mlir::tosa::TanhOp>(loc, output, input);
    mb->add2Pool(res);
    return true;
  };
}


// tosa.tile (mlir::tosa::TileOp)
OpGenerator tosaTileOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 4;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
        operandCandidates.push_back(
          generateRankedTensor(
            builder, 
            loc, 
            randomRankedTensorType(builder.getContext()), 
            mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    SmallVector<int64_t> a;
    for (int i = 0; i < input.getType().cast<RankedTensorType>().getRank(); i++) {
      a.push_back(1);
    }
    auto multiples = builder.getDenseI64ArrayAttr(a);

    mlir::Value res = builder.create<mlir::tosa::TileOp>(loc, output, input, multiples);
    mb->add2Pool(res);
    return true;
  };
}


// tosa.transpose_conv2d (mlir::tosa::TransposeConv2DOp) 
// ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, 
// ::mlir::DenseI64ArrayAttr out_pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr out_shape, 
// /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
OpGenerator tosaTransposeConv2DOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> inputCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 4;
      }, 
      &inputCandidates
    );
    if (inputCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<4;i++){
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      inputCandidates.push_back(dest);
    }
    auto src1 = inputCandidates[rollIdx(inputCandidates.size())];

    auto input = src1;
    auto output = src1.getType();

    std::vector<mlir::Value> biasCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 1;
      }, 
      &biasCandidates
    );
    if (biasCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      auto destTy = RankedTensorType::get(newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      biasCandidates.push_back(dest);
    }
    auto src2 = biasCandidates[rollIdx(biasCandidates.size())];
    auto bias = src2;

    SmallVector<Type> supported_types;
    // supported_types.push_back(builder.getI4Type());
    supported_types.push_back(builder.getI8Type());
    supported_types.push_back(builder.getF16Type());
    supported_types.push_back(builder.getF32Type());

    std::vector<mlir::Value> filterCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() == 4 && (
                // tval.type.dyn_cast<RankedTensorType>().getElementType() == builder.getI4Type() || 
                v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI8Type() ||
                v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getF32Type() ||
                v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getF16Type()
                );
      }, 
      &filterCandidates
    );
    if (filterCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for(int i = 0;i<4;i++){
        newShape.push_back(rollIdx(20));
      }
      auto destTy = RankedTensorType::get(newShape, supported_types[rollIdx(3)]);
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      filterCandidates.push_back(dest);
    }
    auto src3 = filterCandidates[rollIdx(filterCandidates.size())];
    auto filter = src3;

    SmallVector<int64_t> a, b, c;
    for(int i = 0;i<4;i++){
      a.push_back(1);
    }
    for(int i = 0;i<2;i++){
      b.push_back(1);
    }
    for(int i = 0;i<4;i++){
      c.push_back(1);
    }
    auto out_pad = builder.getDenseI64ArrayAttr(a);
    auto stride = builder.getDenseI64ArrayAttr(b);
    auto out_shape = builder.getDenseI64ArrayAttr(c);

    mlir::Value res = builder.create<mlir::tosa::TransposeConv2DOp>(
      loc, output, input, filter, bias, out_pad, stride, out_shape);
    mb->add2Pool(res);
    return true;
  };
}

// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value perms) {
// tosa.transpose (mlir::tosa::TransposeOp) 

OpGenerator tosaTransposeOpGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> operandCandidates = std::vector<mlir::Value>();
    mb->search(
      [] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getRank() > 0 && 
          v.getType().dyn_cast<RankedTensorType>().getRank() <= 6;
      }, 
      &operandCandidates
    );
    if (operandCandidates.empty()) {
        operandCandidates.push_back(
          generateRankedTensor(
            builder, 
            loc, 
            randomRankedTensorType(builder.getContext()), 
            mb));
    }
    auto src1 = operandCandidates[rollIdx(operandCandidates.size())];

    std::vector<mlir::Value> permsCandidates = std::vector<mlir::Value>();
    mb->search(
      [&] (std::string s, mlir::Value v) -> bool {
        if (!ranked_tensor_filter(s, v)) return false;
        return v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI32Type()||
                v.getType().dyn_cast<RankedTensorType>().getElementType() == builder.getI64Type();
      }, 
      &permsCandidates
    );
    if (permsCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto destTy = RankedTensorType::get(newShape, builder.getI64Type());
      if (rollIdx(2) == 1) {
        auto destTy = RankedTensorType::get(newShape, builder.getI32Type());
      }
      auto dest = generateRankedTensor(builder, loc, destTy, mb);
      permsCandidates.push_back(dest);
    }
    auto src2 = permsCandidates[rollIdx(permsCandidates.size())];

    auto output = src1.getType();
    auto input1 = src1;
    auto perms = src2;

    mlir::Value res_tensor = builder.create<mlir::tosa::TransposeOp>(
      loc, output, input1, perms);
    mb->add2Pool(res_tensor);
    return true;
  };
}
