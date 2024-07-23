//
// Created by Stan Wang on 2022/10/28.
//
#include "smith/Util.h"
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"

/*---------------------arith operations-------------------*/

std::vector<arith::CmpFPredicate> cmpFPredicates = {
    arith::CmpFPredicate::AlwaysFalse, arith::CmpFPredicate::AlwaysTrue,
    arith::CmpFPredicate::OEQ,         arith::CmpFPredicate::OGE,
    arith::CmpFPredicate::OGT,         arith::CmpFPredicate::OLE,
    arith::CmpFPredicate::OLT,         arith::CmpFPredicate::ONE,
    arith::CmpFPredicate::ORD,         arith::CmpFPredicate::UEQ,
    arith::CmpFPredicate::UGT,         arith::CmpFPredicate::UGE,
    arith::CmpFPredicate::ULT,         arith::CmpFPredicate::ULE,
    arith::CmpFPredicate::UNE,         arith::CmpFPredicate::UNO};

std::vector<arith::CmpIPredicate> cmpIPredicates = {
    arith::CmpIPredicate::eq,  arith::CmpIPredicate::ne,
    arith::CmpIPredicate::slt, arith::CmpIPredicate::sle,
    arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge,
    arith::CmpIPredicate::ult, arith::CmpIPredicate::ule,
    arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge};

OpGenerator addFGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.addF");
    auto value =
        builder.create<arith::AddFOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.addF");
    return value.getOperation();
  };
}

OpGenerator addIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;

    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());

    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.addI");
    auto value =
        builder.create<arith::AddIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.addI");
    return value.getOperation();
  };
}

OpGenerator andIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.andI");
    auto value =
        builder.create<arith::AndIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.andI");
    return value.getOperation();
  };
}

OpGenerator ceilDivSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.ceilDivSI");
    auto value = builder.create<arith::CeilDivSIOp>(loc, operands[0].val,
                                                    operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.ceilDivSI");
    return value.getOperation();
  };
}

OpGenerator ceilDivUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.ceilDivUI");
    auto value = builder.create<arith::CeilDivUIOp>(loc, operands[0].val,
                                                    operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.ceilDivUI");
    return value.getOperation();
  };
}

OpGenerator cmpFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.cmpF");
    auto predicate = cmpFPredicates[UR(cmpFPredicates.size())];
    auto value = builder.create<arith::CmpFOp>(loc, predicate, operands[0].val,
                                               operands[1].val);
    if (operands[0].type.dyn_cast<FloatType>()) {
      auto tVal = TypeValue(builder.getI1Type(), value);
      typedValuePool.addIntOrFloat(tVal, "");
    } else if (operands[0].type.dyn_cast<RankedTensorType>()) {
      auto tVal =
          TypeValue(RankedTensorType::get(
                        operands[0].type.dyn_cast<ShapedType>().getShape(),
                        builder.getI1Type()),
                    value);
      typedValuePool.addTypeValue(tVal);
    }
    return value.getOperation();
  };
}

OpGenerator cmpIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.cmpI");
    auto predicate = cmpIPredicates[UR(cmpIPredicates.size())];
    auto value = builder.create<arith::CmpIOp>(loc, predicate, operands[0].val,
                                               operands[1].val);
    if (operands[0].type.dyn_cast<IntegerType>()) {
      auto tVal = TypeValue(builder.getI1Type(), value);
      typedValuePool.addIntOrFloat(tVal,"");
    } else if (operands[0].type.dyn_cast<RankedTensorType>()) {
      auto tVal =
          TypeValue(RankedTensorType::get(
                        operands[0].type.dyn_cast<ShapedType>().getShape(),
                        builder.getI1Type()),
                    value);
      typedValuePool.addTypeValue(tVal);
    }
    return value.getOperation();
  };
}

OpGenerator constantGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    std::vector<mlir::Type> supportedElementaryTypes =
        getSupportedIntOrFloatTypes(builder.getContext());
    auto elementType =
        supportedElementaryTypes[UR(supportedElementaryTypes.size())];
    IntegerType iType = elementType.dyn_cast<IntegerType>();
    assert(elementType.isIntOrFloat());
    if (iType) {
      // TODO- support value width greater than 32.
      unsigned width = min(iType.getWidth(), 32);
      long long val = UR(((long long)1) << width);
      Value cons = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(elementType, val));
      auto tVal = TypeValue(iType, cons);
      typedValuePool.addIntOrFloat(tVal, "arith.constant");
      return cons.getDefiningOp();
    } else {
      FloatType fType = elementType.dyn_cast<FloatType>();
      unsigned width = min(fType.getWidth(), 32);
      long long valf = UR(((long long)1) << width);
      double val = static_cast<double>(valf);
      Value cons = builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(elementType, val));
      auto tVal = TypeValue(fType, cons);
      typedValuePool.addIntOrFloat(tVal, "arith.constant");
      return cons.getDefiningOp();
    }
  };
}

OpGenerator divFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.divF");
    auto value =
        builder.create<arith::DivFOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.divF");
    return value.getOperation();
  };
}

OpGenerator divSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.divSI");
    auto value =
        builder.create<arith::DivSIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.divSI");
    return value.getOperation();
  };
}

OpGenerator divUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.divUI");
    auto value =
        builder.create<arith::DivUIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.divUI");
    return value.getOperation();
  };
}

OpGenerator floorDivSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.floorDivSI");
    auto value = builder.create<arith::FloorDivSIOp>(loc, operands[0].val,
                                                     operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.floorDivSI");
    return value.getOperation();
  };
}

OpGenerator minSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.minSI");
    auto value =
        builder.create<arith::MinSIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.minSI");
    return value.getOperation();
  };
}

OpGenerator minUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.minUI");
    auto value =
        builder.create<arith::MinUIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.minUI");
    return value.getOperation();
  };
}

OpGenerator mulFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.mulF");
    auto value =
        builder.create<arith::MulFOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.mulF");
    return value.getOperation();
  };
}

OpGenerator mulIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.mulI");
    auto value =
        builder.create<arith::MulIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.mulI");
    return value.getOperation();
  };
}

OpGenerator negFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operand0 =
        searchFloatUnaryOperandFrom(builder, loc, candidates, "arith.negF");
    auto value = builder.create<arith::NegFOp>(loc, operand0.val);
    auto tVal = TypeValue(operand0.type, value);
    typedValuePool.addTypeValue(tVal, "arith.negF");
    return value.getOperation();
  };
}

OpGenerator orIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.orI");
    auto value =
        builder.create<arith::OrIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.orI");
    return value.getOperation();
  };
}

OpGenerator remFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.remF");
    auto value =
        builder.create<arith::RemFOp>(loc, operands[0].val, operands[1].val);

    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.remF");
    return value.getOperation();
  };
}

OpGenerator remSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.remSI");
    auto value =
        builder.create<arith::RemSIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.remSI");
    return value.getOperation();
  };
}

OpGenerator remUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.remUI");
    auto value =
        builder.create<arith::RemUIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.remUI");
    return value.getOperation();
  };
}

OpGenerator shlIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.shlI");
    auto value =
        builder.create<arith::ShLIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.shlI");
    return value.getOperation();
  };
}

OpGenerator shrSIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.shrSI");
    auto value =
        builder.create<arith::ShRSIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.shrSI");
    return value.getOperation();
  };
}

OpGenerator shrUIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    auto operands = searchIntegerBinaryOperandsFrom(builder, loc, candidates,
                                                    "arith.shrUI");
    auto value =
        builder.create<arith::ShRUIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.shrUI");
    return value.getOperation();
  };
}

OpGenerator subFGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getF32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.subF");
    auto value =
        builder.create<arith::RemFOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.subF");
    return value.getOperation();
  };
}

OpGenerator subIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.subI");
    auto value =
        builder.create<arith::SubIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.subI");
    return value.getOperation();
  };
}

OpGenerator xorIGenerator() {

  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto typedValuePool = region.pool;
    auto candidates = typedValuePool.getCandidatesFromIntOrFloats(
        builder, loc, builder.getI32Type());
    //    candidates.insert(candidates.end(), typedValuePool.tensorPool.begin(),
    //                      typedValuePool.tensorPool.end());
    auto operands =
        searchIntegerBinaryOperandsFrom(builder, loc, candidates, "arith.xorI");
    auto value =
        builder.create<arith::XOrIOp>(loc, operands[0].val, operands[1].val);
    auto tVal = TypeValue(operands[0].type, value);
    typedValuePool.addTypeValue(tVal, "arith.xorI");
    return value.getOperation();
  };
}