
#include "NodeGenerator.h"

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
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // auto typedValuePool = region.pool;
    // auto candidates = typedValuePool.getCandidatesFromIntOrFloats(builder, loc, builder.getF32Type());  // 取 INT FLOAT 或者 INDEX
    // auto operands = searchFloatBinaryOperandsFrom(builder, loc, candidates, "arith.addF");  // 取两个相同的operand

    // Get first operand 
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");

    mlir::Value value = builder.create<arith::AddFOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator addIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    mlir::Value value = builder.create<arith::AddIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator andIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create the Operation
    mlir::Value value = builder.create<arith::AndIOp>(loc, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator ceilDivSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Generate operation
    mlir::Value value = builder.create<arith::CeilDivSIOp>(loc, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator ceilDivUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Generate Operation
    mlir::Value value = builder.create<arith::CeilDivUIOp>(loc, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator cmpFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Select Predicate
    arith::CmpFPredicate predicate = cmpFPredicates[rollIdx(cmpFPredicates.size())];
    // Generate peration
    mlir::Value value = builder.create<arith::CmpFOp>(loc, predicate, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator cmpIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Select Predicate
    arith::CmpIPredicate predicate = cmpIPredicates[rollIdx(cmpIPredicates.size())];
    // create operation
    mlir::Value value = builder.create<arith::CmpIOp>(loc, predicate, first, second);

    mb->add2Pool(value);
    return true;
  };
}

OpGenerator constantGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Type> supportedElementaryTypes = getSupportedIntOrFloatTypes(builder.getContext());
    auto elementType = supportedElementaryTypes[rollIdx(supportedElementaryTypes.size())];
    IntegerType iType = elementType.dyn_cast<IntegerType>();
    assert(elementType.isIntOrFloat());
    // Generation Logic
    if (iType) {
      unsigned width = min(iType.getWidth(), 32);
      long long val = rollIdx(((long long)1) << width);
      mlir::Value cons = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(elementType, val));
      mb->add2Pool(cons);
    } else {
      FloatType fType = elementType.dyn_cast<FloatType>();
      unsigned width = min(fType.getWidth(), 32);
      long long valf = rollIdx(((long long)1) << width);
      double val = static_cast<double>(valf);
      mlir::Value cons = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(elementType, val));
      mb->add2Pool(cons);
    }
    return true;
  };
}

OpGenerator divFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create the Operation
    mlir::Value value = builder.create<arith::DivFOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator divSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::DivSIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator divUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation    
    mlir::Value value = builder.create<arith::DivUIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator floorDivSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::FloorDivSIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

// OpGenerator maxFGenerator() {
//   return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
//     // Get first operand
//     std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
//     mb->search(float_filter, &candidates);
//     if (!candidates.size())
//       return false;
//     mlir::Value first = candidates[rollIdx(candidates.size())];
//     // Get second operand
//     std::string ty = getValueTypeStr(first);
//     if (!mb->valuePool.pool.count(ty))
//       return false;
//     mlir::Value second = mb->search(ty, "Unknown");
//     // Create Operation
//     auto value = builder.create<arith::MaxFOp>(loc, first, second);
//     return true;
//   };
// }

OpGenerator maxSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation    
    mlir::Value value = builder.create<arith::MaxSIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator maxUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::MaxUIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

// OpGenerator minFGenerator() {
//   return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
//     // Get first operand
//     std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
//     mb->search(float_filter, &candidates);
//     if (!candidates.size())
//       return false;
//     mlir::Value first = candidates[rollIdx(candidates.size())];
//     // Get second operand
//     std::string ty = getValueTypeStr(first);
//     if (!mb->valuePool.pool.count(ty))
//       return false;
//     mlir::Value second = mb->search(ty, "Unknown");
//     // Create Opeartion
//     auto value = builder.create<arith::MinFOp>(loc, first, second);
//     return true;
//   };
// }

OpGenerator minSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::MinSIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator minUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation    
    mlir::Value value = builder.create<arith::MinUIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator mulFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::MulFOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator mulIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion
    mlir::Value value = builder.create<arith::MulIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator negFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Create Operation
    mlir::Value value = builder.create<arith::NegFOp>(loc, first);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator orIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::OrIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator remFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::RemFOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator remSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::RemSIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator remUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion    
    mlir::Value value = builder.create<arith::RemUIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator shlIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion    
    mlir::Value value = builder.create<arith::ShLIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator shrSIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Opeartion 
    mlir::Value value = builder.create<arith::ShRSIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator shrUIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::ShRUIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator subFGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(float_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::RemFOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator subIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::SubIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}

OpGenerator xorIGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    // Get first operand
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
    mb->search(int_filter, &candidates);
    if (!candidates.size())
      return false;
    mlir::Value first = candidates[rollIdx(candidates.size())];
    // Get second operand
    std::string ty = getValueTypeStr(first);
    if (!mb->valuePool.pool.count(ty))
      return false;
    mlir::Value second = mb->search(ty, "Unknown");
    // Create Operation
    mlir::Value value = builder.create<arith::XOrIOp>(loc, first, second);
    mb->add2Pool(value);
    return true;
  };
}