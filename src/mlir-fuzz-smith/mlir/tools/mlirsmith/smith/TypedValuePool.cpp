//
// Created by Stan Wang on 2022/12/8.
//

#include "../include/smith/TypedValuePool.h"

void TypedValuePool::addTypeValue(const TypeValue &typedValue, std::string op,
                                  PoolType ty) {
  assert(typedValue.val);
  switch (ty) {
  case PoolType::Index: {
    assert(typedValue.type.isa<IndexType>());
  }
  case PoolType::IntOrFloat: {
    assert(typedValue.type.isIntOrFloat());
  }
  case PoolType::Vector: {
    assert(typedValue.type.isa<VectorType>());
  }
  case PoolType::StaticShapedMemref: {
    assert(typedValue.type.isa<MemRefType>() &&
           typedValue.type.dyn_cast<MemRefType>().hasStaticShape());
  }
  case PoolType::DynamicShapedMemref: {
    assert(typedValue.type.isa<MemRefType>() &&
           typedValue.type.dyn_cast<MemRefType>().hasRank());
  }
  case PoolType::StaticShapedTensor: {
    assert(typedValue.type.isa<TensorType>() &&
           typedValue.type.dyn_cast<TensorType>().hasStaticShape());
  }
  case PoolType::DynamicShapedTensor: {
    assert(typedValue.type.isa<TensorType>() &&
           typedValue.type.dyn_cast<TensorType>().hasRank());
  }
  case PoolType::Unknown: {
    if (typedValue.type.dyn_cast<RankedTensorType>()) {
      auto t = typedValue.type.dyn_cast<RankedTensorType>();
      if (t.hasStaticShape()) {
        addStaticShapedTensor(typedValue, op);
      } else {
        addDynamicShapedTensor(typedValue, op);
      }
    } else if (typedValue.type.dyn_cast<MemRefType>()) {
      if (typedValue.type.dyn_cast<MemRefType>().hasStaticShape()) {
        addStaticShapedMemref(typedValue, op);
      } else {
        addDynamicShapedMemref(typedValue, op);
      }
    } else if (typedValue.type.isIntOrFloat()) {
      addIntOrFloat(typedValue, op);
    } else if (typedValue.type.dyn_cast<IndexType>()) {
      addIndex(typedValue, op);
    }
    return;
  }
  }
  typeValues[ty].push_back(typedValue);
}

void TypedValuePool::addIntOrFloat(const TypeValue &element,
                                   std::string op = "") {
  assert(element.type.isIntOrFloat());
  // check type
  intOrFloatPool.push_back(element);
}

void TypedValuePool::addElement(const TypeValue &element, std::string op = "") {
  assert(element.val);
  bool typeCheck = (element.type.dyn_cast<IntegerType>() ||
                    element.type.dyn_cast<FloatType>() ||
                    element.type.dyn_cast<IndexType>());

  assert(typeCheck && "addElement() require Integer or Float or Index");
  if (element.type.isIntOrFloat()) {
    intOrFloatPool.push_back(element);
  } else {
    indexPool.push_back(element);
  }
}

void TypedValuePool::addRankedMemref(const TypeValue &memref,
                                     std::string op = "") {
  auto memTy = memref.val.getType().dyn_cast<MemRefType>();
  assert(memTy && memTy.hasRank());
  if (memTy.hasStaticShape()) {
    addStaticShapedMemref(memref, op);
  } else {
    addDynamicShapedMemref(memref, op);
  }
}

void TypedValuePool::addStaticShapedMemref(const TypeValue &memref,
                                           std::string op = "") {
  assert(memref.val.getType().isa<MemRefType>() &&
         memref.val.getType().dyn_cast<MemRefType>().hasStaticShape());
  staticShapedMemrefPool.push_back(memref);
}

void TypedValuePool::addDynamicShapedMemref(const TypeValue &memref,
                                            std::string op = "") {
  auto memrefTy = memref.type.dyn_cast<MemRefType>();
  assert(memrefTy.hasRank());
  if (memrefTy.hasStaticShape()) {
    addStaticShapedMemref(memref, op);
  }
  dynamicShapedMemrefPool.push_back(memref);
}

void TypedValuePool::addRankedTensor(const TypeValue &tensor,
                                     std::string op = "") {
  assert(tensor.val);
  auto t = tensor.type.dyn_cast<RankedTensorType>();
  if (t.hasStaticShape()) {
    addStaticShapedTensor(tensor, op);
  } else {
    addDynamicShapedTensor(tensor, op);
  }
}

void TypedValuePool::addStaticShapedTensor(const TypeValue &tensor,
                                           std::string op = "") {
  assert(tensor.val);
  auto t = tensor.type.dyn_cast<RankedTensorType>();
  if (t && t.hasStaticShape()) {
    staticShapedTensorPool.push_back(tensor);
  } else if (conf.debug) {
    std::cout << "MisType inserted in static shaped tensors, op: " << op
              << std::endl;
  }
}

void TypedValuePool::addDynamicShapedTensor(const TypeValue &dynamicTensor,
                                            std::string op = "") {
  assert(dynamicTensor.val);
  assert(dynamicTensor.type.dyn_cast<RankedTensorType>() &&
         "Require RankedTensorType here.");
  dynamicShapedTensorPool.push_back(dynamicTensor);
}

void TypedValuePool::addVector(const TypeValue &vector, std::string op = "") {
  assert(vector.val);
  auto ty = vector.type.dyn_cast<VectorType>();
  if (ty) {
    vectorPool.push_back(vector);
  } else if (conf.debug) {
    std::cout << "MisType inserted in tensors, op: " << op << std::endl;
  }
}

void TypedValuePool::addIndex(const TypeValue &index, std::string op = "") {
  assert(index.val);
  if (index.type.dyn_cast<IndexType>()) {
    indexPool.push_back(index);
  } else if (conf.debug) {
    std::cout << "MisType inserted in tensors, op: " << op << std::endl;
  }
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromIntOrFloats(OpBuilder &builder, Location loc,
                                             Type t) {

  if (intOrFloatPool.empty()) {
    generateElement(builder, loc, t);
  }
  return intOrFloatPool;
}

std::vector<TypeValue> TypedValuePool::getCandidatesFromStaticShapedMemref(
    OpBuilder &builder, Location loc, MemRefType t) {
  assert(t.hasStaticShape());
  if (staticShapedMemrefPool.empty()) {
    generateStaticShapedMemref(builder, loc, t);
  }
  return staticShapedMemrefPool;
}

std::vector<TypeValue> TypedValuePool::getCandidatesFromDynamicShapedMemref(
    OpBuilder &builder, Location loc, MemRefType t) {
  if (dynamicShapedMemrefPool.empty()) {
    generateDynamicShapedMemref(builder, loc, t);
  }
  return dynamicShapedMemrefPool;
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromVector(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Type t) {
  if (vectorPool.empty()) {
    generateVector(builder, loc, t.dyn_cast<VectorType>());
  }
  return vectorPool;
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromStaticShapedTensor(OpBuilder &builder,
                                                    Location loc, Type t) {
  if (staticShapedTensorPool.empty()) {
    generateStaticShapedTensor(builder, loc, t.dyn_cast<RankedTensorType>());
  }
  return staticShapedTensorPool;
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromIndex(OpBuilder &builder, Location loc) {
  if (indexPool.empty()) {
    indexPool.push_back(
        TypeValue(builder.getIndexType(),
                  builder.create<arith::ConstantIndexOp>(loc, 1)));
  }
  return indexPool;
}

std::vector<TypeValue>
TypedValuePool::searchCandidatesFrom(std::vector<PoolType> poolTypes,
                                     TypeValueFilter filter) {
  std::vector<TypeValue> candidates;
  for (auto type : poolTypes) {
    std::vector<TypeValue> temp;
    switch (type) {
    case StaticShapedTensor: {
      temp = staticShapedTensorPool;
      break;
    }
    case DynamicShapedTensor: {
      temp = dynamicShapedTensorPool;
      break;
    }
    case StaticShapedMemref: {
      temp = staticShapedMemrefPool;
      break;
    }
    case DynamicShapedMemref: {
      temp = dynamicShapedMemrefPool;
      break;
    }
    case IntOrFloat: {
      temp = intOrFloatPool;
      break;
    }
    case Index: {
      temp = indexPool;
      break;
    }
    case Vector: {
      temp = vectorPool;
      break;
    }
    }
    for (auto tval : temp) {
      if (filter(tval)) {
        candidates.push_back(tval);
      }
    }
  }
  return candidates;
}

TypeValue TypedValuePool::generateInteger(OpBuilder &builder, Location loc,
                                          IntegerType type) {
  auto res = builder.create<arith::ConstantIntOp>(loc, UR(2), type);
  auto tVal = TypeValue(type, res);
  addIntOrFloat(tVal);
  return tVal;
}

TypeValue TypedValuePool::generateFloat(OpBuilder &builder, Location loc,
                                        FloatType type) {

  auto one = APFloat(type.getFloatSemantics(), 1);
  auto res = builder.create<arith::ConstantFloatOp>(loc, one, type);
  auto tVal = TypeValue(type, res);
  addIntOrFloat(tVal);
  return tVal;
}

TypeValue TypedValuePool::generateIndex(mlir::OpBuilder &builder,
                                        mlir::Location loc) {
  auto res = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto tVal = TypeValue(builder.getIndexType(), res);
  addIndex(tVal, "");
  return tVal;
}

Value TypedValuePool::getConstantIndex(OpBuilder &builder, Location loc,
                                       int num) {
  if (constantIndices.find(num) != constantIndices.end()) {
    assert(llvm::detail::isPresent(constantIndices[num]));
    return constantIndices[num];
  }
  auto res = builder.create<arith::ConstantIndexOp>(loc, num);
  constantIndices.insert(std::make_pair(num, res));
  return res;
}

TypeValue TypedValuePool::generateElement(OpBuilder &builder, Location loc,
                                          Type type) {
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

TypeValue TypedValuePool::generateRankedTensor(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::RankedTensorType rtTy) {
  assert(rtTy.hasRank());
  if (rtTy.hasStaticShape()) {
    return generateStaticShapedTensor(builder, loc, rtTy);
  } else {
    return generateDynamicShapedTensor(builder, loc, rtTy);
  }
}

TypeValue TypedValuePool::generateStaticShapedTensor(OpBuilder &builder,
                                                     Location loc,
                                                     RankedTensorType type) {
  assert(type.hasStaticShape());
  auto shapedType = type.dyn_cast<ShapedType>();
  auto t = builder.create<tensor::EmptyOp>(loc, shapedType.getShape(),
                                           shapedType.getElementType());
  auto tVal = TypeValue(type, t);
  addStaticShapedTensor(tVal, "");
  return tVal;
}

// generate an empty d tensor
TypeValue TypedValuePool::generateDynamicShapedTensor(OpBuilder &builder,
                                                      Location loc,
                                                      RankedTensorType type) {
  auto dynDimNum = type.getNumDynamicDims();
  SmallVector<Value> dynDims;
  for (int i = 0; i < dynDimNum; ++i) {
    dynDims.push_back(sampleTypedValueFrom(getCandidatesFromIndex(builder, loc),
                                           "tensor.empty")
                          .val);
  }
  auto t = builder.create<tensor::EmptyOp>(loc, type, ValueRange(dynDims));
  auto tVal = TypeValue(type, t);
  addDynamicShapedTensor(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateRankedMemref(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::MemRefType type) {
  if (type.hasStaticShape()) {
    return generateStaticShapedMemref(builder, loc, type);
  } else {
    return generateDynamicShapedMemref(builder, loc, type);
  }
}

TypeValue TypedValuePool::generateStaticShapedMemref(OpBuilder &builder,
                                                     Location loc,
                                                     MemRefType type) {
  auto memref = builder.create<memref::AllocOp>(loc, type);
  auto tVal = TypeValue(type, memref);
  addStaticShapedMemref(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateDynamicShapedMemref(OpBuilder &builder,
                                                      Location loc,
                                                      MemRefType type) {
  auto dyNum = type.getNumDynamicDims();
  SmallVector<Value> dyDims;
  auto idxCandidates = getCandidatesFromIndex(builder, loc);
  for (int i = 0; i < dyNum; ++i) {
    dyDims.push_back(sampleTypedValueFrom(idxCandidates, "memref.alloc").val);
  }
  auto memref = builder.create<memref::AllocOp>(loc, type, dyDims);
  auto tval = TypeValue(memref.getType(), memref);
  addRankedMemref(tval);
  return tval;
}

TypeValue TypedValuePool::generateVector(OpBuilder &builder, Location loc,
                                         VectorType type) {
  auto shapeTy = type.dyn_cast<ShapedType>();
  auto elemTy = shapeTy.getElementType();
  auto candidates =
      searchCandidatesFrom({PoolType::Index, PoolType::IntOrFloat},
                           [&](TypeValue tVal) { return elemTy == tVal.type; });
  if (candidates.empty()) {
    if (elemTy.dyn_cast<IndexType>()) {
      candidates.push_back(generateIndex(builder, loc));
    } else {
      candidates.push_back(generateElement(builder, loc, elemTy));
    }
  }
  auto elem = sampleTypedValueFrom(candidates, "vector.broadcast");
  auto vec = builder.create<vector::BroadcastOp>(loc, type, elem.val);
  auto tVal = TypeValue(type, vec);
  addVector(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateTypedValue(OpBuilder &builder, Location loc,
                                             Type type) {
  if (type.isa<IntegerType>()) {
    return generateInteger(builder, loc, type.dyn_cast<IntegerType>());
  } else if (type.isa<FloatType>()) {
    return generateFloat(builder, loc, type.dyn_cast<FloatType>());
  } else if (type.isa<MemRefType>()) {
    auto memTy = type.dyn_cast<MemRefType>();
    if (memTy.hasStaticShape()) {
      return generateStaticShapedMemref(builder, loc, memTy);
    } else {
      return generateDynamicShapedMemref(builder, loc, memTy);
    }
  } else if (type.isa<RankedTensorType>()) {
    auto rtTy = type.dyn_cast<RankedTensorType>();
    if (rtTy.hasStaticShape()) {
      return generateStaticShapedTensor(builder, loc, rtTy);
    } else {
      return generateDynamicShapedTensor(builder, loc, rtTy);
    }
  } else if (type.isa<VectorType>()) {
    return generateVector(builder, loc, type.dyn_cast<VectorType>());
  } else if (type.isa<IndexType>()) {
    return generateIndex(builder, loc);
  } else {
    std::cout << "unsupported type for new value generation\n";
    exit(-1);
  }
}

// TODO deprecate this
std::vector<TypeValue> searchTypedValueFrom(const TypedValuePool pool,
                                            Type type) {
  std::vector<TypeValue> candidates;
  if (type.isa<RankedTensorType>()) {
    candidates = searchShapedInputFrom(type.dyn_cast<ShapedType>(),
                                       pool.staticShapedTensorPool);
  } else if (type.isa<MemRefType>()) {
    candidates = searchShapedInputFrom(type.dyn_cast<ShapedType>(),
                                       pool.staticShapedMemrefPool);
  } else if (type.isa<FloatType>() || type.isa<IntegerType>()) {
    candidates = searchTypedValueFrom(pool.intOrFloatPool, type);
  } else if (type.isa<VectorType>()) {
    candidates = searchTypedValueFrom(pool.vectorPool, type);
  } else if (type.isa<IndexType>()) {
    candidates = searchTypedValueFrom(pool.indexPool, type);
  } else {
    llvm::errs() << "unsupported type" << type << "\n";
    exit(-1);
  }
  return candidates;
}

SmallVector<Value>
TypedValuePool::randomIndicesForShapedType(ShapedType shapedType,
                                           OpBuilder &builder, Location loc) {

  SmallVector<Value, 8> indices;
  for (size_t dim = 0; dim < shapedType.getRank(); ++dim) {
    if (shapedType.isDynamicDim(dim)) {
      indices.push_back(constantIndices.begin()->second);
    } else {
      auto idx = UR(shapedType.getDimSize(dim));
      auto idxVal = getConstantIndex(builder, loc, idx);
      indices.push_back(idxVal);
    }
  }
  return indices;
}