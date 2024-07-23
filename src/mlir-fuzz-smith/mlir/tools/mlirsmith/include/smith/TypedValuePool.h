//
// Created by Stan Wang on 2022/12/8.
//

#ifndef MLIR_FUZZ_TYPEDVALUEPOOL_H
#define MLIR_FUZZ_TYPEDVALUEPOOL_H

#include "Util.h"

enum PoolType {
  StaticShapedTensor,
  DynamicShapedTensor,
  StaticShapedMemref,
  DynamicShapedMemref,
  IntOrFloat,
  Vector,
  Index,
  SPIRVArray,
  SPIRVImage,
  SPIRVPointer,
  SPIRVRtArray,
  SPIRVSampledImage,
  SPIRVStruct,
  Unknown
};

inline std::vector<PoolType> allTypes = {PoolType::StaticShapedTensor,
                                         PoolType::DynamicShapedTensor,
                                         PoolType::StaticShapedMemref,
                                         PoolType::DynamicShapedMemref,
                                         PoolType::IntOrFloat,
                                         PoolType::Vector,
                                         PoolType::Index,
                                         PoolType::SPIRVArray,
                                         PoolType::SPIRVImage,
                                         PoolType::SPIRVPointer,
                                         PoolType::SPIRVRtArray,
                                         PoolType::SPIRVSampledImage,
                                         PoolType::SPIRVStruct};

using InitValueGenerator = std::function<TypeValue(OpBuilder &, Location)>;

struct TypedValuePool {
  std::map<PoolType, std::vector<TypeValue>> typeValues;
  std::vector<TypeValue> staticShapedTensorPool;
  std::vector<TypeValue> dynamicShapedTensorPool;
  std::vector<TypeValue> staticShapedMemrefPool;
  std::vector<TypeValue> dynamicShapedMemrefPool;
  std::vector<TypeValue> unrankedTensorPool;
  std::vector<TypeValue> unrankedMemrefPool;
  std::vector<TypeValue> intOrFloatPool;
  std::vector<TypeValue> vectorPool;
  std::vector<TypeValue> indexPool;
  std::map<int, Value> constantIndices;
  // for linalg.generic
  std::vector<SmallVector<AffineExpr>> affineExprPool;
  std::vector<AffineMap> affineMapPool;
  std::vector<IntegerSet> integerSetPool;

  TypedValuePool(
      std::vector<TypeValue> staticShapedTensorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> dynamicShapedTensorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> staticShapedMemrefPool = std::vector<TypeValue>(),
      std::vector<TypeValue> dynamicShapedMemrefPool = std::vector<TypeValue>(),
      std::vector<TypeValue> unrankedMemrefPool = std::vector<TypeValue>(),
      std::vector<TypeValue> unrankedTensorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> intOrFloatPool = std::vector<TypeValue>(),
      std::vector<TypeValue> vectorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> indexPool = std::vector<TypeValue>(),
      std::map<int, Value> constantIndices = std::map<int, Value>(),
      std::vector<SmallVector<AffineExpr>> affineExprPool =
          std::vector<SmallVector<AffineExpr>>(),
      std::vector<AffineMap> affineMapPool = std::vector<AffineMap>(),
      std::vector<IntegerSet> integerSetPool = std::vector<IntegerSet>())
      : staticShapedTensorPool(staticShapedTensorPool),
        dynamicShapedTensorPool(dynamicShapedTensorPool),
        staticShapedMemrefPool(staticShapedMemrefPool),
        dynamicShapedMemrefPool(dynamicShapedMemrefPool),
        unrankedTensorPool(unrankedTensorPool),
        unrankedMemrefPool(unrankedMemrefPool), intOrFloatPool(intOrFloatPool),
        vectorPool(vectorPool), indexPool(indexPool),
        constantIndices(constantIndices), affineExprPool(affineExprPool),
        affineMapPool(affineMapPool), integerSetPool(integerSetPool) {
    for (auto type : allTypes) {
      typeValues.insert(std::make_pair(type, std::vector<TypeValue>()));
    }
    typeValues.insert(std::make_pair(PoolType::IntOrFloat, intOrFloatPool));
    typeValues.insert(std::make_pair(PoolType::Vector, vectorPool));
    typeValues.insert(std::make_pair(PoolType::Index, indexPool));
    typeValues.insert(
        std::make_pair(PoolType::DynamicShapedTensor, dynamicShapedTensorPool));
    typeValues.insert(
        std::make_pair(PoolType::StaticShapedTensor, staticShapedTensorPool));
    typeValues.insert(
        std::make_pair(PoolType::DynamicShapedMemref, dynamicShapedMemrefPool));
    typeValues.insert(
        std::make_pair(PoolType::StaticShapedMemref, staticShapedMemrefPool));
  }

  void clearAll() {
    for (auto &item : typeValues) {
      item.second.clear();
    }
    constantIndices.clear();
    affineExprPool.clear();
    affineMapPool.clear();
    integerSetPool.clear();
    unrankedMemrefPool.clear();
    unrankedTensorPool.clear();
  }

  void merge(TypedValuePool &pool) {
    staticShapedTensorPool.insert(staticShapedTensorPool.end(),
                                  pool.staticShapedTensorPool.begin(),
                                  pool.staticShapedTensorPool.end());
    dynamicShapedTensorPool.insert(dynamicShapedTensorPool.end(),
                                   pool.dynamicShapedTensorPool.begin(),
                                   pool.dynamicShapedTensorPool.end());
    staticShapedMemrefPool.insert(staticShapedMemrefPool.end(),
                                  pool.staticShapedMemrefPool.begin(),
                                  pool.staticShapedMemrefPool.end());
    dynamicShapedMemrefPool.insert(dynamicShapedMemrefPool.end(),
                                   pool.dynamicShapedMemrefPool.begin(),
                                   pool.dynamicShapedMemrefPool.end());
    unrankedTensorPool.insert(unrankedTensorPool.begin(),
                              pool.unrankedTensorPool.begin(),
                              pool.unrankedTensorPool.end());
    unrankedMemrefPool.insert(unrankedMemrefPool.begin(),
                              pool.unrankedMemrefPool.begin(),
                              pool.unrankedMemrefPool.end());
    intOrFloatPool.insert(intOrFloatPool.end(), pool.intOrFloatPool.begin(),
                          pool.intOrFloatPool.end());
    indexPool.insert(indexPool.end(), pool.indexPool.begin(),
                     pool.indexPool.end());
    vectorPool.insert(vectorPool.end(), pool.vectorPool.begin(),
                      pool.vectorPool.end());
    constantIndices.insert(pool.constantIndices.begin(),
                           pool.constantIndices.end());
    affineExprPool.insert(affineExprPool.end(), pool.affineExprPool.begin(),
                          pool.affineExprPool.end());
    affineMapPool.insert(affineMapPool.end(), pool.affineMapPool.begin(),
                         pool.affineMapPool.end());
    integerSetPool.insert(integerSetPool.begin(), pool.integerSetPool.begin(),
                          pool.integerSetPool.end());
  }

  void addTypeValue(const TypeValue &typedValue, std::string op = "", PoolType ty = PoolType::Unknown);
  void addIntOrFloat(const TypeValue &val, std::string op);
  void addElement(const TypeValue &element, std::string op);
  void addRankedMemref(const TypeValue &memref, std::string op);
  void addUnrankedMemref(const TypeValue &unrankedTensor, std::string op);
  void addStaticShapedMemref(const TypeValue &memref, std::string op);
  void addDynamicShapedMemref(const TypeValue &memref, std::string op);
  void addRankedTensor(const TypeValue &tensor, std::string op);
  void addUnrankedTensor(const TypeValue &unrankedTensor, std::string op);
  void addStaticShapedTensor(const TypeValue &staticTensor, std::string op);
  void addDynamicShapedTensor(const TypeValue &dynamicTensor, std::string op);
  void addVector(const TypeValue &vector, std::string op);
  void addIndex(const TypeValue &index, std::string op);

  /* Giving TypeValue constraints, get candidates from pool  */
  std::vector<TypeValue> getCandidatesFromIntOrFloats(OpBuilder &builder,
                                                      Location loc, Type t);
  std::vector<TypeValue> getCandidatesFromStaticShapedMemref(OpBuilder &builder,
                                                             Location loc,
                                                             MemRefType t);
  std::vector<TypeValue>
  getCandidatesFromDynamicShapedMemref(OpBuilder &builder, Location loc,
                                       MemRefType t);
  std::vector<TypeValue>
  getCandidatesFromStaticShapedTensor(OpBuilder &builder, Location loc, Type t);
  std::vector<TypeValue>
  getCandidatesFromDynamicShapedTensor(OpBuilder &builder, Location loc,
                                       Type t);
  std::vector<TypeValue> getCandidatesFromVector(OpBuilder &builder,
                                                 Location loc, Type t);
  std::vector<TypeValue> getCandidatesFromIndex(OpBuilder &builder,
                                                Location loc);
  std::vector<TypeValue>
  searchCandidatesFrom(std::vector<PoolType>,
                       TypeValueFilter typeValueFilter = emptyFilter());

  /* generate a new value corresponding to provided type, add the return value
   * to the pool */
  TypeValue generateInteger(OpBuilder &builder, Location loc, IntegerType type);

  TypeValue generateFloat(OpBuilder &builder, Location loc, FloatType type);

  TypeValue generateIndex(OpBuilder &builder, Location loc);
  Value getConstantIndex(OpBuilder &builder, Location loc, int num);

  TypeValue generateElement(OpBuilder &builder, Location loc, Type type);

  TypeValue generateRankedTensor(OpBuilder &builder, Location loc,
                                 RankedTensorType type);

  // generate an empty s tensor
  TypeValue generateStaticShapedTensor(OpBuilder &builder, Location loc,
                                       RankedTensorType type);
  // generate an empty d tensor
  TypeValue generateDynamicShapedTensor(OpBuilder &builder, Location loc,
                                        RankedTensorType type);

  // generate an empty memref, assume it is ranked.
  TypeValue generateRankedMemref(OpBuilder &builder, Location loc,
                                 MemRefType type);
  TypeValue generateStaticShapedMemref(OpBuilder &builder, Location loc,
                                       MemRefType type);

  TypeValue generateDynamicShapedMemref(OpBuilder &builder, Location loc,
                                        MemRefType type);

  // generate an empty vector
  TypeValue generateVector(OpBuilder &builder, Location loc, VectorType type);

  TypeValue generateTypedValue(OpBuilder &builder, Location loc, Type type);

  SmallVector<Value> randomIndicesForShapedType(ShapedType shapedType,
                                                OpBuilder &builder,
                                                Location loc);
};

TypeValue sampleTypedValueFrom(TypedValuePool pool);

std::vector<TypeValue> searchTypedValueFrom(const TypedValuePool pool,
                                            Type type);

#endif // MLIR_FUZZ_TYPEDVALUEPOOL_H
