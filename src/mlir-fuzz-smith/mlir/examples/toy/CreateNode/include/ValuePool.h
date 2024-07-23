#ifndef VALUE_POOL_HAAAAA
#define VALUE_POOL_HAAAAA

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMOps.h.inc"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "MutationUtil.h"
#include "Debug.h"

struct MutationValuePool {
  std::map<std::string, std::vector<mlir::Value>> pool;
  std::vector<mlir::AffineMap> affineMapPool;

  MutationValuePool(
    std::map<std::string, std::vector<mlir::Value>> pool = std::map<std::string, std::vector<mlir::Value>>(),
    std::vector<mlir::AffineMap> affineMapPool = std::vector<mlir::AffineMap>()
  ) : 
  pool(pool),
  affineMapPool(affineMapPool) {}

  // Functions for pool
  void addValue(mlir::Value& v);
  void merge(MutationValuePool mvp);
  mlir::Value search(std::string ty, std::string operationName);
  void search(bool (*filter)(std::string), std::vector<mlir::Value>* candidate);
  void search(bool (*filter)(mlir::Value), std::vector<mlir::Value>* candidate);
  // void search(bool (*filter)(std::string, mlir::Value), std::vector<mlir::Value>* candidate);
  void search(std::function<bool(std::string, mlir::Value)> filter, std::vector<mlir::Value>* candidate);
  void getAll(std::vector<mlir::Value>* candidate);

  // Functions for affine.map
  void addAffineMap(mlir::AffineMap& m);
};

struct MutationBlock {
  MutationValuePool valuePool;

  MutationBlock() {
    valuePool = MutationValuePool();
  }

  MutationBlock(const MutationBlock& block) {
    valuePool = MutationValuePool();
    valuePool.merge(block.valuePool);
  }

  void add2Pool(mlir::Value& v);
  mlir::Value search(std::string ty, std::string operationName);
  void search(bool (*filter)(std::string), std::vector<mlir::Value>* candidate);
  void search(bool (*filter)(mlir::Value), std::vector<mlir::Value>* candidate);
  // void search(bool (*filter)(std::string, mlir::Value), std::vector<mlir::Value>* candidate);
  void search(std::function<bool(std::string, mlir::Value)> filter, std::vector<mlir::Value>* candidate);
  void getAll(std::vector<mlir::Value>* candidate);

  void add2AffineMapPool(mlir::AffineMap& m);
};

bool int_float_index_filter(std::string s);
bool int_float_filter(std::string s);
bool float_filter(std::string s);
bool int_filter(std::string s);
bool index_filter(std::string s);
bool memref_filter(std::string s);
bool vector_filter(std::string s);

bool ranked_memref_filter(std::string s, mlir::Value v);
bool static_memref_filter(std::string s, mlir::Value v);
bool ranked_tensor_filter(std::string s, mlir::Value v);
bool static_tensor_filter(std::string s, mlir::Value v);
bool ranked_memref_tensor_filter(std::string s, mlir::Value v);
bool static_memref_tensor_filter(std::string s, mlir::Value v);
bool ranked_memref_tensor_filter_has_dim(std::string s, mlir::Value v);
bool static_memref_tensor_filter_has_dim(std::string s, mlir::Value v);



#endif // VALUE_POOL_H