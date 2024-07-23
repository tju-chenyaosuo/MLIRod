#include "ValuePool.h"

// -------------------------------------------------------------------------
void MutationValuePool::addValue(mlir::Value& v) {
  std::string vTy = getValueTypeStr(v);
  std::vector<mlir::Value> typedValuePool;
  if (!pool.count(vTy))
    pool[vTy] = std::vector<mlir::Value>();
  pool[vTy].push_back(v);
  return ;
}

// Merge the existing MutationValuePool into this pool.
void MutationValuePool::merge(MutationValuePool mvp) {
  // merge the pool
  for (auto it = mvp.pool.begin(); it != mvp.pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;

    if (!pool.count(key)) {
      pool.insert(std::make_pair(key, std::vector<mlir::Value>()));
    }
    pool[key].insert(pool[key].end(), mvp.pool[key].begin(), mvp.pool[key].end());
  }

  // merge the affine map pool
  affineMapPool.insert(affineMapPool.end(), mvp.affineMapPool.begin(), mvp.affineMapPool.end());
  return ;
}

/**
 * Find a mlir::Value from current region, acoording to the type.
 * The type here should strictly match the existing mlir::Value in this region.
*/
mlir::Value MutationValuePool::search(std::string ty, std::string operationName) {
  std::vector<int> candidate = std::vector<int>();
  for (int idx = 0; idx < pool[ty].size(); ++idx) {
    mlir::Value var = pool[ty][idx];

    if (getOperationName(var.getDefiningOp()) != operationName)
      candidate.push_back(idx);
  }
  int idx = 0;
  if (candidate.size()) {
    idx = candidate[rollIdx(candidate.size())];
  } else {
    idx = rollIdx(pool[ty].size());
  }
  return pool[ty][idx];
}

/**
 * Find all of the mlir::Value in current region, function point filter determines whether the value is under requirement.
*/
void MutationValuePool::search(bool (*filter)(std::string), std::vector<mlir::Value>* candidate) {
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;
    if (filter(key))
      candidate->insert(candidate->end(), value.begin(), value.end());
  }
}

void MutationValuePool::search(bool (*filter)(mlir::Value), std::vector<mlir::Value>* candidate) {
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;
    for (mlir::Value v : value) {
      if (filter(v))
        candidate->push_back(v);
    }
  }
}

// void MutationValuePool::search(bool (*filter)(std::string, mlir::Value), std::vector<mlir::Value>* candidate) {
//   for (auto it = pool.begin(); it != pool.end(); ++it) {
//     std::string key = it->first;
//     std::vector<mlir::Value> value = it->second;
//     for (mlir::Value v : value) {
//       if (filter(key, v))
//         candidate->push_back(v);  
//     }
//   }
// }

void MutationValuePool::search(std::function<bool(std::string, mlir::Value)> filter, std::vector<mlir::Value>* candidate) {
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;
    for (mlir::Value v : value) {
      if (filter(key, v))
        candidate->push_back(v);  
    }
  }
}

void MutationValuePool::addAffineMap(mlir::AffineMap& m) {
  affineMapPool.push_back(m);
}

void MutationValuePool::getAll(std::vector<mlir::Value>* candidate) {
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;
    candidate->insert(candidate->end(), value.begin(), value.end());
  }
}

// -------------------------------------------------------------------------
void MutationBlock::add2Pool(mlir::Value& v) {
  valuePool.addValue(v);
}

mlir::Value MutationBlock::search(std::string ty, std::string operationName) {
  return valuePool.search(ty, operationName);
}

void MutationBlock::search(bool (*filter)(std::string), std::vector<mlir::Value>* candidate) {
  valuePool.search(filter, candidate);
}

void MutationBlock::search(bool (*filter)(mlir::Value), std::vector<mlir::Value>* candidate) {
  valuePool.search(filter, candidate);
}

// void MutationBlock::search(bool (*filter)(std::string, mlir::Value), std::vector<mlir::Value>* candidate) {
//   valuePool.search(filter, candidate);
// }

void MutationBlock::search(std::function<bool(std::string, mlir::Value)> filter, std::vector<mlir::Value>* candidate) {
  valuePool.search(filter, candidate);
}

void MutationBlock::add2AffineMapPool(mlir::AffineMap& m) {
  valuePool.addAffineMap(m);
}

void MutationBlock::getAll(std::vector<mlir::Value>* candidate) {
  valuePool.getAll(candidate);
}

// -------------------------------------------------------------------------
// Some variable seach method
bool int_float_index_filter(std::string s) {
  return (s.find("i") == 0 || s.find("f") == 0 || s.find("index") == 0);
}

bool int_float_filter(std::string s) {
  return ((s.find("i") == 0 || s.find("f") == 0) && s.find("index") != 0);
}

bool float_filter(std::string s) {
  return s.find("f") == 0;
}

bool int_filter(std::string s) {
  return s.find("i") == 0 && s.find("index") != 0;
}

bool index_filter(std::string s) {
  return s.find("index") == 0;
}

bool memref_filter(std::string s) {
  return s.find("memref") == 0;
}

bool vector_filter(std::string s) {
  return s.find("vector") == 0;
}

bool ranked_memref_filter(std::string s, mlir::Value v) {
  bool cond1 = (s.find("memref") == 0);
  if (!cond1)  // trancate
    return false;
  mlir::Type valueType = v.getType();
  bool cond2 = false;
  if (auto rankedMemRefType = valueType.dyn_cast<mlir::MemRefType>()) {
    cond2 = true;
  }
  return cond1 && cond2;
}

bool static_memref_filter(std::string s, mlir::Value v) {
  bool cond1 = (s.find("memref") == 0);
  if (!cond1)  // trancate
    return false;
  mlir::Type valueType = v.getType();
  bool cond2 = false;
  if (auto rankedMemRefType = valueType.dyn_cast<mlir::MemRefType>()) {
    bool cond2 = rankedMemRefType.hasStaticShape();
  }
  return cond1 && cond2;
}

bool ranked_tensor_filter(std::string s, mlir::Value v) {
  bool cond1 = (s.find("tensor") == 0);
  if (!cond1)  // trancate
    return false;
  mlir::Type valueType = v.getType();
  bool cond2 = false;
  if (auto rankedTensorType = valueType.dyn_cast<mlir::TensorType>()) {
    cond2 = rankedTensorType.hasRank();
  }
  return cond2;
}

bool static_tensor_filter(std::string s, mlir::Value v) {
  bool cond1 = (s.find("tensor") == 0);
  if (!cond1)  // trancate
    return false;
  mlir::Type valueType = v.getType();
  bool cond2 = false;
  if (auto rankedTensorType = valueType.dyn_cast<mlir::TensorType>()) {
    cond2 = rankedTensorType.hasStaticShape();
  }
  return cond2;
}

bool ranked_memref_tensor_filter(std::string s, mlir::Value v) {
  return ranked_tensor_filter(s, v) || ranked_memref_filter(s, v);
}

bool static_memref_tensor_filter(std::string s, mlir::Value v) {
  return static_memref_filter(s, v) || static_tensor_filter(s, v);
}

bool ranked_memref_tensor_filter_has_dim(std::string s, mlir::Value v) {
  bool cond1 = ranked_tensor_filter(s, v) || ranked_memref_filter(s, v);
  if (!cond1) { return false; }
  bool cond2 = v.getType().dyn_cast<ShapedType>().getRank() > 0;
  return cond2;
}

bool static_memref_tensor_filter_has_dim(std::string s, mlir::Value v) {
  bool cond1 = static_tensor_filter(s, v) || static_memref_filter(s, v);
  if (!cond1) { return false; }
  bool cond2 = v.getType().dyn_cast<ShapedType>().getRank() > 0;
  return cond2;
}