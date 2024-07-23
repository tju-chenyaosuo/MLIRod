#ifndef TRAVERSE_h
#define TRAVERSE_h

#include <algorithm>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Value.h"

#include "MutationUtil.h"
#include "Debug.h"


// -------------------------------------------------------------------------
// To record the variables in the region.
struct MutationValuePool {
  std::map<std::string, std::vector<mlir::Value>> pool;

  MutationValuePool(
    std::map<std::string, std::vector<mlir::Value>> pool = std::map<std::string, std::vector<mlir::Value>>()
  ) : pool(pool) {}

  void addValue(mlir::Value& v);
  void merge(MutationValuePool mvp);
  void clear();
  mlir::Value search(std::string ty, std::string operationName);
  mlir::Value searchFilter(std::string ty, std::string operationId);
};

// Add a value to MutationValuePool
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

  for (auto it = mvp.pool.begin(); it != mvp.pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;
    if (!pool.count(key)) {
      pool.insert(std::make_pair(key, std::vector<mlir::Value>()));
    }
    pool[key].insert(pool[key].end(), mvp.pool[key].begin(), mvp.pool[key].end());
  }
  return ;
}

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

mlir::Value MutationValuePool::searchFilter(std::string ty, std::string operationId) {
  std::vector<int> candidate = std::vector<int>();
  for (int idx = 0; idx < pool[ty].size(); ++idx) {
    mlir::Value var = pool[ty][idx];
    mlir::Operation* op = var.getDefiningOp();
    std::string opId = getOperationName(op) + "-" + getLocationStr(op);
    if (opId != operationId)
      candidate.push_back(idx);
  }
  unsigned idx = candidate[rollIdx(candidate.size())];
  return pool[ty][idx];
}

// -------------------------------------------------------------------------
// A warpper of region.
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
  mlir::Value searchFilter(std::string ty, std::string operationId);
};

void MutationBlock::add2Pool(mlir::Value& v) {
  valuePool.addValue(v);
}



mlir::Value MutationBlock::search(std::string ty, std::string operationName) {
  return valuePool.search(ty, operationName);
}

mlir::Value MutationBlock::searchFilter(std::string ty, std::string operationId) {
  return valuePool.searchFilter(ty, operationId);
}

// -------------------------------------------------------------------------
// parser/mutator
struct MutationParser {
  double operationProb;
  unsigned mutationCnt;
  unsigned mutPos;
  unsigned curPos;

  MutationParser(unsigned mutPos) : 
                 mutPos(mutPos) { 
                  mutationCnt = 0;
                  curPos = 0;
                 }

  void printOperation(Operation *op, MutationBlock* mb);
};

bool hasSameValue(mlir::Operation* op, MutationBlock* mb) {
  bool res = true;
  for (int rIdx = 0; rIdx < op->getNumResults(); rIdx ++) {
    Value resVal = op->getOpResult(rIdx);
    std::string ty = getValueTypeStr(resVal);  
    res = res && (mb->valuePool.pool.count(ty) > 1);
    if (!res) 
      return res;
  }
  return res;
}

void getDerectDependencies(mlir::Operation* op, std::vector<mlir::Operation*>* dependencies) {
  auto users = op->getUsers();
  std::vector<std::string> opIds = std::vector<std::string>();
  for (mlir::Operation* u : users) {
    std::string opId = getOperationName(u) + "-" + getLocationStr(u);
    if (std::find(opIds.begin(), opIds.end(), opId) == opIds.end()) {
      opIds.push_back(opId);
      dependencies->push_back(u);
    }
  }
}

void changeUsage(mlir::Operation* op, std::vector<mlir::Operation*>* users, MutationBlock* mb) {
  std::string opId = getOperationName(op) + "-" + getLocationStr(op);
  std::string opName = getOperationName(op);
  for (mlir::Operation* u : *users) {
    // Find the corresponding result, and replace its value.
    for (int oIdx = 0; oIdx < u->getNumOperands(); oIdx++) {
      mlir::Value oldOperand = u->getOperand(oIdx);
      mlir::Operation* oldOperandOperation = oldOperand.getDefiningOp();
      std::string oldOpId = getOperationName(oldOperandOperation) + "-" + getLocationStr(oldOperandOperation);
      
      llvm::outs() << opId << " => " << oldOpId << "\n";
      
      if (oldOpId == opId) {
        std::string oldTy = getValueTypeStr(oldOperand);
        mlir::Value newOperand = mb->searchFilter(oldTy, opName);
        u->setOperand(oIdx, newOperand);
      }

      llvm::outs() << "end change\n";
    }
  }
}

void MutationParser::printOperation(Operation *op, MutationBlock* mb) {
  #ifdef DEBUG
  llvm::outs() << "[MutationParser::printOperation] traverse: " << getOperationName(op) << "-" << getLocationStr(op) << "\n";
  #endif

  // traverse first
  unsigned oldMutation = mutationCnt;
  
  // Mutation logic
  bool isValidOperation = true;
  std::string opName = getOperationName(op);
  std::vector<std::string> invalidOpts = {"builtin.module", "func.func", "scf.yield", "linalg.yield", "vector.yield",
                                          "scf.reduce.return", "affine.yield", "scf.condition", "memref.alloca_scope.return", 
                                          "scf.reduce", "tensor.yield", };
  
  if (curPos >= mutPos) {
    if (isValidOperation) {
      for (int idx = 0; idx < invalidOpts.size(); ++idx) {
        if (opName == invalidOpts[idx]) {
          isValidOperation = false;
          break;
        }
      }
    }
    if (isValidOperation) {
      bool precondition = hasSameValue(op, mb);
      if (precondition) {
        // Extract all of the users of op
        std::vector<mlir::Operation*> dependencies = std::vector<mlir::Operation*>();
        getDerectDependencies(op, &dependencies);

        llvm::outs() << "[MutationParser::printOperation] Selected Operation: " << getOperationName(op) << "-" << getLocationStr(op) << "\n";
        for (mlir::Operation* o : dependencies) 
          llvm::outs() << "[MutationParser::printOperation] " << getOperationName(o) << "-" << getLocationStr(o) << "\n";

        // Replace the operands of users
        changeUsage(op, &dependencies, mb);

        #ifdef DEBUG
        llvm::outs() << "[MutationParser::printOperation] Erase Op\n";
        #endif
        // Remove this operation
        op->erase();
        ++mutationCnt;
      }
    }
  }

  if (oldMutation != mutationCnt) {  return ;  }
  
  for (Region &region : op->getRegions()) {
    if (oldMutation != mutationCnt) { break; }  // if mutate
    for (Block &block : region.getBlocks()) {
      if (oldMutation != mutationCnt) { break; }  // if mutate
      MutationBlock* childMb = new MutationBlock(*mb);
      for (Operation &op : block.getOperations()) {
        if (oldMutation != mutationCnt) { break; }  // if mutate
        printOperation(&op, childMb);
        if (oldMutation != mutationCnt) { break; }  // if mutate
      }
      delete childMb;
      if (oldMutation != mutationCnt) { break; }  // if mutate
    }
    if (oldMutation != mutationCnt) { break; }  // if mutate
  }

  if (oldMutation != mutationCnt) { return ; }

  for (int rIdx = 0; rIdx < op->getNumResults(); rIdx ++) {
    Value res = op->getOpResult(rIdx);
    mb->add2Pool(res);
  }
  ++curPos;
}


#endif // TRAVERSE_h