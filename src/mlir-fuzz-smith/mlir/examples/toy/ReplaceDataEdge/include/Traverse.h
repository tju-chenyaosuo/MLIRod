#ifndef TRAVERSE_h
#define TRAVERSE_h

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
  #ifdef DEBUG
  void print();
  #endif
  mlir::Value search(std::string ty, std::string operationName);
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
  #ifdef DEBUG
  llvm::outs() << "[MutationValuePool::merge] mvp info:";
  mvp.print();
  #endif

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

#ifdef DEBUG
void MutationValuePool::print() {
  llvm::outs() << "[MutationValuePool::print] \n";
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    std::string key = it->first;
    std::vector<mlir::Value> value = it->second;
    int count = 0;

    llvm::outs() << key << " : [";
    for (const mlir::Value& v : value) {
      mlir::Operation* o = v.getDefiningOp();
      llvm::StringRef opName = o->getName().getStringRef();
      mlir::Location loc = o->getLoc();
      llvm::outs() << "(" << opName << ", " << loc <<")";
      if (++count < value.size())
        llvm::outs() << ", ";
    }
    llvm::outs() << "]\n";
  }
}
#endif

mlir::Value MutationValuePool::search(std::string ty, std::string operationName) {
  std::vector<int> candidate = std::vector<int>();
  for (int idx = 0; idx < pool[ty].size(); ++idx) {
    mlir::Value var = pool[ty][idx];
    if (getOperationName(var.getDefiningOp()) != operationName)
      candidate.push_back(idx);
  }


  // if (!candidate.size()) {
  //   for (int idx = 0; idx < pool[ty].size(); ++idx) {
  //     mlir::Value var = pool[ty][idx];
  //     candidate.push_back(idx);
  //   }
  // }

  // // Verify whether 
  // std::string::size_type pos = opName.find('affine');
  // if (pos == std::string::npos) {
  //   #ifdef DEBUG
  //   llvm::outs() << "affine dialect mutation!\n"
  //   #endif
  //   for (int idx = 0; idx < candidate.size(); ++idx) {
  //     pool[ty][candidate[idx]];
  //   }
  // }


  int idx = 0;
  if (candidate.size()) {
    idx = candidate[rollIdx(candidate.size())];
  } else {
    idx = rollIdx(pool[ty].size());
  }
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
  #ifdef DEBUG
  void print();
  #endif
  mlir::Value search(std::string ty, std::string operationName);
};

void MutationBlock::add2Pool(mlir::Value& v) {
  #ifdef DEBUG
  llvm::outs() << "[MutationBlock::add2Pool] add info: " << v.getType() << ", " << v.getLoc() << "\n";
  #endif
  valuePool.addValue(v);
}

#ifdef DEBUG
void MutationBlock::print() {
  valuePool.print();
}
#endif

mlir::Value MutationBlock::search(std::string ty, std::string operationName) {
  return valuePool.search(ty, operationName);
}

// -------------------------------------------------------------------------
// parser/mutator
struct MutationParser {
  double operationProb;
  double operandProb;
  int mutationCnt;
  unsigned mutPos;
  unsigned curPos;

  MutationParser(double operandProb, unsigned mutPos) : 
                 operandProb(operandProb),
                 mutPos(mutPos) {
    mutationCnt = 0;
    curPos = 0;
  }

  void printOperation(Operation *op, MutationBlock* mb);
};

void MutationParser::printOperation(Operation *op, MutationBlock* mb) {
  
  // Mutator
  // TODO: This is a quick fix for invalid mutation, please replace this logic with official verify logic.
  std::vector<std::string> invalidParent = {"linalg.matmul", "linalg.dot"};
  std::vector<std::string> invalidOpts = {"affine.store", "tensor.unpack", "affine.load", "affine.vector_store", "affine.vector_load", "affine.if", "linalg.dot"};
  bool isValidOperation = true;
  std::string opName = getOperationName(op);
  std::string parentName = getOperationName(op->getParentOp());
  if (isValidOperation) {
    for (int idx = 0; idx < invalidOpts.size(); ++idx) {
      if (opName == invalidOpts[idx]) {
        isValidOperation = false;
        break;
      }
    }
  }
  if (isValidOperation) {
    for (int idx = 0; idx < invalidParent.size(); ++idx) {
      if (parentName == invalidParent[idx]) {
        isValidOperation = false;
        break;
      }
    }
  }

  bool operationDice = curPos >= mutPos;
  if (op->getNumOperands() && operationDice && isValidOperation) {
    for (int oIdx = 0; oIdx < op->getNumOperands(); oIdx++) {
      bool operandDice = rollDice(operandProb);
      if (!operandDice) {
        continue;
      }
      mlir::Value oldOperand = op->getOperand(oIdx);
      std::string oldTy = getValueTypeStr(oldOperand);
      mlir::Operation* oldOperandOperation = oldOperand.getDefiningOp();
      std::string oldOperandOpName = getOperationName(oldOperandOperation);
      
      // Mutation: replace old operation with a new one.
      if (!mb->valuePool.pool.count(oldTy)) {
        // non-candidates
        /* TODO: Add the value into function's arguments */
        continue;
      } else {
        // any-candidates
        mlir::Value newOperand = mb->search(oldTy, oldOperandOpName);
        op->setOperand(oIdx, newOperand);
        ++mutationCnt;
      }
    }
  }

  // Traverse first
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      MutationBlock* childMb = new MutationBlock(*mb);
      for (Operation &op : block.getOperations()) {
        printOperation(&op, childMb);
      }
      delete childMb;
    }
  }

  // Then add the return value of current operation into variable pool
  // Use this strategy to deal with some opreation that create the region.
  for (int rIdx = 0; rIdx < op->getNumResults(); rIdx ++) {
    Value res = op->getOpResult(rIdx);
    mb->add2Pool(res);
  }
  ++curPos;
}


#endif // TRAVERSE_h