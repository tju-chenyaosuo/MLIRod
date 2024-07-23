#ifndef TRAVERSE_h
#define TRAVERSE_h

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Value.h"

#include "MutationUtil.h"
#include "Debug.h"


// -------------------------------------------------------------------------
// parser/mutator
struct MutationParser {
  void printOperation(Operation *op, std::set<std::string>* information);
};

void MutationParser::printOperation(Operation *op, std::set<std::string>* information) {
  // Traverse first
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        printOperation(&op, information);
      }
    }
  }

  // Record information
  for (int rIdx = 0; rIdx < op->getNumResults(); rIdx ++) {
    Value res = op->getOpResult(rIdx);
    information->insert(getValueTypeStr(res));
  }
  information->insert(getOperationName(op));
}


#endif // TRAVERSE_h