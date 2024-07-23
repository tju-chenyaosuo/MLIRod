#ifndef MLIR_MUTATION_UTIL_H
#define MLIR_MUTATION_UTIL_H

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Operation.h"

#include <stdlib.h>

#include "Debug.h"


using namespace mlir;

// -------------------------------------------------------------------------
// Function with std::string or for print infomation.
inline std::string getValueTypeStr(Value v) {
  Type vTy = v.getType();

  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  vTy.print(stream);

  std::string ty = stream.str().str();
  return ty;
}

inline std::string getOperationName(Operation* o) {
  std::string s = "Unknown";
  if (o) {
    s = o->getName().getStringRef().str();
  }
  return s;
}

#ifdef DEBUG
inline void printOpInfo(Operation* o) {
  llvm::outs() << "Operation(name: " << o->getName().getStringRef() 
               << ", loc: " << o->getLoc() << ")";
}

inline void printValueInfo(mlir::Value v) {
  llvm::outs() << "Value(type: " << getValueTypeStr(v) 
               << ", loc: " << v.getLoc() << ")";
}
#endif

// -------------------------------------------------------------------------
// Functions with random / probability / decision
inline bool rollDice(double prob) {
  // llvm::outs() << "[rollDice] " << std::to_string(rand()) << "," << std::to_string(prob) << "\n";
  return ((double)rand() / RAND_MAX) < prob;
}

inline int rollIdx(int size) {
  return rand() % size;
}



#endif // MLIR_MUTATION_UTIL_H
