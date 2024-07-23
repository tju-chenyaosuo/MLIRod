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

inline std::string getValueTypeStr(Type vTy) {
  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  vTy.print(stream);

  std::string ty = stream.str().str();
  return ty;
}

inline std::string getLocationStr(Operation* o) {
  Location loc = o->getLoc();

  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  loc.print(stream);

  std::string locStr = stream.str().str();
  return locStr;
}

inline std::string getOperationName(Operation* o) {
  std::string s = "Unknown";
  if (o) {
    s = o->getName().getStringRef().str();
  }
  return s;
}

inline std::string getOperationNameParamType(Operation* o) {
  std::string opName = "Unknown";
  std::string paramTys = "";
  if (o) {
    opName = o->getName().getStringRef().str();
    if (o->getNumOperands() != 0) {
      paramTys = getValueTypeStr(o->getOperand(0));
    }
    for (unsigned operandIdx = 1; operandIdx < o->getNumOperands(); operandIdx++) {
      std::string paramTy = getValueTypeStr(o->getOperand(operandIdx));
      paramTys = paramTys + ", " + paramTy;
    }
  }
  std::string s = opName + "(" + paramTys + ")";
  return s;
}

inline int getLineNumber(Operation* o) {
  // loc("test.mlir":1:1)
  if (!o) 
    return 0;
  std::string pos = getLocationStr(o);
  int end = pos.find_last_of(":");
  int start = pos.find_first_of(":");
  if (end == std::string::npos || start == std::string::npos)
    return getLineNumber(o->getParentOp());
  int strLen = end - start - 1;
  std::string line = pos.substr(start+1, strLen);
  return std::stoi(line);
}

inline std::string trim(std::string s) 
{
    if (!s.empty())  {
      s.erase(0, s.find_first_not_of(" "));
      s.erase(s.find_last_not_of(" ") + 1);
    }
    return s;
}


// -------------------------------------------------------------------------
// Functions with random / probability / decision
inline bool rollDice(double prob) {
  // llvm::outs() << "[rollDice] " << std::to_string(rand()) << "," << std::to_string(prob) << "\n";
  return ((double)rand() / RAND_MAX) < prob;
}

inline int rollIdx(int size) {
  return rand() % size;
}

// -------------------------------------------------------------------------
// Functions for extract information(for easier use of MLIR-API)
// Return negative number if o1 before o2, zero if o1 is o2, positive number if o1 after o2.
inline int comparePosition(Operation* o1, Operation* o2) {
  int line1 = getLineNumber(o1);
  int line2 = getLineNumber(o2);
  return line1 - line2;
}

inline unsigned getControlDepth(Operation* o, unsigned depth=0) {
  Operation* parent = o->getParentOp();
  if (parent) {
    ++depth;
    return getControlDepth(parent, depth);
  }
  return depth;
}




#endif // MLIR_MUTATION_UTIL_H
