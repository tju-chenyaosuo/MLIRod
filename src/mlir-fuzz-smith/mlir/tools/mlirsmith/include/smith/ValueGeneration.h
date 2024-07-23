//
// Created by Stan Wang on 2022/12/4.
//

#ifndef MLIR_FUZZ_VALUEGENERATION_H
#define MLIR_FUZZ_VALUEGENERATION_H

#include "Util.h"
#include "smith/TypeValue.h"


// helper to fill a memref
void fillMemRef(Location loc, OpBuilder &b, int dimension, Value alloc,
                MemRefType memType, std::map<int, Value> &constantIndices,
                SmallVector<Value> indices);

Value initVector(OpBuilder &builder, Location loc, VectorType type);

Value initMemref(OpBuilder &builder, Location loc, MemRefType type);



#endif // MLIR_FUZZ_VALUEGENERATION_H
