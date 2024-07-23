//
// Created by Stan Wang on 2022/12/4.
//

#include "../include/smith/ValueGeneration.h"

void fillMemRef(Location loc, OpBuilder &b, int dimension, Value alloc,
                MemRefType memType, std::map<int, Value> &constantIndices,
                SmallVector<Value> indices) {
  auto shape = memType.getShape();
  auto eleType = memType.getElementType();
  if (dimension == shape.size()) {
    b.create<affine::AffineStoreOp>(loc, getConstant(loc, b, eleType, 1), alloc,
                            llvm::makeArrayRef(indices));
    return;
  }
  // Otherwise, iterate over the current dimension and add the indices
  // to the list.
  for (uint64_t i = 0, e = shape[dimension]; i != e; ++i) {
    indices.push_back(constantIndices[i]);
    fillMemRef(loc, b, dimension + 1, alloc, memType, constantIndices, indices);
    indices.pop_back();
  }
}

Value initMemref(OpBuilder &builder, Location loc, MemRefType type) {
  return builder.create<memref::AllocOp>(loc, type);
}
