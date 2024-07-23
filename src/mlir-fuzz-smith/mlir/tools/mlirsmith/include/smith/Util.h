//
// Created by Stan Wang on 2022/9/13.
//

#ifndef _UTIL_H_
#define _UTIL_H_
#include "smith/TypeValue.h"
#include "smith/config.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <iostream>
#include <random>

#include <cassert>

using namespace mlir;

//
// Created by Stan Wang on 2022/9/13.
//

bool isElementType(Type t);

std::vector<mlir::Type> getSupportedIntTypes(MLIRContext *ctx);
std::vector<mlir::Type> getSupportedFloatTypes(MLIRContext *ctx);
std::vector<mlir::Type> getSupportedIntOrFloatTypes(MLIRContext *ctx);

int min(int i1, int i2);
int max(int i1, int i2);

Value getConstant(Location loc, OpBuilder &b, Type type, long long val);

bool isTheSameShapedType(ShapedType typeA, ShapedType typeB);

long long UR(long long mod);

TypeValue sampleTypedValueFrom(std::vector<TypeValue> candidates,
                               std::string opToBuild="");

std::vector<TypeValue>
searchIntegerCandidatesFrom(const std::vector<TypeValue> pool);
std::vector<TypeValue>
searchFloatCandidatesFrom(const std::vector<TypeValue> pool);

std::vector<TypeValue> searchTypedValueFrom(const std::vector<TypeValue> pool,
                                            Type type);

// search single operand
TypeValue searchIntegerUnaryOperandFrom(OpBuilder &builder, Location loc,
                                        const std::vector<TypeValue> pool,
                                        std::string op);
TypeValue searchFloatUnaryOperandFrom(OpBuilder &builder, Location loc,
                                      const std::vector<TypeValue> pool,
                                      std::string op);

// search two operands for same operands and type.
std::vector<TypeValue>
searchIntegerBinaryOperandsFrom(OpBuilder &builder, Location loc,
                                const std::vector<TypeValue> pool,
                                std::string op);
std::vector<TypeValue>
searchFloatBinaryOperandsFrom(OpBuilder &builder, Location loc,
                              const std::vector<TypeValue> pool,
                              std::string op);
// search three
std::vector<TypeValue>
searchFloatTernaryOperandsFrom(OpBuilder &builder, Location loc,
                               const std::vector<TypeValue> pool,
                               std::string op);

/*------- shape related --------*/
// 2D mat-mul
std::vector<TypeValue> search2DTensorsFrom(const std::vector<TypeValue> pool);
std::vector<TypeValue> search2thTensorsFor(const std::vector<TypeValue> pool,
                                           ShapedType shapedType);

std::vector<TypeValue>
searchShapedInputFrom(ShapedType type, const std::vector<TypeValue> &pool);

/*-------- create new empty value -------*/ // Todo-deprecated
TypeValue createNewValue(OpBuilder &builder, Location loc, Type type);

size_t getWeightedRandomIndex(std::vector<float> weights);

void debugPrint(std::string);

void init();

std::string configJsonStr();

std::vector<AffineExpr> randomAffineExprs(OpBuilder &builder, int inputDim,
                                          int symbolNum);

inline bool isLLVMIRIntegerBitWidth(int intBitWidth) {
  return intBitWidth == 8 || intBitWidth == 16 || intBitWidth == 32 ||
         intBitWidth == 64;
}
// Block addEntryBlockForRegion(Region& region)

#endif