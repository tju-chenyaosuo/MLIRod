//
// Created by Stan Wang on 2022/9/13.
//
#ifndef MLIRSMITH_H
#define MLIRSMITH_H

#include "smith/RegionGeneration.h"
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "smith/DiversityCriteria.h"

using namespace mlir;

mlir::OwningOpRef<mlir::ModuleOp> mlirSmith(mlir::MLIRContext &context);

#endif