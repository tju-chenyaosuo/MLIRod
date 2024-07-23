//
// Created by Stan Wang on 2023/9/8.
//
#ifndef MLIR_FUZZ_TMPLINSTANTIATION_H
#define MLIR_FUZZ_TMPLINSTANTIATION_H

#include "json.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "smith/generators/OpGeneration.h"
#include "smith/RegionGeneration.h"
#include "smith/TypeGeneration.h"
using json = nlohmann::json;

mlir::OwningOpRef<mlir::ModuleOp> tmplInstantiation(mlir::MLIRContext &context,
                                          json tmpl);

#endif // MLIR_FUZZ_TMPLINSTANTIATION_H

