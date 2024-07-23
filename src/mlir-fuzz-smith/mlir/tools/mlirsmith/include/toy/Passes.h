//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {
// Count coverages
std::unique_ptr<mlir::Pass> createOpPrinterPass();

// Entrance of mlirsmith.
std::unique_ptr<mlir::Pass> createMLIRSmithPass();

int printConfig();

} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
