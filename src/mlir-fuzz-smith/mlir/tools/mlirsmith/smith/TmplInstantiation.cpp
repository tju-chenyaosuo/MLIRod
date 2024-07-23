//
// Created by Stan Wang on 2023/9/8.
//

#include "smith/TmplInstantiation.h"
#include <iostream>

class TmplInstantiationImpl {
public:
  TmplInstantiationImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp tmplInstantiation(json tmpl) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    initType();
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto point = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&theModule.getBodyRegion().front());
    OpRegion region = OpRegion("builtin.module", 0, tmpl);
    auto regionGen = RegionGen(&region, {});
    regionGen.apply(builder, builder.getUnknownLoc(), 0);
    builder.restoreInsertionPoint(point);
    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;
};

mlir::OwningOpRef<mlir::ModuleOp> tmplInstantiation(mlir::MLIRContext &context,
                                                    json tmpl) {
  return TmplInstantiationImpl(context).tmplInstantiation(tmpl);
}