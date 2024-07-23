//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "smith/DiversityCriteria.h"
#include "smith/ExpSetting.h"
#include "smith/TmplInstantiation.h"
#include "smith/config.h"
#include "smith/MLIRSmith.h"
#include "smith/generators/OpGeneration.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
// #include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string>
    configFileName("c", cl::desc("Specify json config filename"),
                   cl::value_desc("config file name"));

static cl::opt<bool> isDiverse("d", cl::desc("Generate diversely"));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum Action {
  None,
  DumpConfig,
  DumpTmpl,
  TmplInstantiation,
  MLIRSmith,
  CoveredOp
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpConfig, "config",
                          "output the randomly generated configuration")),
    cl::values(clEnumValN(MLIRSmith, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpTmpl, "tmpl-gen",
                          "generate MLIR template randomly")),
    cl::values(clEnumValN(TmplInstantiation, "tmpl-inst",
                          "instantiate MLIR template")),
    cl::values(clEnumValN(CoveredOp, "covered-op",
                          "print covered ops for given input mlir program")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int gen(mlir::MLIRContext &context) {
  auto module = mlirSmith(context);
  if (!module) {
    return 1;
  }
  module->dump();
  return 0;
}

int instantiateTmpl(mlir::MLIRContext &context) {

  std::cout << "Instantiating template: " + inputFilename << std::endl;
  json tmpl;
  std::ifstream tmplFile(inputFilename);
  tmpl = json::parse(tmplFile, nullptr, false);
  if (tmpl.is_discarded()) {
    llvm::errs() << "template parse error"
                 << "\n";
    return 6;
  }
  mlir::OwningOpRef<mlir::ModuleOp> module = tmplInstantiation(context, tmpl);
  if (!module) {
    return 1;
  }
  module->dump();
  return 0;
}

int dumpMLIR() {
  if (configFileName.empty()) {
//    std::cout << "No custum configuration" << std::endl;
    is_default_config = true;
  }
  initConfig(configFileName);
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

//  diverse = isDiverse;
//  //  std::cout << "d: " << diverse << std::endl;
//  diversity.import("cov.json");
  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();
  context.getOrLoadDialect<mlir::amdgpu::AMDGPUDialect>();
  context.getOrLoadDialect<mlir::amx::AMXDialect>();
  context.getOrLoadDialect<mlir::arm_neon::ArmNeonDialect>();
  context.getOrLoadDialect<mlir::arm_sme::ArmSMEDialect>();
  // context.getOrLoadDialect<mlir::arm_sve::ArmSVEDialect>();
  context.getOrLoadDialect<mlir::async::AsyncDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::complex::ComplexDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::DLTIDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::irdl::IRDLDialect>();
  context.getOrLoadDialect<mlir::index::IndexDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::ml_program::MLProgramDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::nvgpu::NVGPUDialect>();
  context.getOrLoadDialect<mlir::acc::OpenACCDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::pdl_interp::PDLInterpDialect>();
  context.getOrLoadDialect<mlir::quant::QuantizationDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
  context.getOrLoadDialect<mlir::shape::ShapeDialect>();
  context.getOrLoadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  context.getOrLoadDialect<mlir::transform::TransformDialect>();
  context.getOrLoadDialect<mlir::ub::UBDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::x86vector::X86VectorDialect>();

  if (emitAction == Action::TmplInstantiation) {
    return instantiateTmpl(context);
  }

  if (emitAction == Action::MLIRSmith) {
    return gen(context);
  }

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Generate MLIR program from scratch.
  if (emitAction == Action::MLIRSmith) {
    pm.addPass(mlir::toy::createMLIRSmithPass());
  }
  if (emitAction == Action::CoveredOp) {
    pm.addPass(mlir::toy::createOpPrinterPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  module->dump();

  return 0;
}

int dumpConfig() {
  mlir::toy::printConfig();
  return 0;
}

json createRegion(std::string opName, int depth) {
  json region;
  int maxDepth = 3;
  int maxLength = 128;
  if (depth >= maxDepth || opNests.find(opName) == opNests.end()) {
    region = opName;
    return region;
  }
  auto ops = std::vector<std::string>(opNests[opName].begin(), opNests[opName].end());

  std::vector<json> opSeq;
  for (int i = 0; i < UR(maxLength); ++i) {
    int idx = UR(ops.size());
    auto op = ops[idx];
    if (opNests.find(op) == opNests.end()) {
      opSeq.push_back(createRegion(op, depth + 1));
    }
  }
  region[opName] = opSeq;
  return region;
}

int dumpTmpl() {
  json tmpl;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);

  /* using nano-second instead of seconds */
  srand((time_t)ts.tv_nsec);

  std::vector<json> funcs;
  for (int i = 0; i < func_num; ++i) {
    funcs.push_back(createRegion("func.func", 0));
  }
  tmpl["builtin.module"] = funcs;
  std::cout << tmpl.dump(4) << "\n";
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  llvm::InitLLVM(argc, argv);

  switch (emitAction) {
  case Action::DumpConfig:
    return dumpConfig();
  case Action::MLIRSmith:
  case Action::CoveredOp:
  case Action::TmplInstantiation:
    return dumpMLIR();
  case Action::DumpTmpl:
    return dumpTmpl();
  default: {
    emitAction = Action::MLIRSmith;
    return dumpMLIR();
  }
  }
}
