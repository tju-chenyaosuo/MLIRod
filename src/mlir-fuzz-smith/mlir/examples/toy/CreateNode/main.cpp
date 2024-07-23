// #include "smith/DiversityCriteria.h"
// #include "smith/ExpSetting.h"
// #include "smith/config.h"
// #include "toy/Dialect.h"
// #include "toy/MLIRGen.h"
// #include "toy/Parser.h"
// #include "toy/Passes.h"

// #include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/IR/AsmState.h"
// #include "mlir/IR/BuiltinOps.h"


#include "mlir/IR/MLIRContext.h"
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
#include <ctime>
#include <fstream>

#include "Traverse.h"
#include "Debug.h"

using namespace mlir;

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
static cl::opt<std::string> outputFilename(cl::Positional,
                                          cl::desc("<output mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
// static cl::opt<std::string> generateNmae("generate-name",
//                             cl::desc("Expected node name to be generated"),
//                             cl::init("None"),
//                             cl::value_desc("const string."));

unsigned lineNumber() {
  std::ifstream file(inputFilename);
  if (!file.is_open()) {
      std::cerr << "Unable to open file: " << inputFilename << std::endl;
      return 1;
  }
  std::string line;
  unsigned lineNumber = 0;
  while (std::getline(file, line)) {
      lineNumber++;
      // Process the line as needed
      std::cout << "Line " << lineNumber << ": " << line << std::endl;
  }
  file.close();
  return lineNumber;
}

// Usage: /data/src/mlirsmith-dev/build/bin/mutator --operation-prob=0.1 test.mlir 1.mlir
int main(int argc, char **argv) {
  // regist the options
  cl::ParseCommandLineOptions(argc, argv, "mlir mutator\n");
  llvm::InitLLVM(argc, argv);

  // Load the mlir file
  MLIRContext context;
  registerAllDialects(context);
  context.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module;
  llvm::SourceMgr sourceMgr;

  for (auto op : context.getRegisteredOperations()) {
    llvm::outs() << op << "\n";
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[main] Error can't load file " << inputFilename << "\n";
    return 3;
  }
  #ifdef DEBUG
  llvm::errs() << "[main] Load file " << inputFilename << " success!\n";
  #endif

  // auto firstOp = module.getOps();
  // llvm::errs() << "[main] " << firstOp.getName() << "\n";

  unsigned fileLineNumber = lineNumber();
  unsigned mutPos = rollIdx(fileLineNumber);

  srand(time(0));
  Operation* op = module.get();
  MutationParser mp = MutationParser(op, mutPos);
  MutationBlock* mb = new MutationBlock();
  mp.printOperation(op, mb);

  std::error_code error;
  llvm::raw_fd_ostream output(outputFilename, error);
  if (error) {
      llvm::errs() << "Error opening file for writing: " << error.message() << "\n";
      return 1;
  }

  llvm::outs() << "[main] End traverse!\n";

  module->print(output);

  llvm::outs() << "[main] Mutation count: " << std::to_string(mp.mutationCnt) << "\n";

  return 0;

}