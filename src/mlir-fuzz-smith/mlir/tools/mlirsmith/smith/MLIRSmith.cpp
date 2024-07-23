//
// Created by Stan Wang on 2022/9/13.
//

#include "smith/generators/OpGeneration.h"
#include "smith/MLIRSmith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

using namespace mlir;

void consumeEachTensor(OpBuilder &builder, Location loc, TypedValuePool &p) {
  for (auto tval : p.staticShapedTensorPool) {
    auto tensorType = tval.type.dyn_cast<RankedTensorType>();
    if (tensorType) {
      auto shapedType = tval.type.dyn_cast<ShapedType>();
      auto memrefType =
          MemRefType::get(shapedType.getShape(), shapedType.getElementType());

      auto mem = TypeValue(memrefType,
                           builder.create<memref::AllocOp>(loc, memrefType));
      p.addStaticShapedMemref(mem, "memref.alloc(consumeT)");

      builder.create<memref::TensorStoreOp>(loc, tval.val, mem.val);
    }
  }
}

struct GenOpLowering : public OpRewritePattern<toy::GenOp> {
  using OpRewritePattern<toy::GenOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::GenOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    /* using nano-second instead of seconds */
    srand((time_t)ts.tv_nsec);

    OpRegion region("builtin.module", 0);
    std::set<std::string> opsForModule = {"func.func"};
    auto regionGen = RegionGen(&region, {OpNameFilter(opsForModule)});
    regionGen.apply(rewriter, loc, func_num);

    // ----------------------- End of Random Generating --------------------
    rewriter.eraseOp(op);
    return success();
  }
};


namespace {
struct MLIRSmithPass
    : public PassWrapper<MLIRSmithPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MLIRSmithPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        scf::SCFDialect, linalg::LinalgDialect, affine::AffineDialect,
        arith::ArithDialect, index::IndexDialect, memref::MemRefDialect,
        BuiltinDialect, math::MathDialect, bufferization::BufferizationDialect,
        func::FuncDialect, vector::VectorDialect, spirv::SPIRVDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void MLIRSmithPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arithmetic`, `MemRef`, and `Standard` dialects.
  target.addLegalDialect<
      scf::SCFDialect, linalg::LinalgDialect, affine::AffineDialect,
      arith::ArithDialect, memref::MemRefDialect, math::MathDialect,
      func::FuncDialect, index::IndexDialect, BuiltinDialect,
      bufferization::BufferizationDialect, tensor::TensorDialect,
      vector::VectorDialect, spirv::SPIRVDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<toy::ToyDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<GenOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

int mlir::toy::printConfig() {
  std::cout << configJsonStr() << std::endl;
  return 0;
}

void init(){
  initType();
  registerSPIRVGenerators();
}

class MLIRSmithImpl {
public:
  MLIRSmithImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp smith() {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    init();
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto point = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&theModule.getBodyRegion().front());
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    /* using nano-second instead of seconds */
    srand((time_t)ts.tv_nsec);

    OpRegion region("builtin.module", 0);
    std::set<std::string> opsForModule = {"func.func"};
    auto regionGen = RegionGen(&region, {OpNameFilter(opsForModule)});
    regionGen.apply(builder, builder.getUnknownLoc(), func_num);

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

mlir::OwningOpRef<mlir::ModuleOp> mlirSmith(mlir::MLIRContext &context) {
  return MLIRSmithImpl(context).smith();
}

std::unique_ptr<Pass> mlir::toy::createMLIRSmithPass() {
  //  initConfig();
  init();
  return std::make_unique<MLIRSmithPass>();
}
