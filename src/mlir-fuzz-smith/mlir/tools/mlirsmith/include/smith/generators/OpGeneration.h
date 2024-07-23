//
// Created by Stan Wang on 2022/9/13.
//
#ifndef OP_GENERATOR_PROVIDER_H
#define OP_GENERATOR_PROVIDER_H

#include "../TypedValuePool.h"
#include "../Util.h"
#include <map>
#include <set>

using namespace mlir;

inline std::vector<arith::AtomicRMWKind> floatRmwKinds = {
    arith::AtomicRMWKind::addf, arith::AtomicRMWKind::assign,
    arith::AtomicRMWKind::mulf};

inline std::vector<arith::AtomicRMWKind> intRmwKinds = {
    arith::AtomicRMWKind::addi, arith::AtomicRMWKind::assign,
    arith::AtomicRMWKind::maxs, arith::AtomicRMWKind::maxu,
    arith::AtomicRMWKind::mins, arith::AtomicRMWKind::minu,
    arith::AtomicRMWKind::muli, arith::AtomicRMWKind::ori,
    arith::AtomicRMWKind::andi};

// TODO define parent region - define struct recursively
struct OpRegion {
  std::string parent_op;
  int depth;
  std::vector<std::string> filter;
  std::vector<std::string> terminatorOp;
  TypedValuePool pool;
  json tmpl;
  json cur_child;

  OpRegion(std::string op, int depth, json tmpl = nullptr)
      : parent_op(op), depth(depth), tmpl(tmpl) {
    pool = TypedValuePool();
    filter = std::vector<std::string>();
    terminatorOp = std::vector<std::string>();
  }
};

using OpGenerator =
    std::function<Operation *(OpBuilder &, Location, OpRegion &)>;

// TODO- some terminators are optional
// for example, scf.yield
inline std::map<std::string, std::vector<std::string>> terminatorMap = {
    {"func.return", {"func.func"}},
    {"linalg.yield", {"linalg.generic", "linalg.map", "linalg.reduce"}}};

OpGenerator emptyGenerator();

struct OpGen {
  std::string opName;
  // We do not maintain these flags for each Op Generation since the requirement
  // ] are specified by only a little ops. bool hasSideEffect;
  OpGenerator generator;

  OpGen(std::string opName = "empty", OpGenerator generator = emptyGenerator())
      : opName(opName), generator(generator) {}

  Operation *apply(OpBuilder &builder, Location loc, OpRegion &region) {
    auto op = generator(builder, loc, region);
    return op;
  }

  bool isTerminatorOf(OpRegion opRegion) {
    return terminatorMap.find(opName) != terminatorMap.end() &&
           std::find(terminatorMap[opName].begin(), terminatorMap[opName].end(),
                     opRegion.parent_op) != terminatorMap[opName].end();
  }
};

// global(module) function pool
inline std::vector<func::FuncOp> funcPool;

OpGenerator initGenerator();

/*-------------- bufferization generators -------------*/
OpGenerator bufferizationAllocTensorGenerator();
OpGenerator bufferizationCloneGenerator();
OpGenerator bufferizationDeallocGenerator();
OpGenerator bufferizationToMemrefGenerator();
OpGenerator bufferizationToTensorGenerator();

/*---------------- linalg operation generators ---------------*/
OpGenerator linalgGenericGenerator();
// OpGenerator randomLinalgElementwiseGenerator();
OpGenerator linalgMapGenerator();
OpGenerator linalgReduceGenerator();
OpGenerator linalgMatMulGenerator();
OpGenerator linalgTransposeGenerator();
OpGenerator linalgBroadCastGenerator();
OpGenerator linalgCopyGenerator();
OpGenerator linalgVecmatGenerator();
OpGenerator linalgDotGenerator();
OpGenerator linalgBatchMatmulGenerator();
OpGenerator linalgBatchMatmulTransposeBGenerator();
OpGenerator linalgBatchMatvecGenerator();
OpGenerator linalgBatchReduceMatmulGenerator();
OpGenerator linalgConv1dNcwFcwGenerator();
OpGenerator linalgConv1dNwcWcfGenerator();
OpGenerator linalgConv1dGenerator();
OpGenerator linalgConv2dNchwFchwGenerator();
OpGenerator linalgConv2dNgchwFgchwGenerator();
OpGenerator linalgConv2dBhwcFhwcGenerator();
OpGenerator linalgConv2dBhwcHwcfGenerator();
OpGenerator linalgConv2dBhwcHwcfQGenerator();
OpGenerator linalgConv2dGenerator();
OpGenerator linalgConv3dNdhwcDhwcfGenerator();
OpGenerator linalgConv3dNdhwcDhwcfQGenerator();
OpGenerator linalgConv3dGenerator();
OpGenerator linalgDepthwiseConv1dNwcWcGenerator();
OpGenerator linalgDepthwiseConv1dNwcWcmGenerator();
OpGenerator linalgDepthwiseConv2dNchwChwGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcQGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcmGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcmQGenerator();
OpGenerator linalgDepthwiseConv3dNdhwcDhwcGenerator();
OpGenerator linalgDepthwiseConv3dNdhwcDhwcmGenerator();
OpGenerator linalgElemwiseBinaryGenerator();
OpGenerator linalgElemwiseUnaryGenerator();
OpGenerator linalgFillGenerator();
OpGenerator linalgFillRng2dGenerator();
OpGenerator linalgIndexGenerator();
OpGenerator linalgMatmulTransposeBGenerator();
OpGenerator linalgMatmulUnsignedGenerator();
OpGenerator linalgMatvecGenerator();
OpGenerator linalgMmt4dGenerator();
OpGenerator linalgPoolingNchwMaxGenerator();
OpGenerator linalgPoolingNchwSumGenerator();
OpGenerator linalgPoolingNcwMaxGenerator();
OpGenerator linalgPoolingNcwSumGenerator();
OpGenerator linalgPoolingNdhwcMaxGenerator();
OpGenerator linalgPoolingNdhwcMinGenerator();
OpGenerator linalgPoolingNdhwcSumGenerator();
OpGenerator linalgPoolingNhwcMaxGenerator();
OpGenerator linalgPoolingNhwcMaxUnsignedGenerator();
OpGenerator linalgPoolingNhwcMinGenerator();
OpGenerator linalgPoolingNhwcMinUnsignedGenerator();
OpGenerator linalgPoolingNhwcSumGenerator();
OpGenerator linalgPoolingNwcMaxGenerator();
OpGenerator linalgPoolingNwcMaxUnsignedGenerator();
OpGenerator linalgPoolingNwcMinGenerator();
OpGenerator linalgPoolingNwcMinUnsignedGenerator();
OpGenerator linalgPoolingNwcSumGenerator();
OpGenerator linalgQuantizedBatchMatmulGenerator();
OpGenerator linalgQuantizedMatmulGenerator();

/*---------------- memref operation generators -------------------*/
OpGenerator assumeAlignmentGenerator();
OpGenerator reallocGenerator();
OpGenerator allocGenerator();
OpGenerator tensorStoreGenerator();
OpGenerator memrefLoadGenerator();
OpGenerator memrefStoreGenerator();
OpGenerator atomicRMWGenerator();
OpGenerator genericAtomicRMWGenerator();
OpGenerator memrefCopyGenerator();
OpGenerator allocaGenerator();
OpGenerator allocaScopeGenerator();
OpGenerator memrefCastGenerator();
OpGenerator subViewGenerator();
OpGenerator transposeGenerator();

/*---------------- arith operation generators -------------------*/
OpGenerator addFGenerator();
OpGenerator addIGenerator();
OpGenerator andIGenerator();
OpGenerator ceilDivSIGenerator();
OpGenerator ceilDivUIGenerator();
OpGenerator cmpFGenerator();
OpGenerator cmpIGenerator();
OpGenerator constantGenerator();
OpGenerator divFGenerator();
OpGenerator divSIGenerator();
OpGenerator divUIGenerator();
OpGenerator floorDivSIGenerator();
OpGenerator minUIGenerator();
OpGenerator minSIGenerator();
OpGenerator mulFGenerator();
OpGenerator mulIGenerator();
OpGenerator negFGenerator();
OpGenerator orIGenerator();
OpGenerator remFGenerator();
OpGenerator remSIGenerator();
OpGenerator remUIGenerator();
OpGenerator shlIGenerator();
OpGenerator shrSIGenerator();
OpGenerator shrUIGenerator();
OpGenerator subFGenerator();
OpGenerator subIGenerator();
OpGenerator xorIGenerator();

/*---------------- math operation generators -------------------*/
OpGenerator absFGenerator();     // f u
OpGenerator absIGenerator();     // i u
OpGenerator atanGenerator();     // float unary
OpGenerator atan2Generator();    // float binary
OpGenerator ceilGenerator();     // f u
OpGenerator copySignGenerator(); // f b
OpGenerator cosGenerator();      // f u
OpGenerator sinGenerator();
OpGenerator ctlzGenerator(); // i u
OpGenerator cttzGenerator();
OpGenerator ctpopGenerator();
OpGenerator erfGenerator(); // f u // What's the semantic of this op?
OpGenerator expGenerator();
OpGenerator exp2Generator();
OpGenerator expm1Generator();
OpGenerator floorGenerator();
OpGenerator fmaGenerator();   // f t
OpGenerator ipowiGenerator(); // i b
OpGenerator logGenerator();   // f u
OpGenerator log10Generator();
OpGenerator log1pGenerator();
OpGenerator log2Generator();
OpGenerator powfGenerator();  // f b
OpGenerator rsqrtGenerator(); // f u
OpGenerator sqrtGenerator();
OpGenerator tanGenerator();
OpGenerator tanhGenerator();
OpGenerator roundEvenGenerator();
OpGenerator roundGenerator();
OpGenerator truncGenerator();
OpGenerator fpowiGenerator(); // f,i

/*------------ func generators -----------*/
OpGenerator funcGenerator();
OpGenerator callGenerator();
// OpGenerator returnGenerator();

/*-------------- scf generators -------------------*/
OpGenerator scfIfGenerator();
OpGenerator executeRegionGenerator();
OpGenerator scfForGenerator();
OpGenerator indexSwitchGenerator();
OpGenerator scfWhileGenerator();
OpGenerator scfForEachThreadGenerator(); // deprecated
OpGenerator scfParallelGenerator();

/*-------------- affine generators ------------*/
OpGenerator affineApplyGenerator();
OpGenerator affineForGenerator();
OpGenerator affineIfGenerator();
OpGenerator affineLoadGenerator();
OpGenerator affineStoreGenerator();
OpGenerator affineMinGenerator();
OpGenerator affineMaxGenerator();
OpGenerator affineParallelGenerator(); // error: 'affine.parallel' op a
// reduction must be specified for each output
// OpGenerator affinePrefetchGenerator();
OpGenerator affineVectorLoadGenerator();
OpGenerator affineVectorStoreGenerator();
// OpGenerator affineDMAStartGenerator();
// OpGenerator affineDMAWaitGenerator();

/*-------------- vector generators -------------*/
OpGenerator vectorBroadcastGenerator();
OpGenerator vectorBitCastGenerator();
OpGenerator vectorCompressStoreGenerator();
OpGenerator vectorConstantMaskGenerator();
OpGenerator vectorCreateMaskGenerator();
OpGenerator vectorContractGenerator();
OpGenerator vectorExpandLoadGenerator();
OpGenerator vectorExtractGenerator();
OpGenerator vectorExtractElementGenerator();
OpGenerator vectorExtractStridedSliceGenerator();
OpGenerator vectorFMAGenerator();
OpGenerator vectorFlatTransposeGenerator();
OpGenerator vectorGatherGenerator();
OpGenerator vectorInsertElementGenerator();
OpGenerator vectorInsertGenerator();
OpGenerator vectorInsertStridedSliceGenerator();
OpGenerator vectorLoadGenerator();
OpGenerator vectorMaskGenerator();
OpGenerator vectorMaskedLoadGenerator();
OpGenerator vectorStoreGenerator();
OpGenerator vectorMaskedStoreGenerator();
OpGenerator vectorMatrixMultiplyGenerator();
OpGenerator vectorMultiReductionGenerator();
OpGenerator vectorOuterProductGenerator();
OpGenerator vectorPrintGenerator();
OpGenerator vectorReductionGenerator();
OpGenerator vectorScanGenerator();
OpGenerator vectorScatterGenerator();
OpGenerator vectorShuffleGenerator();
OpGenerator vectorSplatGenerator();
OpGenerator vectorTransposeGenerator();
OpGenerator vectorTransferReadGenerator();
OpGenerator vectorTransferWriteGenerator();
OpGenerator vectorWarpExecuteOnLane0Op();
OpGenerator vectorReshapeGenerator(); // TODO
OpGenerator vectorShapeCastGenerator();
OpGenerator vectorTypeCastGenerator();

/*-------------- tensor generators -------------*/
OpGenerator tensorCastGenerator();
OpGenerator tensorCollapseShapeGenerator();
OpGenerator tensorDimGenerator();
OpGenerator tensorEmptyGenerator();
OpGenerator tensorExpandShapeGenerator();
OpGenerator tensorExtractGenerator();
OpGenerator tensorExtractSliceGenerator();
OpGenerator tensorFromElementsGenerator();
OpGenerator tensorGatherGenerator();
OpGenerator tensorGenerateGenerator();
OpGenerator tensorInsertGenerator();
OpGenerator tensorInsertSliceGenerator();
// OpGenerator tensorPackGenerator();
OpGenerator tensorPadGenerator();
OpGenerator tensorParallelInsertSliceGenerator();
OpGenerator tensorRankGenerator();
OpGenerator tensorReshapeGenerator();
OpGenerator tensorScatterGenerator();
OpGenerator tensorSplatGenerator();
OpGenerator tensorUnpackGenerator();

/*-------------- index generators -------------*/
OpGenerator indexAddGenerator();
OpGenerator indexAndGenerator();
OpGenerator indexBoolConstantGenerator();
OpGenerator indexCastSGenerator();
OpGenerator indexCastUGenerator();
OpGenerator indexCeilDivSGenerator();
OpGenerator indexCeilDivUGenerator();
OpGenerator indexCmpGenerator();
OpGenerator indexConstantGenerator();
OpGenerator indexDivSGenerator();
OpGenerator indexDivUGenerator();
OpGenerator indexFloorDivSGenerator();
OpGenerator indexMaxSGenerator();
OpGenerator indexMaxUGenerator();
OpGenerator indexMulGenerator();
OpGenerator indexOrGenerator();
OpGenerator indexRemSGenerator();
OpGenerator indexRemUGenerator();
OpGenerator indexShLGenerator();
OpGenerator indexShrSGenerator();
OpGenerator indexShrUGenerator();
OpGenerator indexSizeOfGenerator();
OpGenerator indexSubGenerator();
OpGenerator indexXorGenerator();

/*-------------- SPIRV generators -------------*/
void registerSPIRVGenerators();
OpGen spirvBitCountGenerator();
OpGen spirvBitReverseGenerator();
OpGen spirvFNegateGenerator();
OpGen spirvIsInfGenerator();
OpGen spirvIsNanGenerator();
OpGen spirvLogicalNotGenerator();
OpGen spirvNotGenerator();
OpGen spirvBitFieldInsertGenerator();
OpGen spirvBitFieldSExtractGenerator();
OpGen spirvBitFieldUExtractGenerator();
OpGen spirvBitwiseAndGenerator();
OpGen spirvBitwiseOrGenerator();
OpGen spirvBitwiseXorGenerator();
OpGen spirvCLCeilGenerator();
OpGen spirvCLCosGenerator();
OpGen spirvCLErfGenerator();
OpGen spirvCLExpGenerator();
OpGen spirvCLFAbsGenerator();
OpGen spirvCLFloorGenerator();
OpGen spirvCLLogGenerator();
OpGen spirvCLRintGenerator();
OpGen spirvCLRoundGenerator();
OpGen spirvCLRsqrtGenerator();
OpGen spirvCLSinGenerator();
OpGen spirvCLSqrtGenerator();
OpGen spirvCLTanhGenerator();
OpGen spirvBitwiseOrGenerator();
OpGen spirvBitwiseXorGenerator();
OpGen spirvCLCeilGenerator();
OpGen spirvCLCosGenerator();
OpGen spirvCLErfGenerator();
OpGen spirvCLExpGenerator();
OpGen spirvCLFAbsGenerator();
OpGen spirvCLFloorGenerator();
OpGen spirvCLLogGenerator();
OpGen spirvCLRintGenerator();
OpGen spirvCLRoundGenerator();
OpGen spirvCLRsqrtGenerator();
OpGen spirvCLSinGenerator();
OpGen spirvCLSqrtGenerator();
OpGen spirvCLTanhGenerator();
OpGen spirvFOrdEqualGenerator();
OpGen spirvFOrdGreaterThanEqualGenerator();
OpGen spirvFOrdGreaterThanGenerator();
OpGen spirvFOrdLessThanEqualGenerator();
OpGen spirvFOrdLessThanGenerator();
OpGen spirvFOrdNotEqualGenerator();
OpGen spirvFUnordEqualGenerator();
OpGen spirvFUnordGreaterThanEqualGenerator();
OpGen spirvFUnordGreaterThanGenerator();
OpGen spirvFUnordLessThanEqualGenerator();
OpGen spirvFUnordLessThanGenerator();
OpGen spirvFUnordNotEqualGenerator();
OpGen spirvIEqualGenerator();
OpGen spirvINotEqualGenerator();
OpGen spirvLogicalAndGenerator();
OpGen spirvLogicalOrGenerator();
OpGen spirvLogicalEqualGenerator();
OpGen spirvLogicalNotEqualGenerator();
OpGen spirvSGreaterThanEqualGenerator();
OpGen spirvSGreaterThanGenerator();
OpGen spirvSLessThanEqualGenerator();
OpGen spirvSLessThanGenerator();
OpGen spirvUGreaterThanEqualGenerator();
OpGen spirvUGreaterThanGenerator();
OpGen spirvULessThanEqualGenerator();
OpGen spirvULessThanGenerator();
OpGen spirvUnorderedGenerator();
OpGen spirvGLAcosGenerator();
OpGen spirvGLAsinGenerator();
OpGen spirvGLAtanGenerator();
OpGen spirvGLCeilGenerator();
OpGen spirvGLCosGenerator();
OpGen spirvGLCoshGenerator();
OpGen spirvGLExpGenerator();
OpGen spirvGLFAbsGenerator();
OpGen spirvGLFSignGenerator();
OpGen spirvGLFloorGenerator();
OpGen spirvGLInverseSqrtGenerator();
OpGen spirvGLLogGenerator();
OpGen spirvGLRoundEvenGenerator();
OpGen spirvGLRoundGenerator();
OpGen spirvGLSinGenerator();
OpGen spirvGLSinhGenerator();
OpGen spirvGLSqrtGenerator();
OpGen spirvGLTanGenerator();
OpGen spirvGLTanhGenerator();
OpGen spirvGLFClampGenerator();
OpGen spirvGLFMaxGenerator();
OpGen spirvGLFMinGenerator();
OpGen spirvGLFMixGenerator();
OpGen spirvGLFindUMsbGenerator();
OpGen spirvGLFmaGenerator();
OpGen spirvGLLdexpGenerator();
OpGen spirvGLPowGenerator();
OpGen spirvGLSAbsGenerator();
OpGen spirvGLSClampGenerator();
OpGen spirvGLSMaxGenerator();
OpGen spirvGLSMinGenerator();
OpGen spirvGLSSignGenerator();
OpGen spirvGLUClampGenerator();
OpGen spirvGLUMaxGenerator();
OpGen spirvGLUMinGenerator();
OpGen spirvCLFMaxGenerator();
OpGen spirvCLFMinGenerator();
OpGen spirvCLFmaGenerator();
OpGen spirvCLPowGenerator();
OpGen spirvCLSAbsGenerator();
OpGen spirvCLSMaxGenerator();
OpGen spirvCLSMinGenerator();
OpGen spirvCLUMaxGenerator();
OpGen spirvCLUMinGenerator();
OpGen spirvAccessChainGenerator();
OpGen spirvAddressOfGenerator();
OpGen spirvAtomicAndGenerator();
OpGen spirvAtomicCompareExchangeGenerator();
OpGen spirvAtomicCompareExchangeWeakGenerator();
OpGen spirvAtomicExchangeGenerator();
OpGen spirvAtomicIAddGenerator();
OpGen spirvAtomicIDecrementGenerator();
OpGen spirvAtomicIIncrementGenerator();
OpGen spirvAtomicISubGenerator();
OpGen spirvAtomicOrGenerator();
OpGen spirvAtomicSMaxGenerator();
OpGen spirvAtomicSMinGenerator();
OpGen spirvAtomicUMaxGenerator();
OpGen spirvAtomicUMinGenerator();
OpGen spirvAtomicXorGenerator();
OpGen spirvBitcastGenerator();
OpGen spirvBranchConditionalGenerator();
OpGen spirvBranchGenerator();
OpGen spirvCLPrintfGenerator();
OpGen spirvCompositeConstructGenerator();
OpGen spirvCompositeExtractGenerator();
OpGen spirvCompositeInsertGenerator();
OpGen spirvConstantGenerator();
OpGen spirvControlBarrierGenerator();
OpGen spirvConvertFToSGenerator();
OpGen spirvConvertFToUGenerator();
OpGen spirvConvertPtrToUGenerator();
OpGen spirvConvertSToFGenerator();
OpGen spirvConvertUToFGenerator();
OpGen spirvConvertUToPtrGenerator();
OpGen spirvCopyMemoryGenerator();
OpGen spirvEXTAtomicFAddGenerator();
OpGen spirvEntryPointGenerator();
OpGen spirvExecutionModeGenerator();
OpGen spirvFAddGenerator();
OpGen spirvFConvertGenerator();
OpGen spirvFDivGenerator();
OpGen spirvFModGenerator();
OpGen spirvFMulGenerator();
OpGen spirvFRemGenerator();
OpGen spirvFSubGenerator();
OpGen spirvFuncGenerator();
OpGen spirvFunctionCallGenerator();
OpGen spirvGLFrexpStructGenerator();
OpGen spirvGenericCastToPtrExplicitGenerator();
OpGen spirvGenericCastToPtrGenerator();
OpGen spirvGlobalVariableGenerator();
OpGen spirvGroupBroadcastGenerator();
OpGen spirvGroupFAddGenerator();
OpGen spirvGroupFMaxGenerator();
OpGen spirvGroupFMinGenerator();
OpGen spirvGroupFMulKHRGenerator();
OpGen spirvGroupIAddGenerator();
OpGen spirvGroupIMulKHRGenerator();
OpGen spirvGroupNonUniformBallotGenerator();
OpGen spirvGroupNonUniformBroadcastGenerator();
OpGen spirvGroupNonUniformElectGenerator();
OpGen spirvGroupNonUniformFAddGenerator();
OpGen spirvGroupNonUniformFMaxGenerator();
OpGen spirvGroupNonUniformFMinGenerator();
OpGen spirvGroupNonUniformFMulGenerator();
OpGen spirvGroupNonUniformIAddGenerator();
OpGen spirvGroupNonUniformIMulGenerator();
OpGen spirvGroupNonUniformSMaxGenerator();
OpGen spirvGroupNonUniformSMinGenerator();
OpGen spirvGroupNonUniformShuffleDownGenerator();
OpGen spirvGroupNonUniformShuffleGenerator();
OpGen spirvGroupNonUniformShuffleUpGenerator();
OpGen spirvGroupNonUniformShuffleXorGenerator();
OpGen spirvGroupNonUniformUMaxGenerator();
OpGen spirvGroupNonUniformUMinGenerator();
OpGen spirvGroupSMaxGenerator();
OpGen spirvGroupSMinGenerator();
OpGen spirvGroupUMaxGenerator();
OpGen spirvGroupUMinGenerator();
OpGen spirvIAddCarryGenerator();
OpGen spirvIAddGenerator();
OpGen spirvIMulGenerator();
OpGen spirvINTELConvertBF16ToFGenerator();
OpGen spirvINTELConvertFToBF16Generator();
OpGen spirvINTELJointMatrixLoadGenerator();
OpGen spirvINTELJointMatrixMadGenerator();
OpGen spirvINTELJointMatrixStoreGenerator();
OpGen spirvINTELJointMatrixWorkItemLengthGenerator();
OpGen spirvINTELSubgroupBlockReadGenerator();
OpGen spirvINTELSubgroupBlockWriteGenerator();
OpGen spirvISubBorrowGenerator();
OpGen spirvISubGenerator();
OpGen spirvImageDrefGatherGenerator();
OpGen spirvImageGenerator();
OpGen spirvImageQuerySizeGenerator();
OpGen spirvInBoundsPtrAccessChainGenerator();
OpGen spirvKHRAssumeTrueGenerator();
OpGen spirvKHRCooperativeMatrixLengthGenerator();
OpGen spirvKHRCooperativeMatrixLoadGenerator();
OpGen spirvKHRCooperativeMatrixStoreGenerator();
OpGen spirvKHRSubgroupBallotGenerator();
OpGen spirvLoadGenerator();
OpGen spirvLoopGenerator();
OpGen spirvMatrixTimesMatrixGenerator();
OpGen spirvMatrixTimesScalarGenerator();
OpGen spirvMemoryBarrierGenerator();
OpGen spirvMergeGenerator();
OpGen spirvModuleGenerator();
OpGen spirvNVCooperativeMatrixLengthGenerator();
OpGen spirvNVCooperativeMatrixLoadGenerator();
OpGen spirvNVCooperativeMatrixMulAddGenerator();
OpGen spirvNVCooperativeMatrixStoreGenerator();
OpGen spirvOrderedGenerator();
OpGen spirvPtrAccessChainGenerator();
OpGen spirvPtrCastToGenericGenerator();
OpGen spirvReferenceOfGenerator();
OpGen spirvReturnGenerator();
OpGen spirvReturnValueGenerator();
OpGen spirvSConvertGenerator();
OpGen spirvSDivGenerator();
OpGen spirvSDotAccSatGenerator();
OpGen spirvSDotGenerator();
OpGen spirvSModGenerator();
OpGen spirvSMulExtendedGenerator();
OpGen spirvSNegateGenerator();
OpGen spirvSRemGenerator();
OpGen spirvSUDotAccSatGenerator();
OpGen spirvSUDotGenerator();
OpGen spirvSelectGenerator();
OpGen spirvSelectionGenerator();
OpGen spirvShiftLeftLogicalGenerator();
OpGen spirvShiftRightArithmeticGenerator();
OpGen spirvShiftRightLogicalGenerator();
OpGen spirvSpecConstantCompositeGenerator();
OpGen spirvSpecConstantGenerator();
OpGen spirvSpecConstantOperationGenerator();
OpGen spirvStoreGenerator();
OpGen spirvTransposeGenerator();
OpGen spirvUConvertGenerator();
OpGen spirvUDivGenerator();
OpGen spirvUDotAccSatGenerator();
OpGen spirvUDotGenerator();
OpGen spirvUModGenerator();
OpGen spirvUMulExtendedGenerator();
OpGen spirvUndefGenerator();
OpGen spirvUnreachableGenerator();
OpGen spirvVariableGenerator();
OpGen spirvVectorExtractDynamicGenerator();
OpGen spirvVectorInsertDynamicGenerator();
OpGen spirvVectorShuffleGenerator();
OpGen spirvVectorTimesScalarGenerator();
OpGen spirvYieldGenerator();

// generator registration
inline std::map<std::string, OpGen> operators = {
    {"func.func", OpGen("func.func", funcGenerator())},
    //    {"func.call", OpGen("func.call", callGenerator())}, //TODO
    {"linalg.matmul", OpGen("linalg.matmul", linalgMatMulGenerator())},
    {"linalg.generic", OpGen("linalg.generic", linalgGenericGenerator())},
    {"linalg.map", OpGen("linalg.map", linalgMapGenerator())},
    {"linalg.copy", OpGen("linalg.copy", linalgCopyGenerator())},
    {"linalg.transpose", OpGen("linalg.transpose", linalgTransposeGenerator())},
    {"linalg.broadcast", OpGen("linalg.broadcast", linalgBroadCastGenerator())},
    {"linalg.reduce", OpGen("linalg.reduce", linalgReduceGenerator())},
    {"linalg.dot", OpGen("linalg.dot", linalgDotGenerator())},
    {"memref.alloca", OpGen("memref.alloca", allocaGenerator())},
    {"memref.alloca_scope",
     OpGen("memref.alloca_scope", allocaScopeGenerator())},
    {"memref.assume_alignment",
     OpGen("memref.assume_alignment", assumeAlignmentGenerator())},
    {"memref.alloc", OpGen("memref.alloc", allocGenerator())},
    {"memref.atomic_rmw", OpGen("memref.atomic_rmw", atomicRMWGenerator())},
    {"memref.cast", OpGen("memref.cast", memrefCastGenerator())},
    {"memref.realloc", OpGen("memref.realloc", reallocGenerator())},
    {"memref.tensor_store",
     OpGen("memref.tensor_store", tensorStoreGenerator())},
    {"memref.store", OpGen("memref.store", memrefStoreGenerator())},
    {"memref.load", OpGen("memref.load", memrefLoadGenerator())},
    {"memref.copy", OpGen("memref.copy", memrefCopyGenerator())},
    {"arith.addf", OpGen("arith.addf", addFGenerator())},
    {"arith.addi", OpGen("arith.addi", addIGenerator())},
    {"arith.andi", OpGen("arith.andi", andIGenerator())},
    {"arith.cmpf", OpGen("arith.cmpf", cmpFGenerator())},
    {"arith.cmpi", OpGen("arith.cmpi", cmpIGenerator())},
    {"arith.ceildivsi", OpGen("arith.ceildivsi", ceilDivSIGenerator())},
    {"arith.constant", OpGen("arith.constant", constantGenerator())},
    {"arith.divf", OpGen("arith.divf", divFGenerator())},
    {"arith.divsi", OpGen("arith.divsi", divSIGenerator())},
    {"arith.divui", OpGen("arith.divui", divUIGenerator())},
    {"arith.floordivsi", OpGen("arith.floordivsi", floorDivSIGenerator())},
    {"arith.minsi", OpGen("arith.minsi", minSIGenerator())},
    {"arith.minui", OpGen("arith.minui", minUIGenerator())},
    {"arith.mulf", OpGen("arith.mulf", mulFGenerator())},
    {"arith.muli", OpGen("arith.muli", mulIGenerator())},
    {"arith.negf", OpGen("arith.negf", negFGenerator())},
    {"arith.ori", OpGen("arith.ori", orIGenerator())},
    {"arith.remf", OpGen("arith.remf", remFGenerator())},
    {"arith.remsi", OpGen("arith.remsi", remSIGenerator())},
    {"arith.remui", OpGen("arith.remui", remUIGenerator())},
    {"arith.shli", OpGen("arith.shli", shlIGenerator())},
    {"arith.shrsi", OpGen("arith.shrsi", shrSIGenerator())},
    {"arith.shrui", OpGen("arith.shrui", shrUIGenerator())},
    {"arith.subf", OpGen("arith.subf", subFGenerator())},
    {"arith.subi", OpGen("arith.subi", subIGenerator())},
    {"arith.xori", OpGen("arith.xori", xorIGenerator())},
    {"math.absf", OpGen("math.absf", absFGenerator())},
    {"math.absi", OpGen("math.absi", absIGenerator())},
    {"math.atan", OpGen("math.atan", atanGenerator())},
    {"math.atan2", OpGen("math.atan2", atan2Generator())},
    {"math.ceil", OpGen("math.ceil", ceilGenerator())},
    {"math.copySign", OpGen("math.copySign", copySignGenerator())},
    {"math.cos", OpGen("math.cos", cosGenerator())},
    {"math.ctlz", OpGen("math.ctlz", ctlzGenerator())},
    {"math.cttz", OpGen("math.cttz", cttzGenerator())},
    {"math.ctpop", OpGen("math.ctpop", ctpopGenerator())},
    {"math.exp", OpGen("math.exp", expGenerator())},
    {"math.exp2", OpGen("math.exp2", exp2Generator())},
    {"math.expm1", OpGen("math.expm1", expm1Generator())},
    {"math.floor", OpGen("math.floor", floorGenerator())},
    {"math.fma", OpGen("math.fma", fmaGenerator())},
    {"math.ipowi", OpGen("math.ipowi", ipowiGenerator())},
    {"math.log", OpGen("math.log", logGenerator())},
    {"math.log10", OpGen("math.log10", log10Generator())},
    {"math.log1p", OpGen("math.log1p", log1pGenerator())},
    {"math.log2", OpGen("math.log2", log2Generator())},
    {"math.powf", OpGen("math.powf", powfGenerator())},
    {"math.rsqrt", OpGen("math.rsqrt", rsqrtGenerator())},
    {"math.sqrt", OpGen("math.sqrt", sqrtGenerator())},
    {"math.tan", OpGen("math.tan", tanGenerator())},
    {"math.tanh", OpGen("math.tanh", tanhGenerator())},
    {"math.roundeven", OpGen("math.roundeven", roundEvenGenerator())},
    {"math.round", OpGen("math.round", roundGenerator())},
    {"math.trunc", OpGen("math.trunc", truncGenerator())},
    {"math.fpowi", OpGen("math.fpowi", fpowiGenerator())},
    {"scf.if", OpGen("scf.if", scfIfGenerator())},
    {"scf.execute_region",
     OpGen("scf.execute_region", executeRegionGenerator())},
    {"scf.index_switch", OpGen("scf.index_switch", indexSwitchGenerator())},
    {"scf.while", OpGen("scf.while", scfWhileGenerator())},
    {"scf.parallel", OpGen("scf.parallel", scfParallelGenerator())},
    {"affine.apply", OpGen("affine.apply", affineApplyGenerator())},
    {"affine.for", OpGen("affine.for", affineForGenerator())},
    {"affine.if", OpGen("affine.if", affineIfGenerator())},
    {"affine.load", OpGen("affine.load", affineLoadGenerator())},
    {"affine.store", OpGen("affine.store", affineStoreGenerator())},
    {"affine.min", OpGen("affine.min", affineMinGenerator())},
    {"affine.max", OpGen("affine.max", affineMaxGenerator())},
    {"affine.vector_store",
     OpGen("affine.vector_store", affineVectorStoreGenerator())},
    {"affine.vector_load",
     OpGen("affine.vector_load", affineVectorLoadGenerator())},
    // {"affine.parallel", OpGen("affine.parallel", affineParallelGenerator())},
    {"vector.bitcast", OpGen("vector.bitcast", vectorBitCastGenerator())},
    {"vector.broadcast", OpGen("vector.broadcast", vectorBroadcastGenerator())},
    {"vector.contract", OpGen("vector.contract", vectorContractGenerator())},
    {"vector.compress_store",
     OpGen("vector.compress_store", vectorCompressStoreGenerator())},
    {"vector.create_mask",
     OpGen("vector.create_mask", vectorCreateMaskGenerator())},
    {"vector.extract", OpGen("vector.extract", vectorExtractGenerator())},
    {"vector.extract_strided_slice",
     OpGen("vector.extract_strided_slice",
           vectorExtractStridedSliceGenerator())},
    {"vector.fma", OpGen("vector.fma", vectorFMAGenerator())},
    {"vector.flat_transpose",
     OpGen("vector.flat_transpose", vectorFlatTransposeGenerator())},
    {"vector.insert_element",
     OpGen("vector.insert_element", vectorInsertElementGenerator())},
    {"vector.insert", OpGen("vector.insert", vectorInsertGenerator())},
    {"vector.insert_strided_slice",
     OpGen("vector.insert_strided_slice", vectorInsertStridedSliceGenerator())},
    {"vector.load", OpGen("vector.load", vectorLoadGenerator())},
    {"vector.masked_load",
     OpGen("vector.masked_load", vectorMaskedLoadGenerator())},
    {"vector.matrix_multiply",
     OpGen("vector.matrix_multiply", vectorMatrixMultiplyGenerator())},
    {"vector.multi_reduction",
     OpGen("vector.multi_reduction", vectorMultiReductionGenerator())},
    {"vector.outer_product",
     OpGen("vector.outer_product", vectorOuterProductGenerator())},
    {"vector.print", OpGen("vector.print", vectorPrintGenerator())},
    {"vector.reduction", OpGen("vector.reduction", vectorReductionGenerator())},
    {"vector.scan", OpGen("vector.scan", vectorScanGenerator())},
    {"vector.scatter", OpGen("vector.scatter", vectorScatterGenerator())},
    {"vector.shuffle", OpGen("vector.shuffle", vectorShuffleGenerator())},
    {"vector.splat", OpGen("vector.splat", vectorSplatGenerator())},
    {"vector.transpose", OpGen("vector.transpose", vectorTransposeGenerator())},
    {"vector.gather", OpGen("vector.gather", vectorGatherGenerator())},
    {"vector.mask", OpGen("vector.mask", vectorMaskGenerator())},
    {"vector.transfer_read",
     OpGen("vector.transfer_read", vectorTransferReadGenerator())},
    {"vector.transfer_write",
     OpGen("vector.transfer_write", vectorTransferWriteGenerator())},
    {"vector.warp_execute_on_lane0",
     OpGen("vector.warp_execute_on_lane0", vectorWarpExecuteOnLane0Op())},
    {"tensor.cast", OpGen("tensor.cast", tensorCastGenerator())},
    {"tensor.collapse_shape",
     OpGen("tensor.collapse_shape", tensorCollapseShapeGenerator())},
    {"tensor.dim", OpGen("tensor.dim", tensorDimGenerator())},
    {"tensor.empty", OpGen("tensor.empty", tensorEmptyGenerator())},
    {"tensor.expand_shape",
     OpGen("tensor.expand_shape", tensorExpandShapeGenerator())},
    {"tensor.extract", OpGen("tensor.extract", tensorExtractGenerator())},
    {"tensor.insert_slice",
     OpGen("tensor.insert_slice", tensorInsertSliceGenerator())},
    {"tensor.extract_slice",
     OpGen("tensor.extract_slice", tensorExtractSliceGenerator())},
    {"tensor.from_elements",
     OpGen("tensor.from_elements", tensorFromElementsGenerator())},
    {"tensor.generate", OpGen("tensor.generate", tensorGenerateGenerator())},
    {"tensor.insert", OpGen("tensor.insert", tensorInsertGenerator())},
    //    {"tensor.pack", OpGen("tensor.pack", tensorPackGenerator())},
    {"tensor.rank", OpGen("tensor.rank", tensorRankGenerator())},
    {"tensor.scatter", OpGen("tensor.scatter", tensorScatterGenerator())},
    {"tensor.splat", OpGen("tensor.splat", tensorSplatGenerator())},
    {"tensor.unpack", OpGen("tensor.unpack", tensorUnpackGenerator())},
    {"index.add", OpGen("index.add", indexAddGenerator())},
    {"index.and", OpGen("index.and", indexAndGenerator())},
    {"index.bool.constant",
     OpGen("index.bool.constant", indexBoolConstantGenerator())},
    {"index.casts", OpGen("index.casts", indexCastSGenerator())},
    {"index.castu", OpGen("index.castu", indexCastUGenerator())},
    {"index.ceildivs", OpGen("index.ceildivs", indexCeilDivSGenerator())},
    {"index.ceildivu", OpGen("index.ceildivu", indexCeilDivUGenerator())},
    {"index.constant", OpGen("index.constant", indexConstantGenerator())},
    {"index.divs", OpGen("index.divs", indexDivSGenerator())},
    {"index.divu", OpGen("index.divu", indexDivUGenerator())},
    {"index.floordivs", OpGen("index.floordivs", indexFloorDivSGenerator())},
    {"index.maxs", OpGen("index.maxs", indexMaxSGenerator())},
    {"index.maxu", OpGen("index.maxu", indexMaxUGenerator())},
    {"index.mul", OpGen("index.mul", indexMulGenerator())},
    {"index.or", OpGen("index.or", indexOrGenerator())},
    {"index.rems", OpGen("index.rems", indexRemSGenerator())},
    {"index.remu", OpGen("index.remu", indexRemUGenerator())},
    {"index.shl", OpGen("index.shl", indexShLGenerator())},
    {"index.shrs", OpGen("index.shrs", indexShrSGenerator())},
    {"index.shru", OpGen("index.shru", indexShrUGenerator())},
    {"index.sizeof", OpGen("index.sizeof", indexSizeOfGenerator())},
    {"index.sub", OpGen("index.sub", indexSubGenerator())},
    {"index.xor", OpGen("index.xor", indexXorGenerator())},
    {"bufferization.alloc_tensor",
     OpGen("bufferization.alloc_tensor", bufferizationCloneGenerator())},
    {"bufferization.clone",
     OpGen("bufferization.clone", bufferizationCloneGenerator())},
    {"bufferization.dealloc_tensor",
     OpGen("bufferization.dealloc_tensor", bufferizationDeallocGenerator())},
    {"bufferization.to_memref",
     OpGen("bufferization.to_memref", bufferizationToMemrefGenerator())},
    {"bufferization.to_tensor",
     OpGen("bufferization.to_tensor", bufferizationToTensorGenerator())}};

OpGen getOpGen(std::string name);

// To be deprecated
struct OpGeneration {
  std::string op;
  OpGenerator generator;

  OpGeneration(std::string op, OpGenerator opGenerator)
      : op(op), generator(opGenerator) {}

  void apply(OpBuilder &, Location, OpRegion &);
};

inline std::set<std::string> opsForFunc = {
    //    "func.call",
    "linalg.matmul", "linalg.generic", "linalg.map", "linalg.copy",
    "linalg.transpose", "linalg.broadcast", "linalg.reduce", "linalg.dot",
    "memref.alloca", "memref.alloca_scope", "memref.assume_alignment",
    "memref.alloc", "memref.atomic_rmw", "memref.cast", "memref.realloc",
    "memref.tensor_store", "memref.store", "memref.load", "memref.copy",
    "arith.addf", "arith.addi", "arith.andi", "arith.cmpf", "arith.cmpi",
    "arith.ceildivsi", "arith.constant", "arith.divf", "arith.divsi",
    "arith.divui", "arith.floordivsi", "arith.maxf", "arith.minf",
    "arith.minsi", "arith.minui", "arith.mulf", "arith.muli", "arith.negf",
    "arith.ori", "arith.remf", "arith.remsi", "arith.remui", "arith.shli",
    "arith.shrsi", "arith.shrui", "arith.subf", "arith.subi", "arith.xori",
    "math.absf", "math.absi", "math.atan", "math.atan2", "math.ceil",
    "math.copySign", "math.cos", "math.ctlz", "math.cttz", "math.ctpop",
    "math.exp", "math.exp2", "math.expm1", "math.floor", "math.fma",
    "math.ipowi", "math.log", "math.log10", "math.log1p", "math.log2",
    "math.powf", "math.rsqrt", "math.sqrt", "math.tan", "math.tanh",
    "math.roundeven", "math.round", "math.trunc", "math.fpowi", "scf.if",
    "scf.execute_region", "scf.index_switch", "scf.while", "scf.parallel",
    "affine.apply", "affine.for", "affine.if", "affine.load", "affine.store",
    "affine.min", "affine.max", 
    // "affine.parallel", 
    "affine.vector_store",
    "affine.vector_load", "vector.bitcast", "vector.broadcast",
    "vector.contract", "vector.compress_store", "vector.create_mask",
    "vector.extract", "vector.extract_strided_slice", "vector.fma",
    "vector.flat_transpose", "vector.insert_element", "vector.insert",
    "vector.insert_strided_slice", "vector.load", "vector.masked_load",
    "vector.matrix_multiply", "vector.multi_reduction", "vector.outer_product",
    "vector.print", "vector.reduction", "vector.scan", "vector.scatter",
    "vector.shuffle", "vector.splat", "vector.transpose", "vector.gather",
    "vector.mask", "vector.transfer_read", "vector.transfer_write",
    "vector.warp_execute_on_lane0", "tensor.cast", "tensor.collapse_shape",
    "tensor.dim", "tensor.empty", "tensor.expand_shape", "tensor.extract",
    "tensor.insert_slice", "tensor.extract_slice", "tensor.from_elements",
    "tensor.generate", "tensor.insert",
    //                                           "tensor.pack", //TODO
    "tensor.rank", "tensor.scatter", "tensor.splat", "tensor.unpack",
    "index.add", "index.and", "index.bool.constant", "index.casts",
    "index.castu", "index.ceildivs", "index.ceildivu", "index.constant",
    "index.divs", "index.divu", "index.floordivs", "index.maxs", "index.maxu",
    "index.mul", "index.or", "index.rems", "index.remu", "index.shl",
    "index.shrs", "index.shru", "index.sizeof", "index.sub", "index.xor",
    "bufferization.alloc_tensor", "bufferization.clone",
    "bufferization.dealloc_tensor", "bufferization.to_memref",
    "bufferization.to_tensor"};

inline std::set<std::string> intOpsForGenericAtomicRMW = {
    "arith.addi",     "arith.andi",  "arith.cmpi",  "arith.ceildivsi",
    "arith.constant", "arith.divsi", "arith.divui", "arith.floordivsi",
    "arith.muli",     "arith.ori",   "arith.remsi", "arith.remui",
    "arith.shli",     "arith.shrsi", "arith.shrui", "arith.subi",
    "arith.xori",     "math.absi",   "math.ipowi",  "math.log",
    "math.log10",     "math.fpowi"};

inline std::set<std::string> floatOpsForGenericAtomicRMW = {
    "arith.addf", "arith.cmpf",     "arith.constant", "arith.divf",
    "arith.maxf", "arith.minf",     "arith.mulf",     "arith.negf",
    "arith.remf", "arith.subf",     "math.absf",      "math.atan",
    "math.atan2", "math.ceil",      "math.copySign",  "math.cos",
    "math.ctlz",  "math.cttz",      "math.ctpop",     "math.exp",
    "math.exp2",  "math.expm1",     "math.floor",     "math.fma",
    "math.log",   "math.log10",     "math.log1p",     "math.log2",
    "math.powf",  "math.rsqrt",     "math.sqrt",      "math.tan",
    "math.tanh",  "math.roundeven", "math.round",     "math.trunc",
    "math.fpowi"};

inline std::set<std::string> maskableOps = {
    "vector.multi_reduction",
    //    "vector.reduction", // this op returns non-tensor type
    "vector.transfer_read", "vector.transfer_write"};

// Ops that have can recursively define region .
inline std::set<std::string> nonTerminatorOps = {
    "func.func",        "memref.alloca_scope",
    "scf.if",           "scf.execute_region",
    "scf.index_switch", "scf.while",
    "affine.for",       "affine.if"};

inline std::set<std::string> regionedOps = {
    "func.func",       "memref.alloca_scope",
    "scf.if",          "scf.execute_region",
    "linalg.map",      "scf.for",
    "linalg.generic",  "scf.index_switch",
    "scf.while",       "scf.parallel",
    "affine.for",      "affine.if",
    // "affine.parallel", 
    "vector.warp_execute_on_lane0",
    "tensor.generate"};

inline std::set<std::string> opsForAlloca = opsForFunc;
inline std::set<std::string> opsForScfIf = opsForFunc;
inline std::set<std::string> opsForScfFor = opsForFunc;
inline std::set<std::string> opsForScfWhile = opsForFunc;
// inline std::set<std::string> opsForScfForeachThread = opsForFunc;
inline std::set<std::string> opsForExecuteRegion = opsForFunc;
inline std::set<std::string> opsForIndexSwitch = opsForFunc;
inline std::set<std::string> opsForLinalgMap = opsForFunc;
inline std::set<std::string> opsForLinalgGeneric = opsForFunc;
inline std::set<std::string> opsForAffineFor = opsForFunc;
inline std::set<std::string> opsForAffineIf = opsForFunc;
inline std::set<std::string> opsForAffineParallel = opsForFunc; // TODO
inline std::set<std::string> opsForTensorGenerate = opsForFunc;
inline std::set<std::string> opsForVectorWarpExecuteOnLane0 = opsForFunc;
inline std::set<std::string> opsForLinalgReduce = opsForFunc;
inline std::set<std::string> opsForScfParallel = opsForFunc;
inline std::set<std::string> opsForScfReduce = opsForFunc;

//
inline std::map<std::string, std::set<std::string>> opNests = {
    {"func.func", opsForFunc},
    {"memref.alloca_scope", opsForAlloca},
    {"linalg.map", opsForLinalgMap},
    {"linalg.generic", opsForLinalgGeneric},
    {"linalg.reduce", opsForLinalgReduce},
    {"scf.if", opsForScfIf},
    {"scf.execute_region", opsForExecuteRegion},
    {"scf.for", opsForScfFor},
    {"scf.while", opsForScfWhile},
    {"scf.index_switch", opsForIndexSwitch},
    {"scf.reduce", opsForScfReduce},
    {"scf.parallel", opsForScfParallel},
    {"affine.if", opsForAffineIf},
    {"affine.for", opsForAffineFor},
    // {"affine.parallel", opsForAffineParallel},
    {"tensor.generate", opsForTensorGenerate},
    {"vector.warp_execute_on_lane0", opsForVectorWarpExecuteOnLane0}};

#endif