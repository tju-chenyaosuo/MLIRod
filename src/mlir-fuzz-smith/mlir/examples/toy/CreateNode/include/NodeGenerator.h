#ifndef NODE_GENERATOR_H
#define NODE_GENERATOR_H

#include "TypeBase.h"

using OpGenerator = std::function<bool (mlir::OpBuilder &, mlir::Location, MutationBlock *)>;

inline std::vector<arith::AtomicRMWKind> floatRmwKinds = {
    arith::AtomicRMWKind::addf, arith::AtomicRMWKind::assign,
    // arith::AtomicRMWKind::maxf, arith::AtomicRMWKind::minf,
    arith::AtomicRMWKind::mulf};

inline std::vector<arith::AtomicRMWKind> intRmwKinds = {
    arith::AtomicRMWKind::addi, arith::AtomicRMWKind::assign,
    arith::AtomicRMWKind::maxs, arith::AtomicRMWKind::maxu,
    arith::AtomicRMWKind::mins, arith::AtomicRMWKind::minu,
    arith::AtomicRMWKind::muli, arith::AtomicRMWKind::ori,
    arith::AtomicRMWKind::andi};

// Arith Dialect
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
OpGenerator maxFGenerator();
OpGenerator maxSIGenerator();
OpGenerator maxUIGenerator();
OpGenerator minFGenerator();
OpGenerator minSIGenerator();
OpGenerator minUIGenerator();
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
// Affine Dialect
OpGenerator affineApplyGenerator();
OpGenerator affineForGenerator();
OpGenerator affineIfGenerator();
OpGenerator affineLoadGenerator();
OpGenerator affineStoreGenerator();
OpGenerator affineMaxGenerator();
OpGenerator affineMinGenerator();
OpGenerator affineParallelGenerator();
OpGenerator affinePrefetchGenerator();
OpGenerator affineVectorLoadGenerator();
OpGenerator affineVectorStoreGenerator();
// Bufferization
OpGenerator bufferizationAllocTensorGenerator();
OpGenerator bufferizationCloneGenerator();
OpGenerator bufferizationDeallocGenerator();
OpGenerator bufferizationToMemrefGenerator();
OpGenerator bufferizationToTensorGenerator();
// Index
OpGenerator indexAddGenerator();
OpGenerator indexAndGenerator();
OpGenerator indexBoolConstantGenerator();
OpGenerator indexCastSGenerator();
OpGenerator indexCastUGenerator();
OpGenerator indexCeilDivSGenerator();
OpGenerator indexCeilDivUGenerator();
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
// Linalg
OpGenerator linalgGenericGenerator();
OpGenerator linalgBroadCastGenerator();
OpGenerator linalgTransposeGenerator();
OpGenerator linalgCopyGenerator();
OpGenerator linalgMapGenerator();
OpGenerator linalgReduceGenerator();
OpGenerator linalgDotGenerator();
OpGenerator linalgMatMulGenerator();
// Math 
OpGenerator absFGenerator();
OpGenerator absIGenerator();
OpGenerator atanGenerator();
OpGenerator atan2Generator();
OpGenerator ceilGenerator();
OpGenerator copySignGenerator();
OpGenerator cosGenerator();
OpGenerator sinGenerator();
OpGenerator ctlzGenerator();
OpGenerator cttzGenerator();
OpGenerator ctpopGenerator();
OpGenerator expGenerator();
OpGenerator exp2Generator();
OpGenerator expm1Generator();
OpGenerator floorGenerator();
// OpGenerator fmaGenerator();
OpGenerator ipowiGenerator();
OpGenerator logGenerator();
OpGenerator log10Generator();
OpGenerator log1pGenerator();
OpGenerator log2Generator();
OpGenerator powfGenerator();
OpGenerator rsqrtGenerator();
OpGenerator sqrtGenerator();
OpGenerator tanGenerator();
OpGenerator tanhGenerator();
OpGenerator roundEvenGenerator();
OpGenerator roundGenerator();
OpGenerator truncGenerator();
OpGenerator fpowiGenerator();
// Memref
OpGenerator memrefLoadGenerator();
OpGenerator memrefStoreGenerator();
OpGenerator atomicRMWGenerator();
OpGenerator memrefCopyGenerator();
OpGenerator assumeAlignmentGenerator();
OpGenerator allocGenerator();
OpGenerator reallocGenerator();
OpGenerator tensorStoreGenerator();
OpGenerator genericAtomicRMWGenerator();
OpGenerator allocaGenerator();
OpGenerator allocaScopeGenerator();
OpGenerator memrefCastGenerator();
// SCF
OpGenerator scfIfGenerator();
OpGenerator executeRegionGenerator();
OpGenerator scfForGenerator();
OpGenerator indexSwitchGenerator();
OpGenerator scfWhileGenerator();
OpGenerator scfParallelGenerator();
// Spirv
OpGenerator spirvBitCountGenerator();
OpGenerator spirvBitFieldInsertGenerator();
OpGenerator spirvBitFieldSExtractGenerator();
OpGenerator spirvBitFieldUExtractGenerator();
OpGenerator spirvBitReverseGenerator();
OpGenerator spirvNotGenerator();
OpGenerator spirvBitwiseAndGenerator();
OpGenerator spirvBitwiseOrGenerator();
OpGenerator spirvBitwiseXorGenerator();
OpGenerator spirvCLCeilGenerator();
OpGenerator spirvCLCosGenerator();
OpGenerator spirvCLErfGenerator();
OpGenerator spirvCLExpGenerator();
OpGenerator spirvCLFAbsGenerator();
OpGenerator spirvCLFloorGenerator();
OpGenerator spirvCLLogGenerator();
OpGenerator spirvCLRoundGenerator();
OpGenerator spirvCLRintGenerator();
OpGenerator spirvCLRsqrtGenerator();
OpGenerator spirvCLSinGenerator();
OpGenerator spirvCLSqrtGenerator();
OpGenerator spirvCLTanhGenerator();
OpGenerator spirvCLSAbsGenerator();
OpGenerator spirvCLFMaxGenerator();
OpGenerator spirvCLFMinGenerator();
OpGenerator spirvCLPowGenerator();
OpGenerator spirvCLSMaxGenerator();
OpGenerator spirvCLSMinGenerator();
OpGenerator spirvCLUMaxGenerator();
OpGenerator spirvCLUMinGenerator();
OpGenerator spirvCLFmaGenerator();
OpGenerator spirvFNegateGenerator();
OpGenerator spirvFOrdEqualGenerator();
OpGenerator spirvFOrdGreaterThanEqualGenerator();
OpGenerator spirvFOrdGreaterThanGenerator();
OpGenerator spirvFOrdLessThanEqualGenerator();
OpGenerator spirvFOrdLessThanGenerator();
OpGenerator spirvFOrdNotEqualGenerator();
OpGenerator spirvFUnordEqualGenerator();
OpGenerator spirvFUnordGreaterThanEqualGenerator();
OpGenerator spirvFUnordGreaterThanGenerator();
OpGenerator spirvFUnordLessThanEqualGenerator();
OpGenerator spirvFUnordLessThanGenerator();
OpGenerator spirvFUnordNotEqualGenerator();
OpGenerator spirvIEqualGenerator();
OpGenerator spirvINotEqualGenerator();
OpGenerator spirvLogicalEqualGenerator();
OpGenerator spirvLogicalNotGenerator();
OpGenerator spirvLogicalNotEqualGenerator();
OpGenerator spirvLogicalAndGenerator();
OpGenerator spirvLogicalOrGenerator();
OpGenerator spirvSGreaterThanEqualGenerator();
OpGenerator spirvSGreaterThanGenerator();
OpGenerator spirvSLessThanEqualGenerator();
OpGenerator spirvSLessThanGenerator();
OpGenerator spirvUGreaterThanEqualGenerator();
OpGenerator spirvUGreaterThanGenerator();
OpGenerator spirvULessThanEqualGenerator();
OpGenerator spirvULessThanGenerator();
OpGenerator spirvUnorderedGenerator();
OpGenerator spirvGLAcosGenerator();
OpGenerator spirvGLAsinGenerator();
OpGenerator spirvGLAtanGenerator();
OpGenerator spirvGLCeilGenerator();
OpGenerator spirvGLCosGenerator();
OpGenerator spirvGLCoshGenerator();
OpGenerator spirvGLExpGenerator();
OpGenerator spirvGLFAbsGenerator();
OpGenerator spirvGLFSignGenerator();
OpGenerator spirvGLFloorGenerator();
OpGenerator spirvGLInverseSqrtGenerator();
OpGenerator spirvGLLogGenerator();
OpGenerator spirvGLRoundEvenGenerator();
OpGenerator spirvGLRoundGenerator();
OpGenerator spirvGLSinGenerator();
OpGenerator spirvGLSinhGenerator();
OpGenerator spirvGLSqrtGenerator();
OpGenerator spirvGLTanGenerator();
OpGenerator spirvGLTanhGenerator();
OpGenerator spirvGLFClampGenerator();
OpGenerator spirvGLSAbsGenerator();
OpGenerator spirvGLSSignGenerator();
OpGenerator spirvGLFMaxGenerator();
OpGenerator spirvGLFMinGenerator();
OpGenerator spirvGLFMixGenerator();
OpGenerator spirvGLFindUMsbGenerator();
OpGenerator spirvGLFmaGenerator();
OpGenerator spirvGLLdexpGenerator();
OpGenerator spirvGLPowGenerator();
OpGenerator spirvGLSClampGenerator();
OpGenerator spirvGLSMaxGenerator();
OpGenerator spirvGLSMinGenerator();
OpGenerator spirvGLUMaxGenerator();
OpGenerator spirvGLUMinGenerator();
OpGenerator spirvGLUClampGenerator();
OpGenerator spirvIsInfGenerator();
OpGenerator spirvIsNanGenerator();
// tensor
OpGenerator tensorCastGenerator();
OpGenerator tensorCollapseShapeGenerator();
OpGenerator tensorDimGenerator();
OpGenerator tensorEmptyGenerator();
OpGenerator tensorExpandShapeGenerator();
OpGenerator tensorExtractGenerator();
OpGenerator tensorExtractSliceGenerator();
OpGenerator tensorFromElementsGenerator();
OpGenerator tensorGenerateGenerator();
OpGenerator tensorInsertGenerator();
OpGenerator tensorInsertSliceGenerator();
OpGenerator tensorPackGenerator();
OpGenerator tensorRankGenerator();
OpGenerator tensorScatterGenerator();
OpGenerator tensorSplatGenerator();
OpGenerator tensorUnpackGenerator();
// TOSA
OpGenerator tosaAbsGenerator();
OpGenerator tosaAddGenerator();
OpGenerator tosaApplyScaleGenerator();
OpGenerator tosaArgMaxGenerator();
OpGenerator tosaArithmeticRightShiftGenerator();
OpGenerator tosaBitwiseAndGenerator();
OpGenerator tosaBitwiseNotGenerator();
OpGenerator tosaBitwiseOrGenerator();
OpGenerator tosaBitwiseXorGenerator();
OpGenerator tosaCastGenerator();
OpGenerator tosaCeilGenerator();
OpGenerator tosaIdentityOpGenerator();
OpGenerator tosaLogOpGenerator();
OpGenerator tosaLogicalAndOpGenerator();
OpGenerator tosaLogicalLeftShiftOpGenerator();
OpGenerator tosaLogicalNotOpGenerator();
OpGenerator tosaLogicalOrOpGenerator();
OpGenerator tosaLogicalRightShiftOpGenerator();
OpGenerator tosaLogicalXorOpGenerator();
OpGenerator tosaMatMulOpGenerator();
OpGenerator tosaMaxPool2dOpGenerator();
OpGenerator tosaMaximumOpGenerator();
OpGenerator tosaMinimumOpGenerator();
OpGenerator tosaMulOpGenerator();
OpGenerator tosaNegateOpGenerator();
OpGenerator tosaPadOpGenerator();
OpGenerator tosaPowOpGenerator();
OpGenerator tosaRFFT2dOpGenerator();
OpGenerator tosaReciprocalOpGenerator();
OpGenerator tosaReduceAllOpGenerator();
OpGenerator tosaReduceAnyOpGenerator();
OpGenerator tosaReduceMaxOpGenerator();
OpGenerator tosaReduceMinOpGenerator();
OpGenerator tosaReduceProdOpGenerator();
OpGenerator tosaReduceSumOpGenerator();
OpGenerator tosaRescaleOpGenerator();
OpGenerator tosaReshapeOpGenerator();
OpGenerator tosaResizeOpGenerator();
OpGenerator tosaReverseOpGenerator();
OpGenerator tosaRsqrtOpGenerator();
OpGenerator tosaScatterOpGenerator();
OpGenerator tosaSelectOpGenerator();
OpGenerator tosaSigmoidOpGenerator();
OpGenerator tosaSliceOpGenerator();
OpGenerator tosaSubOpGenerator();
OpGenerator tosaTableOpGenerator();
OpGenerator tosaTanhOpGenerator();
OpGenerator tosaTileOpGenerator();
OpGenerator tosaTransposeConv2DOpGenerator();
OpGenerator tosaTransposeOpGenerator();
// vector 
OpGenerator vectorBroadcastGenerator();
OpGenerator vectorBitCastGenerator();
OpGenerator vectorCompressStoreGenerator();
OpGenerator vectorConstantMaskGenerator();
OpGenerator vectorCreateMaskGenerator();
OpGenerator vectorContractGenerator();
OpGenerator vectorExpandLoadGenerator();
// OpGenerator vectorExtractGenerator();
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
// OpGenerator vectorOuterProductGenerator();
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


static std::vector<OpGenerator> generators = {
	// arith
	addFGenerator(),
	addIGenerator(),
	andIGenerator(),
	ceilDivSIGenerator(),
	ceilDivUIGenerator(),
	cmpFGenerator(),
	cmpIGenerator(),
	constantGenerator(),
	divFGenerator(),
	divSIGenerator(),
	divUIGenerator(),
	floorDivSIGenerator(),
	// maxFGenerator(),
	maxSIGenerator(),
	maxUIGenerator(),
	// minFGenerator(),
	minSIGenerator(),
	minUIGenerator(),
	mulFGenerator(),
	mulIGenerator(),
	negFGenerator(),
	orIGenerator(),
	remFGenerator(),
	remSIGenerator(),
	remUIGenerator(),
	shlIGenerator(),
	shrSIGenerator(),
	shrUIGenerator(),
	subFGenerator(),
	subIGenerator(),
	xorIGenerator(),
	// Affine Dialect
	affineApplyGenerator(),
	// affineForGenerator(),
	// affineIfGenerator(),
	affineLoadGenerator(),
	affineStoreGenerator(),
	affineMaxGenerator(),
	affineMinGenerator(),
	// affineParallelGenerator(),
	// affinePrefetchGenerator(),
	affineVectorLoadGenerator(),
	affineVectorStoreGenerator(),
	bufferizationAllocTensorGenerator(),
	bufferizationCloneGenerator(),
	bufferizationDeallocGenerator(),
	bufferizationToMemrefGenerator(),
	bufferizationToTensorGenerator(),
	// Index Dialect
	indexAddGenerator(),
	indexAndGenerator(),
	indexBoolConstantGenerator(),
	indexCastSGenerator(),
	indexCastUGenerator(),
	indexCeilDivSGenerator(),
	indexCeilDivUGenerator(),
	indexConstantGenerator(),
	indexDivSGenerator(),
	indexDivUGenerator(),
	indexFloorDivSGenerator(),
	indexMaxSGenerator(),
	indexMaxUGenerator(),
	indexMulGenerator(),
	indexOrGenerator(),
	indexRemSGenerator(),
	indexRemUGenerator(),
	indexShLGenerator(),
	indexShrSGenerator(),
	indexShrUGenerator(),
	indexSizeOfGenerator(),
	indexSubGenerator(),
	indexXorGenerator(),
	// linalg
	linalgGenericGenerator(),
	linalgBroadCastGenerator(),
	linalgTransposeGenerator(),
	linalgCopyGenerator(),
	linalgMapGenerator(),
	linalgReduceGenerator(),
	linalgDotGenerator(),
	linalgMatMulGenerator(),
	// Math 
	absFGenerator(),
	absIGenerator(),
	atanGenerator(),
	atan2Generator(),
	ceilGenerator(),
	copySignGenerator(),
	cosGenerator(),
	sinGenerator(),
	ctlzGenerator(),
	cttzGenerator(),
	ctpopGenerator(),
	expGenerator(),
	exp2Generator(),
	expm1Generator(),
	floorGenerator(),
	// fmaGenerator(),
	ipowiGenerator(),
	logGenerator(),
	log10Generator(),
	log1pGenerator(),
	log2Generator(),
	powfGenerator(),
	rsqrtGenerator(),
	sqrtGenerator(),
	tanGenerator(),
	tanhGenerator(),
	roundEvenGenerator(),
	roundGenerator(),
	truncGenerator(),
	fpowiGenerator(),
	//memref
	memrefLoadGenerator(),
	memrefStoreGenerator(),
	atomicRMWGenerator(),
	memrefCopyGenerator(),
	assumeAlignmentGenerator(),
	allocGenerator(),
	reallocGenerator(),
	// tensorStoreGenerator(),
	genericAtomicRMWGenerator(),
	allocaGenerator(),
	allocaScopeGenerator(),
	memrefCastGenerator(),
	// scf
	scfIfGenerator(),
	executeRegionGenerator(),
	scfForGenerator(),
	indexSwitchGenerator(),
	scfWhileGenerator(),
	scfParallelGenerator(),
	// spirv
	spirvBitCountGenerator(),
	spirvBitFieldInsertGenerator(),
	spirvBitFieldSExtractGenerator(),
	spirvBitFieldUExtractGenerator(),
	spirvBitReverseGenerator(),
	spirvNotGenerator(),
	spirvBitwiseAndGenerator(),
	spirvBitwiseOrGenerator(),
	spirvBitwiseXorGenerator(),
	spirvCLCeilGenerator(),
	spirvCLCosGenerator(),
	spirvCLErfGenerator(),
	spirvCLExpGenerator(),
	spirvCLFAbsGenerator(),
	spirvCLFloorGenerator(),
	spirvCLLogGenerator(),
	spirvCLRoundGenerator(),
	spirvCLRintGenerator(),
	spirvCLRsqrtGenerator(),
	spirvCLSinGenerator(),
	spirvCLSqrtGenerator(),
	spirvCLTanhGenerator(),
	spirvCLSAbsGenerator(),
	spirvCLFMaxGenerator(),
	spirvCLFMinGenerator(),
	spirvCLPowGenerator(),
	spirvCLSMaxGenerator(),
	spirvCLSMinGenerator(),
	spirvCLUMaxGenerator(),
	spirvCLUMinGenerator(),
	spirvCLFmaGenerator(),
	spirvFNegateGenerator(),
	spirvFOrdEqualGenerator(),
	spirvFOrdGreaterThanEqualGenerator(),
	spirvFOrdGreaterThanGenerator(),
	spirvFOrdLessThanEqualGenerator(),
	spirvFOrdLessThanGenerator(),
	spirvFOrdNotEqualGenerator(),
	spirvFUnordEqualGenerator(),
	spirvFUnordGreaterThanEqualGenerator(),
	spirvFUnordGreaterThanGenerator(),
	spirvFUnordLessThanEqualGenerator(),
	spirvFUnordLessThanGenerator(),
	spirvFUnordNotEqualGenerator(),
	spirvIEqualGenerator(),
	spirvINotEqualGenerator(),
	spirvLogicalEqualGenerator(),
	spirvLogicalNotGenerator(),
	spirvLogicalNotEqualGenerator(),
	spirvLogicalAndGenerator(),
	spirvLogicalOrGenerator(),
	spirvSGreaterThanEqualGenerator(),
	spirvSGreaterThanGenerator(),
	spirvSLessThanEqualGenerator(),
	spirvSLessThanGenerator(),
	spirvUGreaterThanEqualGenerator(),
	spirvUGreaterThanGenerator(),
	spirvULessThanEqualGenerator(),
	spirvULessThanGenerator(),
	spirvUnorderedGenerator(),
	spirvGLAcosGenerator(),
	spirvGLAsinGenerator(),
	spirvGLAtanGenerator(),
	spirvGLCeilGenerator(),
	spirvGLCosGenerator(),
	spirvGLCoshGenerator(),
	spirvGLExpGenerator(),
	spirvGLFAbsGenerator(),
	spirvGLFSignGenerator(),
	spirvGLFloorGenerator(),
	spirvGLInverseSqrtGenerator(),
	spirvGLLogGenerator(),
	spirvGLRoundEvenGenerator(),
	spirvGLRoundGenerator(),
	spirvGLSinGenerator(),
	spirvGLSinhGenerator(),
	spirvGLSqrtGenerator(),
	spirvGLTanGenerator(),
	spirvGLTanhGenerator(),
	spirvGLFClampGenerator(),
	spirvGLSAbsGenerator(),
	spirvGLSSignGenerator(),
	spirvGLFMaxGenerator(),
	spirvGLFMinGenerator(),
	spirvGLFMixGenerator(),
	spirvGLFindUMsbGenerator(),
	spirvGLFmaGenerator(),
	spirvGLLdexpGenerator(),
	spirvGLPowGenerator(),
	spirvGLSClampGenerator(),
	spirvGLSMaxGenerator(),
	spirvGLSMinGenerator(),
	spirvGLUMaxGenerator(),
	spirvGLUMinGenerator(),
	spirvGLUClampGenerator(),
	spirvIsInfGenerator(),
	spirvIsNanGenerator(),
	// tensor
	tensorCastGenerator(),
	tensorCollapseShapeGenerator(),
	tensorDimGenerator(),
	tensorEmptyGenerator(),
	tensorExpandShapeGenerator(),
	tensorExtractGenerator(),
	tensorExtractSliceGenerator(),
	tensorFromElementsGenerator(),
	tensorGenerateGenerator(),
	tensorInsertGenerator(),
	tensorInsertSliceGenerator(),
	tensorPackGenerator(),
	tensorRankGenerator(),
	tensorScatterGenerator(),
	tensorSplatGenerator(),
	tensorUnpackGenerator(),
	// TOSA
	tosaAbsGenerator(),
	tosaAddGenerator(),
	tosaApplyScaleGenerator(),
	tosaArgMaxGenerator(),
	tosaArithmeticRightShiftGenerator(),
	tosaBitwiseAndGenerator(),
	tosaBitwiseNotGenerator(),
	tosaBitwiseOrGenerator(),
	tosaBitwiseXorGenerator(),
	tosaCastGenerator(),
	tosaCeilGenerator(),
	tosaIdentityOpGenerator(),
	tosaLogOpGenerator(),
	tosaLogicalAndOpGenerator(),
	tosaLogicalLeftShiftOpGenerator(),
	tosaLogicalNotOpGenerator(),
	tosaLogicalOrOpGenerator(),
	tosaLogicalRightShiftOpGenerator(),
	tosaLogicalXorOpGenerator(),
	tosaMatMulOpGenerator(),
	tosaMaxPool2dOpGenerator(),
	tosaMaximumOpGenerator(),
	tosaMinimumOpGenerator(),
	tosaMulOpGenerator(),
	tosaNegateOpGenerator(),
	tosaPadOpGenerator(),
	tosaPowOpGenerator(),
	tosaRFFT2dOpGenerator(),
	tosaReciprocalOpGenerator(),
	tosaReduceAllOpGenerator(),
	tosaReduceAnyOpGenerator(),
	tosaReduceMaxOpGenerator(),
	tosaReduceMinOpGenerator(),
	tosaReduceProdOpGenerator(),
	tosaReduceSumOpGenerator(),
	tosaRescaleOpGenerator(),
	tosaReshapeOpGenerator(),
	tosaResizeOpGenerator(),
	tosaReverseOpGenerator(),
	tosaRsqrtOpGenerator(),
	tosaScatterOpGenerator(),
	tosaSelectOpGenerator(),
	tosaSigmoidOpGenerator(),
	tosaSliceOpGenerator(),
	tosaSubOpGenerator(),
	tosaTableOpGenerator(),
	tosaTanhOpGenerator(),
	tosaTileOpGenerator(),
	tosaTransposeConv2DOpGenerator(),
	tosaTransposeOpGenerator(),
	// vector 
	vectorBroadcastGenerator(),
	vectorBitCastGenerator(),
	vectorCompressStoreGenerator(),
	vectorConstantMaskGenerator(),
	vectorCreateMaskGenerator(),
	vectorContractGenerator(),
	vectorExpandLoadGenerator(),
	// vectorExtractGenerator(),
	vectorExtractElementGenerator(),
	vectorExtractStridedSliceGenerator(),
	vectorFMAGenerator(),
	vectorFlatTransposeGenerator(),
	vectorGatherGenerator(),
	vectorInsertElementGenerator(),
	vectorInsertGenerator(),
	vectorInsertStridedSliceGenerator(),
	vectorLoadGenerator(),
	vectorMaskGenerator(),
	vectorMaskedLoadGenerator(),
	vectorStoreGenerator(),
	vectorMaskedStoreGenerator(),
	vectorMatrixMultiplyGenerator(),
	vectorMultiReductionGenerator(),
	// vectorOuterProductGenerator(),
	vectorPrintGenerator(),
	vectorReductionGenerator(),
	vectorScanGenerator(),
	vectorScatterGenerator(),
	vectorShuffleGenerator(),
	vectorSplatGenerator(),
	vectorTransposeGenerator(),
	vectorTransferReadGenerator(),
	vectorTransferWriteGenerator(),
	vectorWarpExecuteOnLane0Op(),
};

inline std::vector<OpGenerator> intOpsForGenericAtomicRMW = {
	addIGenerator(),
	andIGenerator(),
	cmpIGenerator(),
	ceilDivSIGenerator(),
	constantGenerator(),
	divSIGenerator(),
	divUIGenerator(),
	floorDivSIGenerator(),
	maxSIGenerator(),
	maxUIGenerator(),
	minSIGenerator(),
	minUIGenerator(),
	mulIGenerator(),
	orIGenerator(),
	remSIGenerator(),
	remUIGenerator(),
	shlIGenerator(),
	shrSIGenerator(),
	shrUIGenerator(),
	subIGenerator(),
	xorIGenerator(),
	absIGenerator(),
	ipowiGenerator(),
	logGenerator(),
	log10Generator(),
	fpowiGenerator(),
};

inline std::vector<OpGenerator> floatOpsForGenericAtomicRMW = {
	addFGenerator(),
	cmpFGenerator(),
	constantGenerator(),
	divFGenerator(),
	// maxFGenerator(),
	// minFGenerator(),
	mulFGenerator(),
	negFGenerator(),
	remFGenerator(),
	subFGenerator(),
	absFGenerator(),
	atanGenerator(),
	atan2Generator(),
	ceilGenerator(),
	copySignGenerator(),
	cosGenerator(),
	ctlzGenerator(),
	cttzGenerator(),
	ctpopGenerator(),
	expGenerator(),
	exp2Generator(),
	expm1Generator(),
	logGenerator(),
	log10Generator(),
	log1pGenerator(),
	log2Generator(),
	powfGenerator(),
	rsqrtGenerator(),
	sqrtGenerator(),
	tanGenerator(),
	tanhGenerator(),
	roundEvenGenerator(),
	roundGenerator(),
	truncGenerator(),
	fpowiGenerator(),
};

inline std::vector<OpGenerator> opsForScfFor = generators;
inline std::vector<OpGenerator> maskableOps = {
		// arith
	addFGenerator(),
	addIGenerator(),
	andIGenerator(),
	ceilDivSIGenerator(),
	ceilDivUIGenerator(),
	cmpFGenerator(),
	cmpIGenerator(),
	constantGenerator(),
	divFGenerator(),
	divSIGenerator(),
	divUIGenerator(),
	floorDivSIGenerator(),
	// maxFGenerator(),
	maxSIGenerator(),
	maxUIGenerator(),
	// minFGenerator(),
	minSIGenerator(),
	minUIGenerator(),
	mulFGenerator(),
	mulIGenerator(),
	negFGenerator(),
	orIGenerator(),
	remFGenerator(),
	remSIGenerator(),
	remUIGenerator(),
	shlIGenerator(),
	shrSIGenerator(),
	shrUIGenerator(),
	subFGenerator(),
	subIGenerator(),
	xorIGenerator(),
	// Affine Dialect
	affineApplyGenerator(),
	// affineForGenerator(),
	// affineIfGenerator(),
	affineLoadGenerator(),
	affineStoreGenerator(),
	affineMaxGenerator(),
	affineMinGenerator(),
	// affineParallelGenerator(),
	// affinePrefetchGenerator(),
	affineVectorLoadGenerator(),
	affineVectorStoreGenerator(),
	bufferizationAllocTensorGenerator(),
	bufferizationCloneGenerator(),
	bufferizationDeallocGenerator(),
	bufferizationToMemrefGenerator(),
	bufferizationToTensorGenerator(),
	// Index Dialect
	indexAddGenerator(),
	indexAndGenerator(),
	indexBoolConstantGenerator(),
	indexCastSGenerator(),
	indexCastUGenerator(),
	indexCeilDivSGenerator(),
	indexCeilDivUGenerator(),
	indexConstantGenerator(),
	indexDivSGenerator(),
	indexDivUGenerator(),
	indexFloorDivSGenerator(),
	indexMaxSGenerator(),
	indexMaxUGenerator(),
	indexMulGenerator(),
	indexOrGenerator(),
	indexRemSGenerator(),
	indexRemUGenerator(),
	indexShLGenerator(),
	indexShrSGenerator(),
	indexShrUGenerator(),
	indexSizeOfGenerator(),
	indexSubGenerator(),
	indexXorGenerator(),
	// linalg
	// linalgGenericGenerator(),
	linalgBroadCastGenerator(),
	linalgTransposeGenerator(),
	linalgCopyGenerator(),
	// linalgMapGenerator(),
	linalgReduceGenerator(),
	linalgDotGenerator(),
	linalgMatMulGenerator(),
	// Math 
	absFGenerator(),
	absIGenerator(),
	atanGenerator(),
	atan2Generator(),
	ceilGenerator(),
	copySignGenerator(),
	cosGenerator(),
	sinGenerator(),
	ctlzGenerator(),
	cttzGenerator(),
	ctpopGenerator(),
	expGenerator(),
	exp2Generator(),
	expm1Generator(),
	floorGenerator(),
	// fmaGenerator(),
	ipowiGenerator(),
	logGenerator(),
	log10Generator(),
	log1pGenerator(),
	log2Generator(),
	powfGenerator(),
	rsqrtGenerator(),
	sqrtGenerator(),
	tanGenerator(),
	tanhGenerator(),
	roundEvenGenerator(),
	roundGenerator(),
	truncGenerator(),
	fpowiGenerator(),
	//memref
	memrefLoadGenerator(),
	memrefStoreGenerator(),
	atomicRMWGenerator(),
	memrefCopyGenerator(),
	assumeAlignmentGenerator(),
	allocGenerator(),
	reallocGenerator(),
	// tensorStoreGenerator(),
	genericAtomicRMWGenerator(),
	allocaGenerator(),
	// allocaScopeGenerator(),
	memrefCastGenerator(),
	// scf
	// scfIfGenerator(),
	// executeRegionGenerator(),
	// scfForGenerator(),
	// indexSwitchGenerator(),
	// scfWhileGenerator(),
	// scfParallelGenerator(),
};


// inline std::set<std::string> regionedOps = {
//     "func.func",       "memref.alloca_scope",
//     "scf.if",          "scf.execute_region",
//     "linalg.map",      "scf.for",
//     "linalg.generic",  "scf.index_switch",
//     "scf.while",       "scf.parallel",
//     "affine.for",      "affine.if",
//     // "affine.parallel",
//     "vector.warp_execute_on_lane0",
//     "tensor.generate"};


static std::vector<std::string> generatorsStr = {
"addFGenerator",
"addIGenerator",
"andIGenerator",
"ceilDivSIGenerator",
"ceilDivUIGenerator",
"cmpFGenerator",
"cmpIGenerator",
"constantGenerator",
"divFGenerator",
"divSIGenerator",
"divUIGenerator",
"floorDivSIGenerator",
"maxSIGenerator",
"maxUIGenerator",
"minSIGenerator",
"minUIGenerator",
"mulFGenerator",
"mulIGenerator",
"negFGenerator",
"orIGenerator",
"remFGenerator",
"remSIGenerator",
"remUIGenerator",
"shlIGenerator",
"shrSIGenerator",
"shrUIGenerator",
"subFGenerator",
"subIGenerator",
"xorIGenerator",
"affineApplyGenerator",
"affineLoadGenerator",
"affineStoreGenerator",
"affineMaxGenerator",
"affineMinGenerator",
"affineVectorLoadGenerator",
"affineVectorStoreGenerator",
"bufferizationAllocTensorGenerator",
"bufferizationCloneGenerator",
"bufferizationDeallocGenerator",
"bufferizationToMemrefGenerator",
"bufferizationToTensorGenerator",
"indexAddGenerator",
"indexAndGenerator",
"indexBoolConstantGenerator",
"indexCastSGenerator",
"indexCastUGenerator",
"indexCeilDivSGenerator",
"indexCeilDivUGenerator",
"indexConstantGenerator",
"indexDivSGenerator",
"indexDivUGenerator",
"indexFloorDivSGenerator",
"indexMaxSGenerator",
"indexMaxUGenerator",
"indexMulGenerator",
"indexOrGenerator",
"indexRemSGenerator",
"indexRemUGenerator",
"indexShLGenerator",
"indexShrSGenerator",
"indexShrUGenerator",
"indexSizeOfGenerator",
"indexSubGenerator",
"indexXorGenerator",
"linalgGenericGenerator",
"linalgBroadCastGenerator",
"linalgTransposeGenerator",
"linalgCopyGenerator",
"linalgMapGenerator",
"linalgReduceGenerator",
"linalgDotGenerator",
"linalgMatMulGenerator",
"absFGenerator",
"absIGenerator",
"atanGenerator",
"atan2Generator",
"ceilGenerator",
"copySignGenerator",
"cosGenerator",
"sinGenerator",
"ctlzGenerator",
"cttzGenerator",
"ctpopGenerator",
"expGenerator",
"exp2Generator",
"expm1Generator",
"floorGenerator",
"ipowiGenerator",
"logGenerator",
"log10Generator",
"log1pGenerator",
"log2Generator",
"powfGenerator",
"rsqrtGenerator",
"sqrtGenerator",
"tanGenerator",
"tanhGenerator",
"roundEvenGenerator",
"roundGenerator",
"truncGenerator",
"fpowiGenerator",
"memrefLoadGenerator",
"memrefStoreGenerator",
"atomicRMWGenerator",
"memrefCopyGenerator",
"assumeAlignmentGenerator",
"allocGenerator",
"reallocGenerator",
// "tensorStoreGenerator",
"genericAtomicRMWGenerator",
"allocaGenerator",
"allocaScopeGenerator",
"memrefCastGenerator",
"scfIfGenerator",
"executeRegionGenerator",
"scfForGenerator",
"indexSwitchGenerator",
"scfWhileGenerator",
"scfParallelGenerator",
"spirvBitCountGenerator",
"spirvBitFieldInsertGenerator",
"spirvBitFieldSExtractGenerator",
"spirvBitFieldUExtractGenerator",
"spirvBitReverseGenerator",
"spirvNotGenerator",
"spirvBitwiseAndGenerator",
"spirvBitwiseOrGenerator",
"spirvBitwiseXorGenerator",
"spirvCLCeilGenerator",
"spirvCLCosGenerator",
"spirvCLErfGenerator",
"spirvCLExpGenerator",
"spirvCLFAbsGenerator",
"spirvCLFloorGenerator",
"spirvCLLogGenerator",
"spirvCLRoundGenerator",
"spirvCLRintGenerator",
"spirvCLRsqrtGenerator",
"spirvCLSinGenerator",
"spirvCLSqrtGenerator",
"spirvCLTanhGenerator",
"spirvCLSAbsGenerator",
"spirvCLFMaxGenerator",
"spirvCLFMinGenerator",
"spirvCLPowGenerator",
"spirvCLSMaxGenerator",
"spirvCLSMinGenerator",
"spirvCLUMaxGenerator",
"spirvCLUMinGenerator",
"spirvCLFmaGenerator",
"spirvFNegateGenerator",
"spirvFOrdEqualGenerator",
"spirvFOrdGreaterThanEqualGenerator",
"spirvFOrdGreaterThanGenerator",
"spirvFOrdLessThanEqualGenerator",
"spirvFOrdLessThanGenerator",
"spirvFOrdNotEqualGenerator",
"spirvFUnordEqualGenerator",
"spirvFUnordGreaterThanEqualGenerator",
"spirvFUnordGreaterThanGenerator",
"spirvFUnordLessThanEqualGenerator",
"spirvFUnordLessThanGenerator",
"spirvFUnordNotEqualGenerator",
"spirvIEqualGenerator",
"spirvINotEqualGenerator",
"spirvLogicalEqualGenerator",
"spirvLogicalNotGenerator",
"spirvLogicalNotEqualGenerator",
"spirvLogicalAndGenerator",
"spirvLogicalOrGenerator",
"spirvSGreaterThanEqualGenerator",
"spirvSGreaterThanGenerator",
"spirvSLessThanEqualGenerator",
"spirvSLessThanGenerator",
"spirvUGreaterThanEqualGenerator",
"spirvUGreaterThanGenerator",
"spirvULessThanEqualGenerator",
"spirvULessThanGenerator",
"spirvUnorderedGenerator",
"spirvGLAcosGenerator",
"spirvGLAsinGenerator",
"spirvGLAtanGenerator",
"spirvGLCeilGenerator",
"spirvGLCosGenerator",
"spirvGLCoshGenerator",
"spirvGLExpGenerator",
"spirvGLFAbsGenerator",
"spirvGLFSignGenerator",
"spirvGLFloorGenerator",
"spirvGLInverseSqrtGenerator",
"spirvGLLogGenerator",
"spirvGLRoundEvenGenerator",
"spirvGLRoundGenerator",
"spirvGLSinGenerator",
"spirvGLSinhGenerator",
"spirvGLSqrtGenerator",
"spirvGLTanGenerator",
"spirvGLTanhGenerator",
"spirvGLFClampGenerator",
"spirvGLSAbsGenerator",
"spirvGLSSignGenerator",
"spirvGLFMaxGenerator",
"spirvGLFMinGenerator",
"spirvGLFMixGenerator",
"spirvGLFindUMsbGenerator",
"spirvGLFmaGenerator",
"spirvGLLdexpGenerator",
"spirvGLPowGenerator",
"spirvGLSClampGenerator",
"spirvGLSMaxGenerator",
"spirvGLSMinGenerator",
"spirvGLUMaxGenerator",
"spirvGLUMinGenerator",
"spirvGLUClampGenerator",
"spirvIsInfGenerator",
"spirvIsNanGenerator",
"tensorCastGenerator",
"tensorCollapseShapeGenerator",
"tensorDimGenerator",
"tensorEmptyGenerator",
"tensorExpandShapeGenerator",
"tensorExtractGenerator",
"tensorExtractSliceGenerator",
"tensorFromElementsGenerator",
"tensorGenerateGenerator",
"tensorInsertGenerator",
"tensorInsertSliceGenerator",
"tensorPackGenerator",
"tensorRankGenerator",
"tensorScatterGenerator",
"tensorSplatGenerator",
"tensorUnpackGenerator",
"tosaAbsGenerator",
"tosaAddGenerator",
"tosaApplyScaleGenerator",
"tosaArgMaxGenerator",
"tosaArithmeticRightShiftGenerator",
"tosaBitwiseAndGenerator",
"tosaBitwiseNotGenerator",
"tosaBitwiseOrGenerator",
"tosaBitwiseXorGenerator",
"tosaCastGenerator",
"tosaCeilGenerator",
"tosaIdentityOpGenerator",
"tosaLogOpGenerator",
"tosaLogicalAndOpGenerator",
"tosaLogicalLeftShiftOpGenerator",
"tosaLogicalNotOpGenerator",
"tosaLogicalOrOpGenerator",
"tosaLogicalRightShiftOpGenerator",
"tosaLogicalXorOpGenerator",
"tosaMatMulOpGenerator",
"tosaMaxPool2dOpGenerator",
"tosaMaximumOpGenerator",
"tosaMinimumOpGenerator",
"tosaMulOpGenerator",
"tosaNegateOpGenerator",
"tosaPadOpGenerator",
"tosaPowOpGenerator",
"tosaRFFT2dOpGenerator",
"tosaReciprocalOpGenerator",
"tosaReduceAllOpGenerator",
"tosaReduceAnyOpGenerator",
"tosaReduceMaxOpGenerator",
"tosaReduceMinOpGenerator",
"tosaReduceProdOpGenerator",
"tosaReduceSumOpGenerator",
"tosaRescaleOpGenerator",
"tosaReshapeOpGenerator",
"tosaResizeOpGenerator",
"tosaReverseOpGenerator",
"tosaRsqrtOpGenerator",
"tosaScatterOpGenerator",
"tosaSelectOpGenerator",
"tosaSigmoidOpGenerator",
"tosaSliceOpGenerator",
"tosaSubOpGenerator",
"tosaTableOpGenerator",
"tosaTanhOpGenerator",
"tosaTileOpGenerator",
"tosaTransposeConv2DOpGenerator",
"tosaTransposeOpGenerator",
"vectorBroadcastGenerator",
"vectorBitCastGenerator",
"vectorCompressStoreGenerator",
"vectorConstantMaskGenerator",
"vectorCreateMaskGenerator",
"vectorContractGenerator",
"vectorExpandLoadGenerator",
// "vectorExtractGenerator",
"vectorExtractElementGenerator",
"vectorExtractStridedSliceGenerator",
"vectorFMAGenerator",
"vectorFlatTransposeGenerator",
"vectorGatherGenerator",
"vectorInsertElementGenerator",
"vectorInsertGenerator",
"vectorInsertStridedSliceGenerator",
"vectorLoadGenerator",
"vectorMaskGenerator",
"vectorMaskedLoadGenerator",
"vectorStoreGenerator",
"vectorMaskedStoreGenerator",
"vectorMatrixMultiplyGenerator",
"vectorMultiReductionGenerator",
// "vectorOuterProductGenerator",
"vectorPrintGenerator",
"vectorReductionGenerator",
"vectorScanGenerator",
"vectorScatterGenerator",
"vectorShuffleGenerator",
"vectorSplatGenerator",
"vectorTransposeGenerator",
"vectorTransferReadGenerator",
"vectorTransferWriteGenerator",
"vectorWarpExecuteOnLane0Op",
};

#endif


