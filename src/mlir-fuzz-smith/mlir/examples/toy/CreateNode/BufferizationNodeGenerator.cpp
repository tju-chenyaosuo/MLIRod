#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

OpGenerator bufferizationAllocTensorGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		const int rank_ub = 3;
		const int dim_ub = 32;

		RankedTensorType type;
		bool isDynamic = rollIdx(2);
		int dyDims = 0;
    SmallVector<Value> dyDimVals;
		if (isDynamic) {
			std::vector<int64_t> shape;
			
      auto rank = rollIdx(rank_ub + 1); // 0-3
			std::vector<mlir::Value> indexCandidates = std::vector<mlir::Value>();
			mb->search(index_filter, &indexCandidates);
			if (indexCandidates.empty()) {
				indexCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));
			}
			for (int i = 0; i < rank; ++i) {
        if (rollIdx(2)) {
					dyDimVals.push_back(indexCandidates[rollIdx(indexCandidates.size())]);
          shape.push_back(ShapedType::kDynamic);
        } else {
          shape.push_back(rollIdx(dim_ub)+1);
        }
      }
			type = RankedTensorType::get(shape, randomIntOrFloatType(builder.getContext()));
		} else {
			std::vector<int64_t> shape = getRandomShape();
      type = RankedTensorType::get(shape, randomIntOrFloatType(builder.getContext()));
		}
		mlir::Value val = builder.create<bufferization::AllocTensorOp>(loc, type, ValueRange(dyDimVals));
    mb->add2Pool(val);
		return true;
  };
}

OpGenerator bufferizationCloneGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
		mb->search(static_memref_filter, &memrefCandidates);
		if (memrefCandidates.empty()) {
			memrefCandidates.push_back(generateStaticShapedMemref(
				builder, loc, randomStaticShapedMemrefType(builder.getContext())));
		}
		auto mem = memrefCandidates[rollIdx(memrefCandidates.size())];
		mlir::Value op = builder.create<bufferization::CloneOp>(loc, mem);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator bufferizationDeallocGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &tensorCandidates);

		llvm::outs() << "aaaaaaaaaaaaaaaaaaaaaa\n";

		if (tensorCandidates.empty()) {
			tensorCandidates.push_back(
				generateStaticShapedTensor(
					builder, loc, randomStaticShapedTensorType(builder.getContext())
					)
			);
		}
		auto tensor = tensorCandidates[rollIdx(tensorCandidates.size())];
		builder.create<bufferization::DeallocTensorOp>(loc, tensor);
		return true;
  };
}

OpGenerator bufferizationToMemrefGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> tensorCandidates = std::vector<mlir::Value>();
		mb->search(ranked_tensor_filter, &tensorCandidates);
		if (tensorCandidates.empty()) {
			tensorCandidates.push_back(
				generateRankedTensor(builder, loc, randomRankedTensorType(builder.getContext()), mb));
		}
		auto tensor = tensorCandidates[rollIdx(tensorCandidates.size())];
		auto shapedTy = tensor.getType().dyn_cast<ShapedType>();

    auto resultMemTy = MemRefType::get(shapedTy.getShape(), shapedTy.getElementType());
    mlir::Value op = builder.create<bufferization::ToMemrefOp>(loc, resultMemTy, tensor);
		mb->add2Pool(op);
		return true;
  };
}

OpGenerator bufferizationToTensorGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> memrefCandidates = std::vector<mlir::Value>();
		mb->search(ranked_memref_filter, &memrefCandidates);
		if (memrefCandidates.empty()) {
			memrefCandidates.push_back(generateRankedMemref(
				builder, loc, randomRankedMemrefType(builder.getContext()), mb));
		}
		auto memref = memrefCandidates[rollIdx(memrefCandidates.size())];
		auto shapedTy = memref.getType().dyn_cast<ShapedType>();
    auto resTensorTy = RankedTensorType::get(shapedTy.getShape(), shapedTy.getElementType());
    mlir::Value op = builder.create<bufferization::ToTensorOp>(loc, resTensorTy, memref);
		mb->add2Pool(op);
		return true;
  };
}