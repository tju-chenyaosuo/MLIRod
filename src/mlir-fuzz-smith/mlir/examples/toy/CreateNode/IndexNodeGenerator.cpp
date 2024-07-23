#include "NodeGenerator.h"
#include "MutationUtil.h"
#include "TypeBase.h"

OpGenerator indexAddGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(10)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::AddOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexAndGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(10)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<index::AndOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexBoolConstantGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    mlir::Value res = builder.create<index::BoolConstantOp>(loc, rollIdx(2));
		mb->add2Pool(res);
    return true;
  };
}

OpGenerator indexCastSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    if (rollIdx(2)) {
			std::vector<mlir::Value> intCandidates = std::vector<mlir::Value>();
			mb->search(int_filter, &intCandidates);
			if (intCandidates.empty()) {
				intCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
			}
			auto operand = intCandidates[rollIdx(intCandidates.size())];
			mlir::Value res = builder.create<index::CastSOp>(loc, builder.getIndexType(), operand);
			mb->add2Pool(res);
    } else {
			std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
			mb->search(index_filter, &idxCandidates);
			if (idxCandidates.empty()) {
				idxCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));
			}
			auto operand = idxCandidates[rollIdx(idxCandidates.size())];
      mlir::Value res = builder.create<index::CastSOp>(loc, builder.getI32Type(), operand);
			mb->add2Pool(res);
    }
		return true;
  };
}

OpGenerator indexCastUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    if (rollIdx(2)) {
			std::vector<mlir::Value> intCandidates = std::vector<mlir::Value>();
			mb->search(int_filter, &intCandidates);
			if (intCandidates.empty()) {
				intCandidates.push_back(generateInteger(builder, loc, builder.getI32Type()));
			}
			auto operand = intCandidates[rollIdx(intCandidates.size())];
			mlir::Value res = builder.create<index::CastUOp>(loc, builder.getIndexType(), operand);
			mb->add2Pool(res);
    } else {
			std::vector<mlir::Value> idxCandidates = std::vector<mlir::Value>();
			mb->search(index_filter, &idxCandidates);
			if (idxCandidates.empty()) {
				idxCandidates.push_back(generateIndex(builder, loc, rollIdx(10)));
			}
			auto operand = idxCandidates[rollIdx(idxCandidates.size())];
			mlir::Value res = builder.create<index::CastUOp>(loc, builder.getI32Type(), operand);
			mb->add2Pool(res);
    }
		return true;
  };
}

OpGenerator indexCeilDivSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::CeilDivSOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexCeilDivUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<index::CeilDivUOp>(loc, operand0, operand1);
    mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexConstantGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    mlir::Value res = builder.create<index::ConstantOp>(loc, rollIdx(8));
    mb->add2Pool(res);
		return true;		
  };
}

OpGenerator indexDivSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<index::DivSOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexDivUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<index::DivUOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexFloorDivSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<index::FloorDivSOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexMaxSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
    mlir::Value res = builder.create<index::MaxSOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexMaxUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
    std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::MaxUOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexMulGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::MulOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexOrGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::OrOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexRemSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::RemSOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexRemUGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::RemUOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexShLGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::ShlOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexShrSGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::ShrSOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexShrUGenerator(){
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::ShrUOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexSizeOfGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		mlir::Value res =  builder.create<index::SizeOfOp>(loc);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexSubGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::SubOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}

OpGenerator indexXorGenerator() {
  return [](OpBuilder &builder, Location loc, MutationBlock* mb) -> bool {
		std::vector<mlir::Value> candidates = std::vector<mlir::Value>();
		mb->search(index_filter, &candidates);
		if (candidates.empty()) {
			candidates.push_back(generateIndex(builder, loc, rollIdx(100)));
		}
		auto operand0 = candidates[rollIdx(candidates.size())];
		auto operand1 = candidates[rollIdx(candidates.size())];
		mlir::Value res = builder.create<index::XOrOp>(loc, operand0, operand1);
		mb->add2Pool(res);
		return true;
  };
}