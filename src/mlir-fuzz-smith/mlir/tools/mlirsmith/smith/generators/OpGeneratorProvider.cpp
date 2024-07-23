//
// Created by Stan Wang on 2022/9/13.
//
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"

void OpGeneration::apply(mlir::OpBuilder &builder, mlir::Location loc,
                         OpRegion &region) {

//  std::cout << "apply: " << region.pool.vectorPool.size() << std::endl;
  this->generator(builder, loc, region);
}

OpGenerator initGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    region.pool.clearAll();
    time_t t;
    srand((unsigned)time(&t));

    std::vector<mlir::Type> supportedIntOrFloatTypes =
        getSupportedIntOrFloatTypes(builder.getContext());

    // init element constants pool
    for (int i = 0; i < eleConstantSize; i++) {
      auto elementType =
          supportedIntOrFloatTypes[UR(supportedIntOrFloatTypes.size())];
      IntegerType iType = elementType.dyn_cast<IntegerType>();
      if (iType) {
        // TODO- support value width greater than 32.
        unsigned width = min(iType.getWidth(), 32);
        long long val = UR(((long long)1) << width);
        Value cons = builder.create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(elementType, val));
        auto elem = TypeValue(iType, cons);
        region.pool.addIntOrFloat(elem, "arith.constant");
      }
      FloatType fType = elementType.dyn_cast<FloatType>();
      if (fType) {
        unsigned width = min(fType.getWidth(), 32);
        long long val = UR(((long long)1) << width);
        double valf = static_cast<double>(val);
        Value cons = builder.create<arith::ConstantOp>(
            loc, builder.getFloatAttr(elementType, valf));
        auto elem = TypeValue(fType, cons);
        region.pool.addIntOrFloat(elem, "arith.constant");
      }
    }

    for (int i = 0; i < idx_ub; i++) {
      auto idx = builder.create<arith::ConstantIndexOp>(loc, i);
      region.pool.indexPool.push_back(TypeValue(builder.getIndexType(), idx));
      region.pool.constantIndices.insert(std::make_pair(i, idx));
    }

    // init tensor constants
    for (int i = 0; i < tenConsSize; i++) {
      if (UR(2)) {
        auto rtTy = randomStaticShapedTensorType(builder.getContext());
        auto tensor =
            region.pool.generateStaticShapedTensor(builder, loc, rtTy);
        region.pool.addRankedTensor(tensor, "tensor.init");
      } else {
        auto rtDTy = randomDynamicShapedTensorType(builder.getContext());
        auto tensor =
            region.pool.generateDynamicShapedTensor(builder, loc, rtDTy);
        region.pool.addRankedTensor(tensor, "tensor.init");
      }
    }

    for (int i = 0; i < memAllocsSize; i++) {
      region.pool.generateRankedMemref(builder, loc,
                                 randomRankedMemrefType(builder.getContext()));
    }

    // randomly init affine ExprPool
    for (int i = 0; i < affine_ub; i++) {
      std::vector<int> set;
      for (size_t j = 0; j < rank_ub; j++) {
        set.push_back(j);
      }
      SmallVector<AffineExpr> affineExprs;
      for (size_t j = 0; j < rank_ub; j++) {
        if (UR(2)) {
          int idx = UR(set.size());
          affineExprs.push_back(builder.getAffineDimExpr(set[idx]));
          set.erase(set.begin() + idx);
        }
      }
      if (!affineExprs.empty()) {
        region.pool.affineExprPool.push_back(affineExprs);
      }
    }
    SmallVector<AffineExpr> fullDimAffineExprs;
    for (size_t i = 0; i < rank_ub; i++) {
      fullDimAffineExprs.push_back(builder.getAffineDimExpr(i));
    }
    region.pool.affineExprPool.push_back(fullDimAffineExprs);

    // init affine maps
    for (int i = 0; i < affine_map_pool_num; ++i) {
      int dimCount = UR(affine_expr_dim_ub) + 1;
      auto affineExprs = randomAffineExprs(builder, dimCount, 0);
      auto resultExpr = affineExprs[UR(affineExprs.size())];
      auto map = AffineMap::get(dimCount, UR(2), resultExpr);
      region.pool.affineMapPool.push_back(map);
    }

    // init integer set
    for (int i = 0; i < integer_set_num; ++i) {
      int dimCount = UR(affine_expr_dim_ub) + 1;
      int symbolCount = 0;
      int exprCount = UR(integer_set_exprs_ub) + 1;
      SmallVector<AffineExpr> constraints;
      SmallVector<bool> eqFlags;

      auto affineExprs = randomAffineExprs(builder, dimCount, 0);
      for (int j = 0; j < exprCount; ++j) {
        auto constraint = affineExprs[UR(affineExprs.size())];
        constraints.push_back(constraint);
        eqFlags.push_back(UR(2));
      }
      auto set = IntegerSet::get(dimCount, 0, constraints, eqFlags);
      region.pool.integerSetPool.push_back(set);
    }
    return nullptr;
  };
}

OpGenerator emptyGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    std::cout
        << "The empty generator should not appear in the release version\n";
    return nullptr;
  };
}

OpGen getOpGen(std::string name) {
  if (operators.find(name) != operators.end()) {
    return operators[name];
  } else {
    if (conf.debug) {
      // debug to see any OpGenerator oops.
      auto empty = [](OpBuilder &builder, Location loc, OpRegion &region) {
        return nullptr;
      };
      return OpGen("empty", empty);
    } else {
      auto str = "operation not found: " + name + "\n";
      std ::cout << str; // We should not debug print here. since the debug
                         // option is closed
      exit(-1);
    }
  }
}