//
// Created by Stan Wang on 2022/10/28.
//
#include "smith/RegionGeneration.h"
#include "smith/TypeGeneration.h"
#include "smith/Util.h"
#include "smith/generators/OpGeneration.h"

static int func_index = 0;
void printEachVector(OpBuilder &builder, Location loc, OpRegion &opRegion) {
  for (auto tval : opRegion.pool.vectorPool) {
    builder.create<vector::PrintOp>(loc, tval.val);
  }
  for (auto tval : opRegion.pool.intOrFloatPool) {
    builder.create<vector::PrintOp>(loc, tval.val);
  }
}

void consumeTensor(OpBuilder &builder, Location loc, OpRegion &p) {}

OpGenerator funcGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    assert(parent.parent_op == "builtin.module");
    auto point = builder.saveInsertionPoint();
    OpRegion region = OpRegion("func.func", parent.depth + 1, parent.cur_child);
    std::string funcName = "func" + std::to_string(++func_index);
    auto funcType = randomFunctionType(builder.getContext());

    auto funcOp = builder.create<func::FuncOp>(
        loc, funcName, funcType.dyn_cast<FunctionType>());

    switch (UR(4)) {
    case 0: {
      funcOp.setPrivate();
      break;
    }
    case 1: {
      funcOp.setPublic();
      break;
    }
    case 2: {
      funcOp.setNested();
      break;
    }
    default: {
      break;
    }
    }
    funcOp.addEntryBlock();
    auto inputTypes = funcType.getInputs();
    auto args = funcOp.getBody().getArguments();
    for (uint32_t i = 0; i < funcType.getNumInputs(); ++i) {
      auto type = inputTypes[i];
      auto arg = args[i];
      auto tVal = TypeValue(type, arg);
      region.pool.addTypeValue(tVal, "arg(func)");
    }
    builder.setInsertionPointToEnd(&funcOp.getBody().front());
    initGenerator()(builder, loc, region);
    auto filter = OpNameFilter(opsForFunc);
    auto regionGen = RegionGen(&region, {filter});
    regionGen.apply(builder, loc, 128);
    printEachVector(builder, loc, region);
    if (funcType.getResults().size() > 0) {
      SmallVector<Value> retValues;
      for (auto type : funcType.getResults()) {
        auto candidates = searchTypedValueFrom(region.pool, type);
        if (candidates.empty()) {
          candidates.push_back(
              region.pool.generateTypedValue(builder, loc, type));
        }
        retValues.push_back(
            sampleTypedValueFrom(candidates, "func.return").val);
      }
      builder.create<func::ReturnOp>(loc, ValueRange(retValues));
    } else {
      builder.create<func::ReturnOp>(loc);
    }
    builder.restoreInsertionPoint(point);
    // avoid generate recursive call chain
    funcPool.push_back(funcOp);
    return funcOp.getOperation();
  };
}

OpGenerator callGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &parent) {
    auto callee = funcPool[UR(funcPool.size())];
    SmallVector<Value> arguments;
    for (auto type : callee.getFunctionType().getInputs()) {
      auto candidates = parent.pool.searchCandidatesFrom(
          allTypes, typeEquivalentFilter(type));
      if (candidates.empty()) {
        candidates.push_back(
            parent.pool.generateTypedValue(builder, loc, type));
      }
      arguments.push_back(sampleTypedValueFrom(candidates, "func.call").val);
    }
    return builder.create<func::CallOp>(loc, callee, ValueRange(arguments))
        .getOperation();
  };
}