//
// Created by Stan Wang on 2022/10/31.
//

#ifndef MLIR_FUZZ_REGIONGENERATION_H
#define MLIR_FUZZ_REGIONGENERATION_H

#include "generators/OpGeneration.h"
#include "smith/DiversityCriteria.h"
#include "smith/ExpSetting.h"

using OpGenFilter = std::function<bool(OpGen &)>;

// & needed here, otherwise ops is empty???
// I guess it's due to keyword inline.
inline OpGenFilter OpNameFilter(std::set<std::string> &ops) {
  return [&](OpGen &gen) {
    bool found = ops.find(gen.opName) != ops.end();
    return found;
  };
}

inline OpGenFilter excludeFilter(std::set<std::string> &ops) {
  return [&](OpGen &gen) {
    bool found = ops.find(gen.opName) != ops.end();
    return !found;
  };
}

inline OpGenFilter defaultFilter = [](OpGen &gen) { return true; };

inline OpGenFilter funcCallExFilter = [](OpGen &gen) {
  return !(gen.opName == "func.call" && funcPool.empty());
};

struct RegionGen {
  OpRegion *region;
  std::vector<float> weights;
  std::vector<OpGen> generators;
  std::vector<OpGenFilter> filters;

  RegionGen(OpRegion *region,
            std::vector<OpGenFilter> filters = std::vector<OpGenFilter>()) {
    this->region = region;
    if (filters.empty()) {
      filters.push_back(defaultFilter);
    }
    if (region->depth >= conf.regionDepthLimit) {
      filters.push_back(excludeFilter(regionedOps));
    }
    assert(region->depth <= conf.regionDepthLimit + 2);

    filters.push_back(funcCallExFilter);
    this->filters = filters;
  }

  // select from generators according to weights.
  OpGen *selectOpGenerator();

  OpGen *selectOpGeneratorDiverse();

  // for each region, applyFilters should be called exactly once
  void applyFilters() {
    for (auto op : conf.supported_ops) {
      auto opName = op.getFullName();
      auto opGen = getOpGen(opName);
      auto flag = true;
      for (auto filter : filters) {
        if (!filter(opGen)) {
          flag = false;
          break;
        }
      }
      if (flag) {
        weights.push_back(op.prob);
        generators.push_back(opGen);
      }
    }
  }

  std::vector<Operation *> apply(OpBuilder &builder, Location loc,
                                 int opNumLimit);

  std::vector<Operation *> applyAll(OpBuilder &builder, Location loc);
};

#endif // MLIR_FUZZ_REGIONGENERATION_H
