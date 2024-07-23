//
// Created by Stan Wang on 2022/10/31.
//
#include "smith/RegionGeneration.h"
#include "smith/generators/OpGeneration.h"

OpGen *RegionGen::selectOpGenerator() {
  assert(weights.size() == generators.size());
  if (weights.empty()) {
    applyFilters();
  }
  int id = getWeightedRandomIndex(weights);
  return &generators[id];
}

OpGen *RegionGen::selectOpGeneratorDiverse() {
  assert(weights.size() == generators.size());

  if (weights.empty()) {
    applyFilters();
  }

  std::set<std::string> opsToCover;
  std::set<std::string> opNestsToCover;

  for (auto op : conf.supported_ops) {
    if (diversity.ops.find(op.getFullName()) == diversity.ops.end()) {
      opsToCover.insert(op.getFullName());
    }
  }
  for (auto op : opNests[region->parent_op]) {
    if (diversity.ops.find(op) == diversity.ops.end()) {
      opNestsToCover.insert(op);
    }
  }

  for (size_t i = 0; i < generators.size(); i++) {
    auto opName = generators[i].opName;
    if (opsToCover.find(opName) != opsToCover.end()) {
      weights[i] += 5 * priority_base;
    }
    if (opNestsToCover.find(opName) != opNestsToCover.end()) {
      weights[i] += 5 * priority_base;
    }
  }

  assert(weights.size() == generators.size());

  size_t id = getWeightedRandomIndex(weights);
  if (weights.empty()) {
    return nullptr;
  }
  return &generators[id];
}

std::vector<Operation *> RegionGen::apply(OpBuilder &builder, Location loc,
                                          int opLenLimit) {
  int length = 0;
  std::vector<Operation *> operations;
  if (region->tmpl != nullptr) {
    for (auto op : region->tmpl[region->parent_op]) {
      region->cur_child = op;
      std::string opName;
      if (op.is_string()) { // op has no region
        opName = op;
      } else if (op.is_object()) {
        opName = op.items().begin().key();
      }
      if (operators.find(opName) == operators.end()) {
        llvm::errs() << opName << " not found in supported operations";
        continue ;
      }
      auto gen = operators[opName];
      auto operation = gen.apply(builder, loc, *region);
      if (operation) {
        operations.push_back(operation);
      }
    }
  } else {
    while (true) {
      OpGen *opGen;
      if (diverse) {
        opGen = selectOpGeneratorDiverse();
      } else {
        opGen = selectOpGenerator();
      }
      if (!opGen) {
        continue;
      }
      auto operation = opGen->apply(builder, loc, *region);
      if (operation) {
        operations.push_back(operation);
        auto current_op = operation->getName().getStringRef().str();
        diversity.insertOp(current_op);
        diversity.insertOpNest(region->parent_op, current_op);
        for (const auto &item : operation->getOperands()) {
          if (!item.getDefiningOp()) {
            continue;
          }
          auto from = item.getDefiningOp()->getName().getStringRef().str();
          diversity.insertOpConnection(from, current_op);
        }
      }
      length++;
      if (length >= opLenLimit) {
        // generate terminator
        break;
      }
    }
  }
  return operations;
}

// for debug use
std::vector<Operation *> RegionGen::applyAll(mlir::OpBuilder &builder,
                                             mlir::Location loc) {
  std::vector<Operation *> operations;
  for (auto opGen : generators) {
    auto operation = opGen.apply(builder, loc, *region);
    if (operation) {
      operations.push_back(operation);
    }
  }
  return operations;
}
