//
// Created by Stan Wang on 2023/2/3.
//

#ifndef MLIR_FUZZ_DIVERSITYCRETIRIA_H
#define MLIR_FUZZ_DIVERSITYCRETIRIA_H

#include "smith/json.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;

struct DiversityCriteria {
  // op coverage
  std::set<std::string> ops;

  std::map<std::string, std::set<std::string>> opNests;

  // map op to it's args defining op.
  std::map<std::string, std::set<std::string>> opConnections;

  DiversityCriteria(std::set<std::string> ops = std::set<std::string>(),
                    std::map<std::string, std::set<std::string>> opNests =
                        std::map<std::string, std::set<std::string>>(),
                    std::map<std::string, std::set<std::string>> opConnections =
                        std::map<std::string, std::set<std::string>>())
      : ops(ops), opNests(opNests), opConnections(opConnections) {}

  void insertOp(std::string op) { ops.insert(op); }

  void insertOpNest(std::string parent, std::string op) {
//    std::cout << parent << " { " << op + " } " << std::endl;
    if (opNests.find(parent) == opNests.end()) {
      auto emptySet = std::set<std::string>();
      auto p = std::make_pair(parent, emptySet);
      opNests.insert(p);
    }
    opNests[parent].insert(op);
  }

  void insertOpConnection(std::string from, std::string to) {
//    std::cout << to << " <- " << from << std::endl;
    if (opConnections.find(to) == opConnections.end()) {
      opConnections.insert(std::make_pair(from, std::set<std::string>()));
    }
    opConnections[to].insert(from);
  }

  void exportSelf(std::string path) {
    std::ofstream cov(path);
    assert(cov); // should check the existence of file path
    json data;
    for (auto op : ops) {
      data["covered_ops"].push_back(op);
    }
    for (auto it = opConnections.begin(); it != opConnections.end(); it++) {
      json connectionPair;
      connectionPair["to"] = it->first;
      for (auto to : it->second) {
        connectionPair["froms"].push_back(to);
      }
      data["covered_op_connections"].push_back(connectionPair);
    }
    for (auto it = opNests.begin(); it != opNests.end(); it++) {
      json nestPair;
      nestPair["parent"] = it->first;
      for (auto to : it->second) {
        nestPair["children"].push_back(to);
      }
      data["covered_op_nests"].push_back(nestPair);
    }
    cov << data.dump(4) << std::endl;
  }

  void import(std::string path) {
    std::ifstream f(path);
    if (!f) {
      std::ofstream covfile;
      covfile.open(path, std::ios::out | std::ios::app );
      assert(covfile.is_open() );
      covfile  << "{}" << std::endl;
      covfile.flush();
    }

    json data = json::parse(f);
    json coveredOps = data["covered_ops"];

    for (json::iterator it = coveredOps.begin(); it != coveredOps.end(); it++) {
      ops.insert(*it);
    }

    json coveredOpConnections = data["covered_op_connections"];
    for (json::iterator it = coveredOpConnections.begin();
         it != coveredOpConnections.end(); it++) {
      json connectionData = *it;
      std::string to = connectionData["to"];
      std::set<std::string> froms;
      json fromsData = connectionData["froms"];
      for (auto it1 = fromsData.begin(); it1 != fromsData.end(); it1++) {
        froms.insert(*it1);
      }
      opConnections.insert(std::make_pair(to, froms));
    }

    json coveredOpNests = data["covered_op_nests"];
    for (json::iterator it = coveredOpNests.begin(); it != coveredOpNests.end();
         it++) {
      json nestData = *it;
      std::string parent = nestData["parent"];
      std::set<std::string> children;
      json childrenData = nestData["children"];
      for (auto it1 = childrenData.begin(); it1 != childrenData.end(); it1++) {
        std::string op = *it1;
//        std::cout <<" ??? "   << op << std::endl;
        children.insert(*it1);
      }

      opNests.insert(std::make_pair(parent, children));
    }
  }
};

inline DiversityCriteria diversity;

#endif // MLIR_FUZZ_DIVERSITYCRETIRIA_H
