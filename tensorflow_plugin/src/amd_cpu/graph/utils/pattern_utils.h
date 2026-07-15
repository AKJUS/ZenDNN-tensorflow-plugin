/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_PATTERN_UTILS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_PATTERN_UTILS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {
namespace graph {
namespace utils {

//------------------------------------------------------------------------------
// A pattern can be defined by the following grammar. Here, op_type is any valid
// op name in the TensorFlow.
//
//    leaf_pattern ::= `{` op_type `}`
//    pattern ::= leaf_pattern |
//                `{` op_type `,` `{` pattern `,` ... `,` pattern `}` `}`
//
// (1) For example, the following pattern syntax describes a pattern for
// _FusedConv2D (Conv2D + BiasAdd + Relu). Note that "*" means any type of op.
//
//  {"Relu",
//    {
//      "BiasAdd",
//      {
//        {"Conv2D"},
//        {"*"}
//      }
//    }
//  }
//
// The syntax above has a root ("Relu") and children (inputs), where each child
// is a sub-pattern. Graph pattern matcher finds a match for the given pattern
// syntax in a graph and returns a set of matched nodes.
//
// (2) In order to match a DAG with a given root, we extend pattern syntax with
// labels. For example, a frequently found pattern in Deep Learning models is a
// residual block like below.
//
//    Placeholder  Const
//          |        |
//    +-----+-----+  |
//    |           |  |
//    |           v  v
//    |          Conv2D   Const
//    |            |        |
//    |            v  v-----+
//    |          BiasAdd
//    |            |
//    v v----------+
//   AddV2
//
// As shown above, it is the same input node (Placeholder) consumed by both
// AddV2 and Conv2D. This constrained can be put as labels in the following
// augmented pattern syntax.
//
//  {"AddV2", "my_add",
//    {
//      {"*", "my_residual_input"},
//      {"BiasAdd", "my_bias_add",
//        {
//          {"Conv2D", "my_conv",
//            {
//              {"*", "my_residual_input"},
//              {"*", "my_filter"}
//            }
//          },
//          {"*", my_bias"}
//        }
//      }
//    }
//  }
//
// Note that the same label "my_residual_input" is used to tell that it is a
// child of both "AddV2" and "Conv2D". Labels are arbitrary strings to associate
// with the nodes to be matched as well as to uniquely identify those nodes.
//
// (3) The motivation for a grammar based pattern matching in grappler is to
// make easy for finding fusion pattern in the remapper. A subgraph that
// matches a given pattern, however, is not fusable if any of the matched node,
// that will be removed as a part of fusion, has a consumer outside the matched
// subgraph. In order to check for such type of external dependencies, we
// further extend pattern syntax by prospective action (NodeStatus) on the
// matched nodes as shown below. This helps cross checking the nodes to be
// removed with the nodes matched intially.
//
//  {"AddV2", "my_add", NodeStatus::kReplace,
//    {
//      {"*", "my_residual_input", NodeStatus::kRemain},
//      {"BiasAdd", "my_bias_add", NodeStatus::kRemove,
//        {
//          {"Conv2D", "my_conv", NodeStatus::kRemove,
//            {
//              {"*", "my_residual_input", NodeStatus::kRemain},
//              {"*", "my_filter", NodeStatus::Remain}
//            }
//          },
//          {"*", my_bias", NodeStatus::kRemain}
//        }
//      }
//    }
//  }
//------------------------------------------------------------------------------

// Pattern matcher recursively matches child subpatterns. The direction
// for children could be toward node's input (fanins) or outputs (fanouts).
enum class MatchingDirection { kFollowInputs, kFollowOutputs };

// Action for each node in the set of matched nodes for a given pattern.
enum class NodeStatus { kRemain, kRemove, kReplace };

// TODO(plugin): Support multiple roots by making them children of a single
// virtual root.
struct OpTypePattern {
  string op;
  string label;
  NodeStatus node_status;
  std::vector<OpTypePattern> children;

  OpTypePattern& AddInput(const OpTypePattern& parent) {
    children.push_back(parent);
    return *this;
  }

  OpTypePattern& AddNSameInput(const OpTypePattern& parent) {
    OpTypePattern new_parent = parent;
    new_parent.op += "*";

    children.push_back(new_parent);
    return *this;
  }

  OpTypePattern& AddNDifferentInput(const OpTypePattern& parent) {
    OpTypePattern new_parent = parent;
    new_parent.op += "*";
    new_parent.label += "*";

    children.push_back(new_parent);
    return *this;
  }

  string DebugString() const {
    string result = "{(op: " + op + ", " + "label: " + label + "), {";
    for (const OpTypePattern& child : children) {
      result += child.DebugString() + ",";
    }
    result += "}}";
    return result;
  }
};

void DumpPattern(const OpTypePattern& pattern, std::string path);

// This is a helpful recursive structure that keeps one-to-one mapping of
// pattern syntax to the matched nodes. User can call DebugString to see what
// has been matched so far and where is the failing point.
struct NodeViewMatch {
  MutableNodeView* node_view = nullptr;
  std::vector<NodeViewMatch> children;

  string DebugString() const {
    string result = "{";
    if (node_view == nullptr) {
      result += "Non-Matched-Node}";
      return result;
    } else {
      result += node_view->node()->DebugString();
      result += ", {";
      for (const NodeViewMatch& child : children) {
        result += child.DebugString() + ",";
      }
      result += "}}";
      return result;
    }
  }

  void Clear() {
    for (NodeViewMatch& child : children) {
      child.Clear();  // child is an object.
    }
    children.clear();  // children is a vector.
    if (node_view != nullptr) {
      node_view = nullptr;
    }
  }
};

template <MatchingDirection DIRECTION = MatchingDirection::kFollowInputs>
class SubGraphMatcher {
 public:
  explicit SubGraphMatcher(MutableGraphView* graph_view)
      : graph_view_(graph_view) {}

  // If a given pattern is matched, this function returns true as well as the
  // matched node and remove node info is populated.
  // When non-null, *matched_pattern_but_unsafe_to_remove is set to true iff
  // DoesOpTypePatternMatch succeeded but IsSafeNodesToRemove failed (external
  // data fanout from a node slated for removal). On pattern mismatch it is set
  // to false. When null, the parameter is ignored.
  bool GetMatchedNodes(const OpTypePattern& pattern,
                       const std::unordered_set<string>& nodes_to_preserve,
                       MutableNodeView* node_view,
                       std::map<string, int>* matched_nodes_map,
                       std::set<int>* remove_node_indices,
                       bool fanin_checking = true,
                       bool ignore_controlled_fanouts = false,
                       bool ignore_controlling_fanins = false,
                       bool* matched_pattern_but_unsafe_to_remove = nullptr);

 private:
  MutableGraphView* graph_view_;
  std::map<string, int> node_label_to_index_;
  std::set<int> matched_node_indices_;
  std::set<int> remove_node_indices_;
  std::unique_ptr<NodeViewMatch> match_ = nullptr;

  bool DoesOpTypePatternMatch(const OpTypePattern& pattern,
                              MutableNodeView* node_view, NodeViewMatch* match,
                              bool fanin_checking = true,
                              bool ignore_controlled_fanouts = false,
                              bool ignore_controlling_fanins = false);

  // This function should be called after the pattern matcher has found
  // potential matched nodes (i.e. when DoesOpTypePatternMatch returns "true").
  // It performs a sanity check if the candidate nodes for removal in subgraph
  // fusion is indeed safe to remove.
  bool IsSafeNodesToRemove(
      const std::unordered_set<string>& nodes_to_preserve) {
    for (const auto& node_idx : remove_node_indices_) {
      auto node_view = graph_view_->GetNode(node_idx);
      // Check if the node to be removed is in the nodes to be preserved.
      string node_name = node_view->GetName();
      if (nodes_to_preserve.count(node_name) > 0) {
        zendnnl::error_handling::apilog_info(
            "Pattern: fusion blocked: remove-node ", node_name, " (",
            node_view->node()->op(), ") is listed in nodes_to_preserve");
        return false;
      }
      // Traverse all the Regular Fanouts. Fanouts are stored as vector of
      // vector, std::vector<std::vector<MutableFaninView>>. Note that
      // a MutableNodeView's fanouts are stored as a nested vector of
      // MutableFaninView type.
      auto fanouts_by_ports = node_view->GetRegularFanouts();
      int leak_count = 0;
      for (int out_port = 0;
           out_port < static_cast<int>(fanouts_by_ports.size()); ++out_port) {
        for (const auto& fanout : fanouts_by_ports[out_port]) {
          if (!matched_node_indices_.count(fanout.node_index())) {
            ++leak_count;
            auto* consumer_view = graph_view_->GetNode(fanout.node_index());
            const string consumer_name = consumer_view && consumer_view->node()
                                             ? consumer_view->node()->name()
                                             : string("?");
            const string consumer_op = consumer_view && consumer_view->node()
                                           ? consumer_view->node()->op()
                                           : string("?");
            // producer:out_port -> consumer:in_port identifies the exact graph
            // edge that leaves the matched fusion subgraph.
            zendnnl::error_handling::apilog_info(
                "Pattern: fusion leak edge producer=", node_name,
                " op=", node_view->node()->op(), " out_port=", out_port,
                " -> consumer=", consumer_name, " op=", consumer_op,
                " in_port=", fanout.index(),
                " (consumer not in matched subgraph; cannot remove producer)");
          }
        }
      }
      if (leak_count > 0) {
        zendnnl::error_handling::apilog_info(
            "Pattern: IsSafeNodesToRemove=false: ", leak_count,
            " leak edge(s) from remove-node ", node_name);
        return false;
      }
    }
    return true;
  }
};

}  // namespace utils
}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_PATTERN_UTILS_H_
