/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_stream_util.h"

#include <fstream>
#include <regex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {
namespace gpu_stream_util {

bool ReadRuleFromFile(
    std::unordered_map<int, std::vector<std::regex>>& stream_assign_rule) {
  // Determine a suitable stream to use.
  static const string stream_from_file = [] {
    string stream_from_file;
    TF_CHECK_OK(ReadStringFromEnvVar("TF_STREAM_FROM_FILE",
                                     /*default_val=*/"", &stream_from_file));
    return stream_from_file;
  }();
  if (!stream_from_file.empty()) {
    std::ifstream file(stream_from_file);
    if (file.is_open()) {
      string line;
      while (std::getline(file, line)) {
        std::vector<string> kv = tensorflow::str_util::Split(line, ":");
        std::vector<string> pattern = tensorflow::str_util::Split(kv[1], ",");
        std::vector<std::regex> re_pattern;
        for (string& p : pattern) {
          re_pattern.push_back(std::regex(p));
        }
        VLOG(2) << "Read " << re_pattern.size() << " rules for stream "
                << atoi(kv[0].c_str());
        stream_assign_rule[atoi(kv[0].c_str())] = std::move(re_pattern);
      }
      file.close();
    } else {
      VLOG(2) << "Read stream assignment rule from file failed! Use the "
                 "default rule...";
      return false;
    }
  } else {
    VLOG(2) << "Read stream assignment rule from file failed! Use the default "
               "rule...";
    return false;
  }
  VLOG(2) << "Successfully read stream assignment rule from file: "
          << stream_from_file;
  return true;
}

int GetFromProto(Node* n, int num_streams) {
  const AttrValue* stream_attr_value = n->attrs().FindByString("_stream_id");
  if (stream_attr_value == nullptr) return 0;
  return stream_attr_value->i();
}

int GetFromRule(
    Node* n, int num_streams,
    std::unordered_map<int, std::vector<std::regex>>& stream_assign_rule) {
  for (const auto& item : stream_assign_rule) {
    for (auto& r : item.second) {
      if (std::regex_match(n->name(), r)) {
        return item.first;
      }
    }
  }
  return 0;
}

void GetStreamWaitList(
    const Graph* graph, DeviceContextID* device_context_id,
    std::vector<std::pair<std::string, int>>& stream_wait_list) {
  for (Node* n : graph->nodes()) {
    const int node_id = n->id();
    VLOG(2) << "Node " << node_id << " " << n->type_string() << " " << n->name()
            << " " << n->out_edges().size() << " outputs";
    int stream_id = (*device_context_id)[node_id];
    std::unordered_map<int, int> out_stream_counter;
    for (const Edge* e : n->out_edges()) {
      int dst_stream_id = (*device_context_id)[e->dst()->id()];
      if (stream_id != dst_stream_id) {
        // Wait on this node because of different streams.
        if (out_stream_counter.find(dst_stream_id) ==
            out_stream_counter.end()) {
          out_stream_counter[dst_stream_id] = 1;
        } else {
          ++out_stream_counter[dst_stream_id];
        }
      }
    }
    // Get the {input_node, stream} pairs that should wait more than once.
    for (auto& item : out_stream_counter) {
      VLOG(2) << "Node " << node_id << " " << n->type_string() << " "
              << n->name() << " on stream " << stream_id << " has "
              << item.second << " outputs assigned on stream " << item.first;
      if (item.second > 1) {
        stream_wait_list.push_back({n->name(), item.first});
      }
    }
  }
}

void GetNodeDenpendencyOnStream(
    const Graph* graph, DeviceContextID* device_context_id,
    std::unordered_map<std::string, std::set<std::pair<int, std::string>>>&
        need_sync_node_deps) {
  static const bool ms_key_info_log = [] {
    bool ms_key_info_log;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_MULTI_STREAM_KEY_INFO_LOG",
                                   /*default_val=*/false, &ms_key_info_log));
    return ms_key_info_log;
  }();

  std::vector<Node*> order;
  GetReversePostOrder(*graph, &order);
  for (Node* n : order) {
    const int node_id = n->id();
    int stream_id = (*device_context_id)[node_id];
    std::set<std::pair<int, std::string>> controls = {};
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        // it is a control dependency input
        auto input_name = e->src()->name();
        int input_stream_id = (*device_context_id)[e->src()->id()];

        auto input_type = e->src()->type_string();
        if (input_type == "NoOp" || input_type == "Const") {
          if (need_sync_node_deps.find(input_name) !=
              need_sync_node_deps.end()) {
            auto predecessor_controls = need_sync_node_deps[input_name];
            for (const auto& item : predecessor_controls) {
              if (item.first != stream_id)
                controls.insert(std::make_pair(item.first, item.second));
            }
          }
        } else {
          if (input_stream_id != stream_id)
            controls.insert(std::make_pair(input_stream_id, input_name));
        }
        if (ms_key_info_log) {
          VLOG(0) << "Need sync control deps: Node: " << n->name()
                  << ", stream: " << stream_id << ", depend on " << input_name
                  << ", on stream: " << input_stream_id;
        }
      }
    }
    if (controls.size() > 0) {
      need_sync_node_deps[n->name()] = controls;
    }
  }
}

Status AssignStreams(const Graph* graph, const AssignStreamsOpts& opts,
                     std::unordered_map<int, int>* node_to_stream_id) {
  VLOG(1) << "AssignStreams";
  Status status;

  static const bool ms_key_info_log = [] {
    bool ms_key_info_log;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_MULTI_STREAM_KEY_INFO_LOG",
                                   /*default_val=*/false, &ms_key_info_log));
    return ms_key_info_log;
  }();

  static const bool allow_forward_propagation = [] {
    bool allow_forward_propagation;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_MULTI_STREAM_ALLOW_FORWARD_PROPAGATION",
                                   /*default_val=*/false,
                                   &allow_forward_propagation));
    return allow_forward_propagation;
  }();

  // Sanity check arguments.
  if (graph == nullptr)
    status.Update(errors::InvalidArgument("Bad graph argument supplied."));
  if (node_to_stream_id == nullptr) {
    status.Update(
        errors::InvalidArgument("Bad node_to_stream_id argument supplied."));
  }
  if ((opts.max_streams < 1) || (opts.send_stream >= opts.max_streams) ||
      (opts.recv_stream >= opts.max_streams) ||
      (opts.const_stream >= opts.max_streams) ||
      (opts.compute_stream >= opts.max_streams)) {
    status.Update(errors::InvalidArgument("Bad graph argument supplied."));
  }
  TF_RETURN_IF_ERROR(status);

  // Topologically sort the nodes.
  std::vector<Node*> order;
  GetReversePostOrder(*graph, &order);
  if (VLOG_IS_ON(2)) {
    for (Node* n : order) {
      const int node_id = n->id();
      VLOG(2) << "Node " << node_id << " " << n->type_string() << " "
              << n->name() << " " << n->in_edges().size() << " inputs";
      for (const Edge* e : n->in_edges()) {
        VLOG(2) << "  Edge from " << e->src()->id() << "  " << e->src()->name()
                << " fanout " << e->src()->out_edges().size();
      }
    }
  }

  std::unordered_map<int, std::vector<std::regex>> stream_assign_rule;
  bool has_rule = ReadRuleFromFile(stream_assign_rule);
  int highest_stream_id = -1;
  for (Node* n : order) {
    VLOG(3) << "Inspecting node " << n->DebugString();
    const int node_id = n->id();
    const string& op = n->type_string();

    // Determine a suitable stream to use.
    // There are two types of stream groups.
    // 1. stream group 0, the default group.
    // 2. stream group 1-N: Ops which exploit multi-stream to parallel
    // accelerate.
    int stream_id = 0;

    std::string stream_assign_debug;

    // Assign to the stream according to proto.
    stream_id = GetFromProto(n, opts.max_streams);
    // Override the stream by rules from file. The rules have higher priority
    // than the proto field. Only valid when the rule of this node != 0
    // because 0 is the default stream group and you never assign node to it.
    if (has_rule) {
      int rule_stream_id = GetFromRule(n, opts.max_streams, stream_assign_rule);
      stream_id = rule_stream_id;
    }

    // Forward propagation.
    if (allow_forward_propagation) {
      static int dice = 0;
      if (stream_id == 0) {
        std::unordered_set<int> src_stream_id;
        for (const Edge* e : n->in_edges()) {
          int tmp = (*node_to_stream_id)[e->src()->id()];
          if (tmp > 0) {
            src_stream_id.insert(tmp);
          }
        }
        std::vector<int> src_stream_id_vec;
        for (auto item : src_stream_id) {
          src_stream_id_vec.push_back(item);
        }
        if (!src_stream_id_vec.empty()) {
          if (src_stream_id_vec.size() > 1) {
            stream_id = src_stream_id_vec[dice % src_stream_id_vec.size()];
            VLOG(1) << "Sources of the node have been assigned with "
                    << src_stream_id_vec.size() << " streams, such as "
                    << src_stream_id_vec[0] << " and " << src_stream_id_vec[1]
                    << ", but the node has not been assigned manually: "
                    << n->DebugString() << ". Use stream group " << stream_id;
            ++dice;
          } else {
            stream_id = src_stream_id_vec[0];
          }

          if (ms_key_info_log)
            VLOG(0) << " Switch form stream 0 to " << stream_id;
        }
      }
    }

    if (ms_key_info_log) {
      VLOG(0) << "Op: " << n->def().op() << ", Name: " << n->name()
              << ", Stream: " << stream_id << "." << stream_assign_debug;
    }

    (*node_to_stream_id)[node_id] = stream_id % opts.max_streams;
    highest_stream_id = std::max(stream_id, highest_stream_id);
    VLOG(3) << "Assign node " << node_id << " (" << n->name() << ") to stream "
            << (*node_to_stream_id)[node_id];
  }

  // TODO: we need to be careful how we handle stream assignment for send and
  // recv nodes to assure performance. There are 4 cases:
  // 1. CPU graph Send & GPU graph Recv: Send and Recv should be assigned to the
  // same stream, otherwise it may launch HtoD in an unexpected threadpool. The
  // stream should be the same as the successor node of Recv to avoid
  // unnecessary StreamWait.
  // 2. GPU graph Send & CPU graph Recv: Send and Recv should be assigned to the
  // same stream, otherwise it may launch DtoH in an unexpected threadpool. The
  // stream should be the same as the predecessor node of Send to avoid
  // unnecessary StreamWait.
  // 3. HostSend->Recv with dependency: Send and Recv should be assigned to the
  // same stream.
  // 4. Send->HostRecv with dependency: Same as 2.
  // Right now we don't have a way to assign streams to nodes on the CPU graph.
  for (Node* n : order) {
    const int node_id = n->id();
    if (IsSend(n)) {
      // For case 2, 3, and 4.
      for (const Edge* e : n->in_edges()) {
        (*node_to_stream_id)[node_id] = (*node_to_stream_id)[e->src()->id()];
        break;
      }
    } else if (IsHostRecv(n)) {
      // For case 4.
      for (const Edge* e : n->in_edges()) {
        if (IsSend(e->src())) {
          (*node_to_stream_id)[node_id] = (*node_to_stream_id)[e->src()->id()];
          break;
        }
      }
    } else if (IsRecv(n)) {
      // For case 1 and 3.
      bool is_hostsend_recv = false;
      for (const Edge* e : n->in_edges()) {
        if (IsHostSend(e->src())) {
          is_hostsend_recv = true;
          break;
        }
      }
      if (is_hostsend_recv) {
        // For case 3.
        for (const Edge* e : n->in_edges()) {
          (*node_to_stream_id)[node_id] = (*node_to_stream_id)[e->src()->id()];
          break;
        }
      } else {
        // For case 1.
        for (const Edge* e : n->out_edges()) {
          (*node_to_stream_id)[node_id] = (*node_to_stream_id)[e->dst()->id()];
          break;
        }
      }
    }
  }

  VLOG(1) << "Identified " << highest_stream_id << " candidate streams for "
          << order.size() << " nodes.";

  return Status::OK();
}

}  // namespace gpu_stream_util
}  // namespace tensorflow
