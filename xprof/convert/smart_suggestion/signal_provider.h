/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_PROVIDER_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_PROVIDER_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/tool_data_provider.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"
#include "util/task/status_macros.h"

namespace tensorflow {
namespace profiler {

// Concrete class to provide signals from a SessionSnapshot.
class SignalProvider {
 public:
  explicit SignalProvider(std::unique_ptr<ToolDataProvider> tool_data_provider)
      : tool_data_provider_(std::move(tool_data_provider)) {}

  // Example signal getters:
  absl::StatusOr<double> GetHbmUtilization() const {
    ASSIGN_OR_RETURN(const auto* overview_page,
                     tool_data_provider_->GetOverviewPage());
    return overview_page->analysis()
        .memory_bw_utilization_relative_to_hw_limit_percent();
  }

  absl::StatusOr<double> GetMxuUtilization() const {
    ASSIGN_OR_RETURN(const auto* overview_page,
                     tool_data_provider_->GetOverviewPage());
    return overview_page->analysis().mxu_utilization_percent();
  }

 private:
  std::unique_ptr<ToolDataProvider> tool_data_provider_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_PROVIDER_H_
