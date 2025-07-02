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
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/xplane_to_tools_data.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"
#include "util/task/status_macros.h"

namespace tensorflow {
namespace profiler {

// Concrete class to provide signals from a SessionSnapshot.
class SignalProvider {
 public:
  explicit SignalProvider(const SessionSnapshot& session_snapshot)
      : session_snapshot_(session_snapshot) {}

  // Example signal getters:
  absl::StatusOr<double> GetHbmUtilization() const {
    ASSIGN_OR_RETURN(const auto* overview_page, GetOverviewPage());
    return overview_page->analysis()
        .memory_bw_utilization_relative_to_hw_limit_percent();
  }

  absl::StatusOr<double> GetMxuUtilization() const {
    ASSIGN_OR_RETURN(const auto* overview_page, GetOverviewPage());
    return overview_page->analysis().mxu_utilization_percent();
  }

 private:
  // Helper function to get or generate OverviewPage data.
  // TODO(b/429210120): Create a data API to encapsulate extracting tool data.
  absl::StatusOr<const OverviewPage*> GetOverviewPage() const{
    if (!overview_page_cache_) {
      ASSIGN_OR_RETURN(std::string overview_page_str,
                       ConvertMultiXSpacesToToolData(session_snapshot_,
                                                     "overview_page", {}));
      auto overview_page = std::make_unique<OverviewPage>();
      if (!overview_page->ParseFromString(overview_page_str)) {
        return absl::InternalError("Failed to parse OverviewPage proto");
      }
      overview_page_cache_ = std::move(overview_page);
    }
    return overview_page_cache_.get();
  }

  const SessionSnapshot& session_snapshot_;
  mutable std::unique_ptr<OverviewPage> overview_page_cache_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_PROVIDER_H_
