# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "signal_provider",
    hdrs = ["signal_provider.h"],
    deps = [
        ":tool_data_provider",
        "//util/task:status",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
    ],
)

cc_library(
    name = "smart_suggestion_rule",
    hdrs = ["smart_suggestion_rule.h"],
    deps = [
        ":signal_provider",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)

cc_library(
    name = "tool_data_provider",
    hdrs = ["tool_data_provider.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
    ],
)

cc_library(
    name = "tool_data_provider_impl",
    hdrs = ["tool_data_provider_impl.h"],
    deps = [
        ":tool_data_provider",
        "//util/task:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//xprof/convert:multi_xplanes_to_op_stats",
        "@org_xprof//xprof/convert:op_stats_to_overview_page",
        "@org_xprof//xprof/convert:repository",
    ],
)
