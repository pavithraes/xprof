# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "signal_provider",
    hdrs = ["signal_provider.h"],
    deps = [
        "//util/task:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//xprof/convert:repository",
        "@org_xprof//xprof/convert:xplane_to_tools_data",
    ],
)

cc_library(
    name = "smart_suggestion_rule",
    hdrs = ["smart_suggestion_rule.h"],
    deps = [
        ":signal_provider",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)
