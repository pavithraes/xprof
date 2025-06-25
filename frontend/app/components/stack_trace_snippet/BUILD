load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "message",
    srcs = [
        "message.ts",
    ],
    assets = [
        ":message_css",
        "message.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
    ],
)

sass_binary(
    name = "message_css",
    src = "message.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
