load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "datasets",
    srcs = [
        "__init__.py",
        "cbench.py"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/envs/llvm:llvm_benchmark",
        "//compiler_gym/errors",
        "//compiler_gym/third_party/llvm",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ]
)