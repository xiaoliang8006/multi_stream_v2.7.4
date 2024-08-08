package(licenses = ["notice"])

py_library(
    name = "cuda",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/cuda:multi_stream",
    ],
)

py_library(
    name = "multi_stream",
    srcs = ["multi_stream.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)