load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//learning/grp/pynball:build_defs.bzl", "ipynb_deps")
load("//research/colab:build_defs.bzl", "colab_binary")

package(
    default_visibility = ["//visibility:public"],
)

ipynb_deps(
    name = "simplified_seanet_deps",
    srcs = ["simplified_seanet.ipynb"],
    deps = [
        "//pyglib:resources",
        "//third_party/py/flax:core",
        "//third_party/py/flax/training:train_state",
        "//third_party/py/jax",
        "//third_party/py/librosa",
        "//third_party/py/matplotlib:matplotlib_base",
        "//third_party/py/numpy",
        "//third_party/py/optax",
        "//third_party/py/resampy",
        "//third_party/py/scipy",
        "//third_party/py/tqdm",
    ],
)

colab_binary(
    name = "tpu_colab",
    deps = [
        ":simplified_seanet_deps",
        "//learning/brain/google/xla:tensorflow_jellyfish_deps",
        "//learning/brain/research/jax:tpu_support",
    ],
)

genmpm(
    name = "tpu_colab_mpm",
    srcs = [":tpu_colab"],
    temporal = True,
)

pytype_strict_library(
    name = "viking_jax",
    srcs = ["viking_jax.py"],
    deps = [
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_contrib_test(
    name = "viking_jax_test",
    srcs = ["viking_jax_test.py"],
    deps = [
        ":viking_jax",
        "//third_party/py/absl:app",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "viking_parity_test",
    srcs = ["viking_parity_test.py"],
    deps = [
        ":viking_jax",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)
