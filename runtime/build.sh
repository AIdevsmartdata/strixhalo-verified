#!/bin/bash
# build.sh -- build the strixhalo-verified runtime (llama.cpp fork, Vulkan) from its public base + the patch series.
# Usage: ./build.sh [work_dir]      (default: ./llama.cpp-strixhalo)
# Needs: git, cmake >= 3.21, a C++17 compiler, Vulkan headers + glslc (Ubuntu: libvulkan-dev glslc), ~5 GB of disk.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-$PWD/llama.cpp-strixhalo}
BASE_REPO=https://github.com/LaurentZuijdwijk/llama.cpp
BASE_COMMIT=322e5cdf4cc        # branch vulkan/qwen4exp-rocmfpx (Qwen3.8-Flash-Next support, ROCmFPx types)

if [ ! -d "$WORK/.git" ]; then
    git clone --filter=blob:none "$BASE_REPO" "$WORK"
fi
cd "$WORK"
git fetch origin vulkan/qwen4exp-rocmfpx 2>/dev/null || true
git checkout -q -B strixhalo-verified "$BASE_COMMIT"
git -c user.name=build -c user.email=build@localhost am -q --committer-date-is-author-date "$HERE"/patches/*.patch

cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-server llama-perplexity -j "$(nproc)"
echo "runtime ready: $WORK/build/bin  (use it as RUNTIME for launch/qwen38-flash-next.sh)"
