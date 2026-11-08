#!/bin/bash
# build.sh -- build the strixhalo-verified HIP runtime (llama.cpp fork for AMD Strix Halo, gfx1151).
# Usage: ./build.sh [work_dir]      (default: ./llama.cpp-strixhalo)
#
# What it does:
#   1. clones the public GSQHalo.cpp fork (Aristo94/GSQHalo.cpp, itself a llama.cpp fork for Qwen3.8-Flash-Next on
#      Strix Halo) at a pinned commit;
#   2. applies runtime/patches/0001-strixhalo-hip-b29.patch (the whole delta of our runtime, 180 commits listed in
#      runtime/COMMITS.md, including the halo-box/strix-llama.cpp pull requests we merged);
#   3. checks that the resulting source tree is byte-for-byte the one we run (git tree hash);
#   4. configures with the exact CMake options of the measured build (runtime/cmake-options.txt) and builds llama-server.
#
# Needs: git, cmake >= 3.21, make, a C++17 compiler, ROCm with gfx1151 support (measured with ROCm 7.14 built by TheRock,
# commit 418cd5f6, AMD clang 23.0.0git;
# ROCM_PATH=/opt/rocm714 here), including rocBLAS and hipBLASLt; ~6 GB of disk, and about 2.5 GB of RAM per build job
# (JOBS=8 is a safe default on a 128 GB Strix Halo with nothing else loaded; do not build while a 100 GB model is loaded).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-$PWD/llama.cpp-strixhalo}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
JOBS=${JOBS:-8}
BASE_REPO=https://github.com/Aristo94/GSQHalo.cpp
BASE_COMMIT=5fc881b114c1ea130f5df6a30a98be2f8d397de6   # branch gsqhalo, 3 Oct 2026 ("docs: server examples with thinking at xhigh")
EXPECTED_TREE=a5106d68620893a2de651be3ec7571f8d338ebaa  # source tree of our production runtime (commit 9b20971f0)

[ -x "$ROCM_PATH/lib/llvm/bin/amdclang++" ] || { echo "amdclang++ not found under ROCM_PATH=$ROCM_PATH" >&2; exit 1; }

if [ ! -d "$WORK/.git" ]; then
    git clone --filter=blob:none "$BASE_REPO" "$WORK"
fi
cd "$WORK"
git fetch -q origin gsqhalo 2>/dev/null || true
git checkout -q --detach "$BASE_COMMIT"
git reset -q --hard "$BASE_COMMIT"
git clean -q -fdx -e build
git apply --index "$HERE/patches/0001-strixhalo-hip-b29.patch"
TREE=$(git write-tree)
if [ "$TREE" != "$EXPECTED_TREE" ]; then
    echo "source tree $TREE differs from the measured runtime ($EXPECTED_TREE): stopping" >&2
    exit 1
fi
echo "source tree OK ($TREE): identical to the measured runtime"

export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH" HIP_PATH="$ROCM_PATH"
mapfile -t OPTS < <(grep -v '^\s*#' "$HERE/cmake-options.txt" | grep -v '^\s*$')
cmake -S . -B build -G "Unix Makefiles" \
    -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=c++ \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/amdclang++" \
    "${OPTS[@]}"
cmake --build build -j "$JOBS" --target llama-server llama-gguf-split
echo "runtime ready: $WORK/build/bin  (use it as RUNTIME for launch/qwen38-flash-next.sh)"
