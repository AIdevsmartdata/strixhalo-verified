#!/bin/bash
# qwen38-flash-next.sh -- Qwen3.8-Flash-Next on Strix Halo: Vulkan, one slot, 262k context, MTP drafting.
# Usage:
#   MODEL=/path/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
#   MMPROJ=/path/mmproj-F16.gguf \
#   MTP=/path/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf \
#   RUNTIME=/path/to/llama.cpp-strixhalo/build/bin \
#   ./qwen38-flash-next.sh [--host 0.0.0.0 --port 8081 ...]      (extra arguments override the defaults below)
# Host prompt cache capped at 6 GiB (--cache-ram 6144). Unlimited (-1) let the server keep every evicted conversation in
# host RAM: with a 138k-token agent context it reached 25 GB on top of the model and the kernel OOM killer took the box down.
# Derived from the measured production launcher: same kernels and flags, minus a private n-gram steering table, the GPU
# clock floor (power_dpm_force_performance_level=high) and a server-side reasoning-effort default, plus --min-p 0.0.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?set MODEL}" "${MMPROJ:?set MMPROJ}" "${MTP:?set MTP}" "${RUNTIME:?set RUNTIME}"

# --- Vulkan kernels
export GGML_VK_DMMV_LARGE_AMD=8192           # large workgroups for small-m mat-vecs on AMD (m <= 8192; kept as measured)
export GGML_VK_DMMV_LARGE_AMD_COLS=8         # ... also for the multi-token batches of MTP draft/verify
export GGML_VK_DMMV_LARGE_AMD_COLS_MAX_M=8   # ... but only for very small heights there (larger ones regress)
export GGML_VK_FUSE_UNARY_MUL=1              # fused activation * gate (+6.5 % decode when adopted; neutral on this quant)
export GGML_VK_MATMUL_ALIGN_SMALLM=1         # aligned matmul path for the tiny hyper-connection projections
export GGML_VK_MMID_WG256=1                  # 256-thread MoE matmul tiles (prefill +3.6-12.8 % when adopted)
export GGML_VK_DISABLE_MOE_WEIGHTED=1        # REQUIRED: the fused weighted MoE combine is inexact at large batch
export GGML_VK_MMID_SCALE_EPILOGUE=1         # MoE router-weight scale fused into the prefill matmul epilogue
export GGML_VK_TOPK_NHIST=4                  # radix top-k: 4 histogram banks (bit-identical)
export GGML_VK_TOPK_UNROLL=4                 # radix top-k: unrolled counting (both: -25 % kernel time, bit-identical)
export GGML_VK_FA_KV_CONTIG=1                # f16 KV made contiguous before flash attention: prefill 64k 244 -> 452 tok/s
export GGML_VK_FA_MR=3                       # 3 flash-attention row groups per workgroup: prefill 64k ~450 -> 484 tok/s

# --- model graph (qwen4exp)
export LLAMA_HC_NORM_DIRECT=1                # RMS_NORM + MUL fusion on the hyper-connection norms
export LLAMA_RS_FILL_ZERO=1                  # zero row of the recurrent state written with FILL (write-only)
export LLAMA_QSA_COMPRESS_RATIO=4            # sparse attention (QSA): compressed indexer keys, ratio 4
export LLAMA_QSA_CACHE_COMPRESSED=1
export LLAMA_QSA_SKIP_PREFILL_SEL=1          # dense attention during prompt processing (sparse top-k ties are not deterministic)
export LLAMA_QSA_RAW_GATHER=1                # decode gather copies the selected KV rows in the cache type (f16), no cast
export LLAMA_QSA_SEL_BLOCK=1
export LLAMA_QSA_BLK_VIEW=1
export LLAMA_QSA_FINALIZE=1

# --- MTP drafting
export LLAMA_MTP_HEAD_IDS="$HERE/mtp_head_gen_32k.bin"   # draft head scores 32k frequent tokens instead of 248k
export LLAMA_MTP_ALL_TEMPS=1                 # draft at every temperature (default: greedy requests only)
export LLAMA_SPEC_CHECKPOINT_ON_DEVICE=1     # speculative checkpoints stay on the GPU (host copies were 4x slower)

export LD_LIBRARY_PATH="$RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$RUNTIME/llama-server" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --no-mmproj-offload \
  --metrics \
  -ngl 999 \
  -fa on \
  -fit off \
  --no-mmap \
  -c 262144 \
  --parallel 1 \
  -t 24 \
  -b 16384 \
  -ub 2048 \
  --cache-ram 6144 \
  --slots \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --ngram-on-disk \
  --ngram-cache 512 \
  --no-ngram-direct-io \
  --jinja \
  --no-webui \
  --temp 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0.0 \
  --backend-sampling \
  --reasoning-preserve \
  --spec-type draft-mtp \
  --spec-draft-type-k q8_0 \
  --spec-draft-type-v q8_0 \
  --spec-draft-model "$MTP" \
  --spec-draft-ngl 999 \
  --spec-draft-n-max 4 \
  --spec-draft-n-min 1 \
  --spec-draft-p-min 0.75 \
  --spec-draft-backend-sampling \
  --host 127.0.0.1 --port 8081 --alias qwen3.8-flash-next "$@"
