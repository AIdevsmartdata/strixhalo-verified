#!/bin/bash
# qwen38-flash-next.sh -- Qwen3.8-Flash-Next on Strix Halo (gfx1151), HIP runtime: 4 slots on one unified KV cache,
# 262k context, MTP + n-gram drafting, prompt cache on disk.
# Usage:
#   MODEL=/path/Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf \      (Unsloth UD-Q4_K_XL; production loads it merged into one file)
#   MMPROJ=/path/mmproj-F16.gguf \
#   MTP=/path/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf \        (MTP draft head, Q4_K_M)
#   MTP_IDS=/path/to/launch/mtp_vocab_65k_fr4096.bin \     (the 69,632 token ids the draft head scores, raw int32)
#   RUNTIME=/path/to/runtime/bin \                         (directory holding llama-server and the runtime libraries)
#   [CACHE_DIR=$HOME/.cache/qwen38-promptcache] [PORT=8081] \
#   ./qwen38-flash-next.sh [extra llama-server arguments]  (appended last, so they override the defaults below)
# Paths must be absolute: as in production, the launcher changes to $RUNTIME before starting the server.
#
# This is our production launcher of 8 November 2026 (runtime commit 9b20971f0): the same environment with the same
# values, and the same llama-server arguments in the same order. Differences: paths are variables, LD_LIBRARY_PATH is
# $RUNTIME + ROCm, the port is $PORT, and --chat-template-file is dropped (see the end). bench/check_launcher_parity.py
# compares the two files.
#
# Build requirements (compile time, not set here): GGML_HIP_NO_VMM=ON and GGML_CUDA_NO_VMM=ON (otherwise an assertion on
# the free order of the VMM pool), GGML_HIP_GRAPHS=OFF (otherwise "operation not permitted when stream is capturing").
# With them the model loads in 17 s instead of 6 min and MTP drafting works (08/10).
#
# In the comments: "bit-exact" = same outputs with the switch on and off, as checked by the gate that admitted it;
# "plain decode" = decode without drafting; dates are DD/MM 2026. The exports are grouped by production switchover,
# oldest first, in the order of the production launcher.
set -euo pipefail
: "${MODEL:?set MODEL}" "${MMPROJ:?set MMPROJ}" "${MTP:?set MTP}" "${MTP_IDS:?set MTP_IDS}" "${RUNTIME:?set RUNTIME}"
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/qwen38-promptcache}
PORT=${PORT:-8081}
for _v in MODEL MMPROJ MTP MTP_IDS RUNTIME CACHE_DIR; do
    case ${!_v} in /*) ;; *) echo "$_v must be an absolute path: ${!_v}" >&2; exit 1 ;; esac
done

# ===== 08/10 (move from Vulkan to HIP) and 10/10
# rocBLAS GEMMs go through hipBLASLt. ROCm variable, set by the GSQHalo base image; not measured on its own.
export ROCBLAS_USE_HIPBLASLT=1
# Q5_1 and ROCmFP4 experts also take the routed-compact MoE kernel. With TILE2X: prefill +1.2 %, decode +3.3 % (08/10).
export GGML_CUDA_MMQ_MOE_COMPACT=1
# MoE tile width sized on twice the average tokens per expert (AMD-Ecosystem #39) instead of the average.
export GGML_CUDA_MMQ_MOE_TILE2X=1
# Same-type device copies up to 1 MiB by a scalar kernel instead of hipMemcpyAsync. Bit-exact. With the other kernel
# changes of that switchover: MTP decode +1.3/+1.5 % (10/10).
export GGML_CUDA_CPY_KERNEL=1048576

# ===== 11/10, switchover 5: bit-exact decode levers. Together: plain decode +2.85 % at 0 / +2.88 % at 8k,
# MTP decode +2.07 % / +3.12 %.
# QSA indexer score chain (RELU, head sum, visibility) fused at decode with the prefill kernel. Bit-exact. Plain decode
# +0.6/+1.5 %.
export GGML_OPUS_IDXSCORE=1
# Q8_0 matvecs that quantize the activation once (ssm_out, attn_output) and R rows per wave (shared-expert down).
# Bit-exact. Plain decode +0.4/+1.4 %.
export GGML_OPUS_PREQ=1
# Long context: QSA sparse KV pack, one-workgroup block top-k, incremental key pooling for big batches. Bit-exact.
# +1 % at 64k.
export GGML_OPUS_LONG=1
# ... only from this cache depth (tokens); below it the original code runs.
export GGML_OPUS_LONG_MIN_KV=16384
# One Q8_1 copy of an activation, shared by every matvec that reads it. Bit-exact. MTP decode +0.5/+1.3 %.
export GGML_OPUS_Q8PROD=1
# RDNA3.5 Q8_0 multi-column kernels also used at 2 and 3 columns (MTP verification). Bit-exact. MTP decode +1.3 %.
export GGML_OPUS_VE1=1
# Q8_0 matvecs that read the same activation share one quantization and one launch (2-4 columns). Bit-exact.
export GGML_OPUS_VE1_GRP=1
# (09/10) Recurrent-state checkpoints taken inside the prefill pass instead of cutting the prompt before its end. Same
# output as without checkpoints: cold long4k top-1 +4.6 pts against cut prompts; prefill +1.6 % at 0, +6.7 % at 8k.
export LLAMA_CKPT_INBATCH=1

# ===== 11/10, switchover 8: bit-exact levers. Together: MTP decode +1.1 % at 0 / +1.7 % at 8k,
# plain decode +3.1/+3.4 %.
# Exact BF16 twins of small F32 weights (router, GDN alpha/beta, HC inject), read by small batches. Bit-exact.
export LLAMA_OPUS_BF16=1
# Loads issued before their first use in six kernels (technique of Gufo commit 7e4621c, MIT). Arithmetic unchanged.
export GGML_OPUS_G11=1
# Producer kernels also write the Q8_1 row their consumer needs (technique from Gufo 0.11, MIT). Bit-exact.
export GGML_OPUS_Q8FULL=1
# Per-layer embedding (PLE) rows gathered asynchronously, off the critical path. Bit-exact. SSD stalls at depth 0 seen
# in 3 runs out of 5 without it, in none with it.
export LLAMA_OPUS_PLEOFF=2
# VE1 groups reuse the Q8_1 copy of the producer / of Q8PROD instead of quantizing again. Bit-exact.
export GGML_OPUS_VE1_REUSE=1

# ===== 13/10, switchover 11
# Small host-to-device split inputs sent as one copy plus one scatter kernel, without stalling the host. Bit-exact.
# Plain decode about +1.7 % at 0 and +2-3 % at 8k, prefill about +5 % (11/10). Read once, at load time.
export GGML_OPUS_P1=2
# Non-finite logits: the output is discarded and the slot re-read from token 0 (idea from Strata #1892, MIT,
# rewritten). Does nothing otherwise.
export LLAMA_OPUS_NANGUARD=1
# Draft catch-up after a big batch takes the target's hidden rows on the GPU, without a host round trip. Bit-exact.
# MTP decode +1.29 % (12/10).
export LLAMA_OPUS_MTPCU=1

# ===== 13/10, switchover A2: Q4_K_M draft head (-md) + reduced draft vocabulary. MTP decode +8.16 % at T=0,
# +8.86 % at T=1.
# Token ids the draft head scores (65,536 frequent ones + 4,096 French ones). Verification stays over the full
# vocabulary: same output distribution, not bit-exact at T=0.
export LLAMA_MTP_HEAD_IDS="$MTP_IDS"

# ===== 16/10, switchover 15
# Bits 2+4+32: recycled checkpoint buffers, in-batch checkpoints built after the first token, template analysis
# cached. Bit-exact. Prefill +3-4 % together with WARMHOST (14/10).
export LLAMA_OPUS_PPFLOOR=38
# Warm turns: incremental tokenization, one template render less. Same tokens. Host time 42.2 -> 16.6 ms at 20k,
# 86.9 -> 28.8 ms at 60k (14/10).
export LLAMA_OPUS_WARMHOST=1
# Bit 4: [K,n1,n2] matvecs on one weight run as n1*n2 columns. Bit 8: wider routed-compact MoE grid, 2-7 % less time
# per step at 5-16 tokens, bit-exact (14/10).
export GGML_OPUS_MIDB=12
# ... bit 4 only up to 8 columns, where it stays on the MMVQ kernels: bit-exact.
export GGML_OPUS_MIDB_CHMAX=8
# While PLE rows are in use, one 4 KiB O_DIRECT read every 70 ms keeps the weights SSD out of APST sleep. Gather p90
# 7.1 -> 1.4 ms, one agent +0.5 to +2 % (14/10). Output unchanged.
export LLAMA_OPUS_PLE_KEEPALIVE=70
# Bits 1+2+8: MTP catch-up rows read on the GPU and queued before the verification ends; sampler snapshot without its
# 3 MB scratch buffer. Bit-exact. The bench did not resolve its speed gain.
export LLAMA_OPUS_CYCLEFUSE=11
# Bits 1+2: while other slots generate, a short prompt tail or the first chunk of a warm turn gets an iteration of its
# own. Admitted by decision: its own gate component was not green (16/10).
export LLAMA_OPUS_TOURPRIO=3

# ===== 17/10, switchover 16B (PPSMALL raised from 1 to 3, with MMB_MIN_T, on 31/10)
# Multi-slot verification at 9-16 columns. Bit 1: small F32 matrices as passes of <= 8 columns instead of hipBLASLt
# (not bit-exact; KLD inconclusive, accepted). Bit 2: ssm_out of the cohort in one pass (bit-exact). Bit 4: multi-slot
# drafts keep the reduced vocabulary. 4 agents 45.4 -> 55.2 tok/s (16/10).
export GGML_OPUS_BU4=7
# Bit 0: chunked GDN and in-batch checkpoints from 64 tokens instead of 256 (short-turn TTFT x0.86, 16/10). Bit 1: BF16
# WMMA mmb kernels from MMB_MIN_T rows instead of 512. Not bit-exact by construction (block-wise GDN, BF16 for int8).
export GGML_OPUS_PPSMALL=3
# ... prefill +6.0 % at 384 tokens, +19.7 % at 640, neutral from 2k to 32k; T=0 text identical on 20 prompts (31/10).
export GGML_OPUS_MMB_MIN_T=256
# Small dense Q8_0 MMQ (weights <= 4 MiB): tile width capped at 16 ...
export GGML_OPUS_MMQJ=16
# ... from 17 to 128 columns. Same K order, bit-exact; speed gate green (16/10), +0.3-0.6 % expected with one agent.
export GGML_OPUS_MMQJ_MAX=128
# Checkpoint spacing (--checkpoint-min-step) applied before the list is full, so long sessions do not pile up one
# 112 MiB checkpoint per user message.
export LLAMA_OPUS_CKPT_EAGER=1

# ===== 18/10, switchover 18
# Draft catch-up batches below LLAMA_MTP_KV_ONLY store K/V only when the head attends densely. Exact. 4 agents +1.42 %.
export LLAMA_OPUS_MTPKVM=1
# Bit 1: surplus checkpoint buffers freed by a separate thread. Bit 2: asymmetric pool margin. Host buffers only.
export LLAMA_OPUS_CKFREE=3

# ===== 19/10, switchover 20
# QSA block top-k (bit 1: several rows, bit 2: one row) in one kernel instead of a full hipCUB sort; same set as the
# stable radix sort. Exact. 4 agents x1.000.
export GGML_OPUS_QSATOPK=3

# ===== 20/10, switchover 23
# KV cache really full: never cut inside a slot's speculative block; free the biggest generating slot cleanly. Removes
# the HTTP 500 "speculative batch index ... not inside the current sub-batch" (upstream #24840).
export LLAMA_OPUS_KVFULL=1

# ===== 20/10, switchover 25
# MoE passes of 5-16 columns by a vector kernel grouped by expert (structure idea from Gufo 0.15, MIT). 0 wrong rows
# in 269,824 checks; changes some single-agent outputs. 4 agents x1.032.
export GGML_OPUS_MOEVEC=1
# ... smallest batch it takes (tokens).
export GGML_OPUS_MOEVEC_MIN=5
# ... largest batch it takes (tokens).
export GGML_OPUS_MOEVEC_MAX=16

# ===== 20/10, switchover 25RR
# T > 0: fresh draw when a speculative checkpoint is replayed (the original code reused a draw the verification had
# already spent). T=0 unchanged. Required by LEVIATHAN.
export LLAMA_OPUS_REPLAY_RNG=1

# ===== 20/10, switchover 21E
# RAM prompt cache: evict the entry cheapest to recompute, keep the biggest recent session, refuse a new state that
# costs more than it gives back.
export LLAMA_OPUS_EVIC_COST=1

# ===== switchover P18
# A batch starting at position 0 is paired with a zero hidden row, not with the last row of the slot's previous
# request. Same output as a fresh process (20/10); speed unchanged.
export LLAMA_OPUS_MTPRESET=1

# ===== 23/10
# Draft cap 3 between 4,096 and 12,288 tokens of depth, 2 elsewhere. +12.3 % at 8k, +0.6 % at 0, 0.0 % at 20k,
# -1.4 % at 32k (23/10).
export LLAMA_OPUS_NMXD=4096:12288:3:2
# 23/10: native context -c 262144, with --ctx-checkpoints 2 and --cache-ram 1024 (see the arguments below).
# The runtime's own libraries first, then ROCm.
export LD_LIBRARY_PATH="$RUNTIME:${ROCM_PATH:-/opt/rocm}/lib"
# Optional, needs root: holds the GPU clock at its high DPM level, as production does (the clock floor of our Vulkan
# launcher). Without write access to sysfs the loop does nothing.
for _d in /sys/class/drm/card*/device; do
    { [ -w "$_d/power_dpm_force_performance_level" ] &&
        echo high > "$_d/power_dpm_force_performance_level"; } 2>/dev/null || true
done

# ===== 29/10, B29: four draft-path levers. 4 slots: decode at 32k 39.9 -> 45.4 tok/s (+12.2 %), +7.0 % at 131k,
# neutral at 0 and 8k, prefill +3.6 % at 32k, same memory. Texts differ from 8k on; next-token KL against the runtime
# without them 0.00000 (code) / 0.00001 (long4k), top-1 40/40 (29/10).
# Exact speculative sampling at T > 0: the draft is drawn from q over 64 candidates, kept with min(1, p/q), exact
# residual (derived from Gufo's mtp_sampling.hpp, MIT). One agent about +13 % (26/10).
export LLAMA_OPUS_LEVIATHAN=1
# After the MTP draft, prompt-lookup copy on a sampled match of >= 24 tokens, at most 8 tokens (adapted from Gufo's
# PromptLookup, MIT). About +3 % (26/10).
export LLAMA_OPUS_COPIE24=1
# ... also inside multi-slot cohorts, at equal width (1 = replace). 8 agents +1.6 % (26/10).
export LLAMA_OPUS_COPIE24_COHORTE=1
# At most 16 verification columns per pass: the cohort's draft width drops as more slots generate. 8 agents +5.3 %,
# +14 % with COPIE24_COHORTE (26/10).
export LLAMA_OPUS_DRAFTLOAD=16

# ===== 06/11, W3
# One agent: a third MTP draft token, kept only when an online estimate of the head's confidence says it pays
# (threshold 33 %; idea from Gufo's MtpLengthController, MIT). Decode 51.3 -> 54.7 tok/s, next-token KL 0.000000.
export LLAMA_OPUS_MTP_W3=1
# At least 3 recurrent rollback snapshots, required by W3 (a rejected 3-token draft is rolled back by snapshot).
export LLAMA_OPUS_RS_SEQ=3

# llama-server arguments, as in production:
#   no --parallel: the server picks 4 slots on one unified KV cache, the configuration of the numbers above (29/10);
#   --spec-type draft-mtp,ngram-simple: MTP head + n-gram lookup, decode +92 % (46.9 -> 79.1 tok/s, 08/10);
#   --spec-draft-n-max 2: +5.3 % at 0 / +7.0 % at 8k on ordinary text against 3 (09/10); NMXD and W3 adjust it;
#   --spec-uniform-draft 2 --spec-uniform-draft-sampling: one draft width for the whole cohort, also when sampling
#     (output distribution unchanged): 3 agents 31.7 -> 46.7 tok/s, 4 agents 38.9 -> 59.5 (11/10);
#   --spec-draft-mtp-vocab 69632: length of the MTP_IDS list;
#   -lm dio -lzm on-direct: weights loaded with direct I/O; PLE rows read from the file on demand with pread();
#   -b 8192 -ub 8192: prefill +7 % against 4096 (08/10);
#   -c 262144 --cache-ram 1024 --ctx-checkpoints 2: the model's native context, with the RAM prompt cache and the
#     checkpoints lowered to keep >= 5 GB free (lowest MemAvailable 10.59 GB at 204,800 tokens); prefill at least that
#     of the 131,072 setting, decode x0.98-1.00 (23/10);
#   --cache-dir --cache-dir-max 16384: prompt cache on disk, 16 GiB; a 60k session resumes after a restart in 1.13 s
#     instead of 38.7 s, decode -0.16 % (23/10);
#   --chat-template-kwargs reasoning_effort=medium: the server-side default; requests can override it.
# Production also passes --chat-template-file with a copy of the chat template that adds a short efficiency
# instruction only when the client sends no system message (8 checked tasks: 17.42 -> 13.83 s, still 16/16, 06/11).
# That template is not shipped here, so the model's own template is used.
cd "$RUNTIME" || exit 1
exec ./llama-server \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  -md "$MTP" \
  --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2 --spec-draft-p-min 0.3 \
  --spec-uniform-draft 2 --spec-uniform-draft-sampling --spec-draft-mtp-vocab 69632 \
  --host 127.0.0.1 --port "$PORT" --metrics --no-webui \
  -ngl 999 -fa on -c 262144 \
  -lm dio -lzm on-direct -t 4 -b 8192 -ub 8192 \
  --cache-ram 1024 --ctx-checkpoints 2 --jinja \
  --cache-dir "$CACHE_DIR" --cache-dir-max 16384 \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  "$@"
