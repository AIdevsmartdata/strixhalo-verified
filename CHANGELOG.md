# Runtime changes (late August → 30 September 2026)

Everything below sits on top of [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), branch
`vulkan/qwen4exp-rocmfpx` (commit `322e5cdf4cc`), which brought Qwen3.8-Flash-Next (`qwen4exp`) and the ROCmFPx types
to llama.cpp's Vulkan backend, including sparse attention (QSA) and the per-layer n-gram table read from SSD. The 54
patches in `runtime/patches/` are the complete delta.

The work was done over five weeks by AI coding agents (DeepSeek, GPT, Claude, and the local Qwen3.8 itself) under my
direction, on one Strix Halo box. This is a list of the main changes, with the measured effect where one was recorded.
Figures marked **(30/09)** were measured on the final runtime; the others are the numbers recorded when the change was
adopted, and later re-measurements are given when they disagree.

## ✅ Correctness and fidelity

| change | effect |
|---|---|
| Hyper-connection injection scale `2·sigmoid`: the graph applied `SCALE(2.0)` twice (`4·sigmoid`) and the fused `hc_combine` kernel reproduced both | KL divergence vs the full-precision reference **0.116** at `2·sigmoid`; on the same binary 1.5σ gives 0.315 and 2.5σ 0.220; the previous `4·sigmoid` build (older Vulkan backend) measured 0.478 **(30/09)** |
| Recurrent state: per-slot rollback planes (upstream #28123) + host-side zeroing of a new sequence's state row | the Vulkan graph optimizer read the state before zero-filling it in all 36 delta-net layers, so a request started from the previous one's state: on the 5th repeated request the first-token probability fell to 0.48 where the fixed runtime gives 0.9992. Fixed: repeated requests give the same greedy output and match the healthy runtime to the 4th digit **(30/09)** |
| MTP drafting hygiene: pending hidden state reset at sequence start (open upstream PR #28333); drafted tokens after an end-of-generation token are no longer accepted | no drafted tokens past the end of a turn (they forced a state restore on the next turn) **(30/09)** |
| Delta-net q/k normalisation `x·rsqrt(Σx² + ε)` as in the official implementation (upstream #28068) | parity with the reference implementation |
| QSA sparse-attention fixes: pooled-key shadowing (segfault), indexer keys written below the budget, indexer sums by chunks (upstream #28023), `seq_cp` of indexer keys (#27941) | long-context runs without crashes |
| Grammar-compatible backend sampling + reasoning budget with backend sampling | tool calls and reasoning budgets stay correct with `--backend-sampling` (with the default sampler chain the backend path was only really taken once the 30/09 grammar fast path landed) |
| Context checkpoint created on slot restore | a restored slot (pi's `kv-restore`) reuses its cache instead of re-reading the whole prompt |

## ⚡ Agent-loop speed (30/09)

| change | effect |
|---|---|
| Streaming loads (`MOVNTDQA`, AVX-512/AVX2/SSE4.1) for CPU reads of device memory | the write-combined carve-out read at ~0.4 GB/s with `memcpy`; state checkpoint read **516 → 41 ms**, first token after a short tool result **1.51 → 0.58 s** |
| Grammar fast path: draw without the constraint, check the single token, full-vocabulary pass only on rejection (same semantics as upstream's default `grammar_first = false`) | the default sampler chain cannot run on the GPU, so every verified position applied the tool-call grammar to all 248,320 tokens; tool-call decode **25 → 48–55 tok/s**, identical output at temperature 0 (diagnosed and written by the local Qwen3.8 through pi) |
| Reuse of an identical context checkpoint instead of reading the state back again | saved ~0.5 s per reuse before the streaming-load fix; ~40 ms now |
| Per-request MTP parameters (`speculative.n_max`, `.p_min`, `.n_min`) | tuning without restarting the server |

Agent loop, 12 shell commands, pi's request parameters: **11.1 → 15.3 tokens per wall-clock second**, one short turn
**3.06 → 1.60 s** (from the afternoon's v2 candidate to the final runtime).

## 🏎️ Decode and prefill kernels (Vulkan)

| change | recorded effect |
|---|---|
| Fused `UNARY(gelu/silu/sigmoid/softplus) + MUL` | +6.5 % decode when adopted; re-measured 25/09 on the current quant: no measurable effect (+0.16 % when removed). Kept as measured |
| Fused hyper-connection combine chain; `RMS_NORM_MUL` fusion on the HC norms (`LLAMA_HC_NORM_DIRECT`) | fewer dispatches per token |
| Delta-net zero-copy at decode: recurrent-cache `CPY` and input-state `GET_ROWS` elided, hardened state-index transport | fewer dispatches per token |
| Zero row of the recurrent state written with `FILL` instead of an in-place `SCALE` (`LLAMA_RS_FILL_ZERO`) | 1.56 → 0.37 ms per graph (write-only) |
| MoE `MUL_MAT_ID` medium/large tiles at 256 threads (`GGML_VK_MMID_WG256`) | prefill +3.6 to +12.8 % when adopted, decode unchanged; within ±1 % in a later re-measurement |
| MoE router-weight scale fused into the prefill `MUL_MAT_ID` epilogue (`GGML_VK_MMID_SCALE_EPILOGUE`) | prefill +2.1 to +7.8 % when adopted (13/09); within ±1 % in a later re-measurement |
| Large workgroups for AMD small-m mat-vecs (`GGML_VK_DMMV_LARGE_AMD`, gate 8192) | kept as measured; neutral (±1 %) on the current quant |
| Aligned, vectorised matmul variant extended to small heights (`GGML_VK_MATMUL_ALIGN_SMALLM`) | the hyper-connection projections (m = 4 and 1) ran at 27–112 GFLOPS and took ~4 % of prefill |
| f16 KV made contiguous before flash attention | prefill at 64k **244 → 452 tok/s** (+85 %) |
| Flash-attention prefill: P-hoist + query-major shared memory (ported from Nathanw1014), 3 row groups (48 rows) per workgroup | prefill at 64k ~450 → 484 tok/s (+7.5 %) |
| Radix top-k: 4 histogram banks, unrolled counting | −25 % kernel time in the model (347 → 259 µs per call), about +0.5 % decode, bit-identical |
| QSA decode gather + radix top-k (ported from apepojken) | sparse attention at decode |
| MTP draft head: reduced 32k-token output vocabulary (`LLAMA_MTP_HEAD_IDS`), speculative checkpoints kept on the GPU | cheaper drafting (host-side speculative checkpoints were 4× slower) |
| KV cells: non-contiguous restore (upstream #27991), n-gram history lookup by position (#28040) | faster restores (#27991), faster long-context decode (#28040) |

## 🗑️ Measured and rejected (so you don't pay for them twice)

The fused weighted MoE combine (numerically inexact at large batch: wrong experts picked on long prompts, several
different outputs at temperature 0; the code stays in the patches and `GGML_VK_DISABLE_MOE_WEIGHTED=1` is required).
Sparse attention during prompt processing (non-deterministic top-k ties; decode keeps it), KV cache in q4_0 (fails the
long-reasoning check), two server slots (25.7–27.9 tok/s in aggregate against 39.8–40.2 for one stream, −30 %), wave32
MoE mode (degenerate output), several flash-attention tile tunings (quality loss), custom quantization mixes (none beat
Unsloth's UD-Q4_K_XL on reference KL divergence).

The patches also carry experimental switches that the launcher leaves off (coalesced ROCmFP4 loads, mat-vec row counts,
extra fusions, tracing). `launch/qwen38-flash-next.sh` is derived from the measured production launcher: same kernels and
flags, minus a private n-gram steering table, the GPU clock floor and the server-side reasoning-effort default, plus
`--min-p 0.0`.

## 🔍 Known issues (found by our own code review of the last two commits, 30/09)

| issue | impact | status |
|---|---|---|
| Host-side zeroing of the recurrent state calls `ggml_backend_tensor_memset`, which asserts on backends whose buffers do not implement it (OpenCL, Hexagon, one SYCL buffer type) | recurrent/hybrid models abort on those backends; Vulkan and CPU are not affected | fix planned: fall back to the in-graph clear |
| Checkpoint reuse matches on position only: with two checkpoints one token apart and a new prompt diverging exactly at the second one, a stale checkpoint can be kept | wrong recurrent state after a later restore; needs that exact layout, never seen in our agent-loop logs | fix planned: erase checkpoints that cover the first diverging token |
| No `MFENCE` before the streaming loads | none observed (a blocking fence wait sits between the GPU write and the read); Mesa issues one | hardening planned |
| A per-request `speculative.n_max` below the launch `n_min` disables drafting instead of clamping `n_min` | slower request, no wrong output | fix planned |
| `__builtin_cpu_supports` in the streaming-load path needs compiler-rt on Windows clang (MSVC target) | link error on that toolchain | guard planned |

The grammar fast path follows upstream's default semantics (sample, check, fall back on rejection): at temperature > 0
with truncating samplers the distribution is the rejection-sampling one, not grammar-first; token probabilities reported
for accepted positions are the unconstrained ones; `mirostat` updates twice on a rejected position. All three are
upstream behaviours.

## 🧰 Tools shipped here

`bench/`: the agent-turn benchmark and the tool-call speed probe used for the numbers above.
