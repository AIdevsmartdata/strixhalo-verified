# Runtime changes

## HIP runtime (8 October → 8 November 2026)

On 8 October production moved from our Vulkan fork to a HIP/ROCm engine, [GSQHalo](https://github.com/Aristo94/GSQHalo.cpp)
(a llama.cpp fork for Strix Halo), from its commit `5fc881b114c`. The production runtime of 8 November is commit
`9b20971f0`: 180 commits (merges excluded) on top of that base, listed in `runtime/COMMITS.md`.
`launch/qwen38-flash-next.sh` is the production launcher of the same day with its paths turned into variables and
without our custom chat template; `bench/check_launcher_parity.py` checks that both files set the same environment and
pass the same arguments.

Almost every change sits behind an environment switch that is off by default. As a rule a switch reached production
only through a gate whose rules were written before the measurement: identical outputs with the switch on and off
(called bit-exact below) or, for a switch that changes the arithmetic or the drafts, a check of the output distribution
(KL divergence against the previous runtime or the BF16 reference, or a statistical test on sampled outputs), plus speed
in alternated runs (ABBA) on the production configuration. Each production change is called a switchover. Exceptions to
the rule are stated in the tables.

The work was done over one month by AI coding agents (Claude, GPT, Kimi and DeepSeek) under my direction, on one Strix
Halo box. Effects are given as recorded on the date in parentheses (DD/MM); when a later measurement corrected one, the
later value is given. "1 agent", "4 agents", "8 agents" = that many concurrent requests; "plain decode" = decode without
drafting; "MTP decode" = decode with drafting. Numbers from different benches are not comparable with each other: the
decode figure of 8 October (81.8 tok/s at 32k) comes from a code task with about 90 % of the drafts accepted, the
39.9 → 45.4 tok/s of 29 October from another bench. <!-- src: CHANTIER-QWEN38.md §3 (09/10); PILOTAGE_GUFO_2610/JOURNAL.md 29/10 14:45 -->

### 🏎️ Engine and kernels

| change | effect |
|---|---|
| Vulkan → HIP engine (GSQHalo), same machine, same model, same battery | prefill at 32k **539.5 → 1550 tok/s**, decode at 32k **41.2 → 81.8 tok/s**, agent turn 65.5 → 23.9 s; quality battery passed (two output-hash locks, middle-of-context test at 11k 6/6, zebra 3/3, hard24 24/25, retrieval at 14k, vision) **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| Build with `GGML_HIP_NO_VMM=ON`, `GGML_CUDA_NO_VMM=ON`, `GGML_HIP_GRAPHS=OFF` | without them: an assertion on the free order of the VMM pool, and "operation not permitted when stream is capturing"; with them the model loads in 17 s instead of 6 min and MTP drafting works **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| Engine update: seven commits on the mmb (BF16 WMMA) kernels and the hyper-connection gate | prefill +28 % **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| `-b 8192 -ub 8192` instead of 4096 | prefill +7 % **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| MoE tiling: Q5_1 / ROCmFP4 experts admitted to the routed-compact kernel, tile sized on twice the tokens per expert (`GGML_CUDA_MMQ_MOE_COMPACT`, `GGML_CUDA_MMQ_MOE_TILE2X`) | prefill +1.2 %, decode +3.3 % **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| Five decode commits from halo-box/strix-llama.cpp, among them the GDN decode-chain fusion that never fired (its matcher expected `L2_NORM`, qwen4exp builds `RMS_NORM + SCALE`) | plain decode +7 % (Gufo's protocol, three concordant runs), identical greedy outputs, KL divergence identical to the 6th digit **(09/10)** | <!-- src: CHANTIER-QWEN38.md 09/10 00:30 and 09:52 -->
| Fused copies, GLU at 2-8 columns, GDN verification, hyper-connection up-mix, grouped F32 matvecs | MTP decode +3.8 % at 0 / +4.2 % at 8k, identical outputs **(09/10)** | <!-- src: CHANTIER-QWEN38.md 09/10 17:53 -->
| Small device copies by a scalar kernel instead of `hipMemcpyAsync` (`GGML_CUDA_CPY_KERNEL`), with other decode kernels | MTP decode +1.3 / +1.5 %, identical outputs **(10/10)** | <!-- src: CHANTIER-QWEN38.md 10/10 20:36 -->
| Switchover 5, bit-exact: indexer score fused at decode (`IDXSCORE`), Q8_0 matvecs with the activation quantized once (`PREQ`), long-context paths from 16k (`LONG`), one Q8_1 copy per activation (`Q8PROD`), Q8_0 verification kernels at 2-4 columns (`VE1`, `VE1_GRP`), DPP lane shuffles on RDNA3 (compiled in) | together: plain decode +2.85 % at 0 / +2.88 % at 8k, MTP decode +2.07 % / +3.12 %, same outputs and drafts; alone: VE1 MTP +1.3 %, IDXSCORE +0.6 / +1.5 %, PREQ +0.4 / +1.4 %, LONG +1 % at 64k, Q8PROD +0.5 / +1.3 % **(11/10)** | <!-- src: CHANTIER-QWEN38.md 11/10 01:30 and 02:33 -->
| Switchover 8, bit-exact: exact BF16 twins of small F32 weights (`LLAMA_OPUS_BF16`), loads issued early in six kernels (`G11`, technique of Gufo commit 7e4621c, MIT), Q8_1 rows written by their producer kernels (`Q8FULL`, technique from Gufo 0.11, MIT), asynchronous PLE gather (`PLEOFF=2`), Q8_1 reuse in VE1 groups (`VE1_REUSE`) | MTP decode +1.1 % at 0 / +1.7 % at 8k, plain decode +3.1 / +3.4 %, same outputs on every run; SSD stalls at depth 0 in 3 runs of 5 before, none after **(11/10)** | <!-- src: CHANTIER-QWEN38.md 11/10 21:35 -->
| Small host-to-device split inputs grouped into one copy and one scatter kernel (`GGML_OPUS_P1=2`), bit-exact | plain decode about +1.7 % at 0 and +2-3 % at 8k, prefill about +5 %, MTP decode about +1.1 % **(11/10)**; in production on 13/10 | <!-- src: CHANTIER-QWEN38.md 11/10 22:36; 13/10 10:30 -->
| `MIDB=12` with `MIDB_CHMAX=8`: matvecs of several channels on one weight in one pass (up to 8 columns), wider routed-compact MoE grid | bit 8: 2-7 % less time per step at 5-16 tokens, 100 % bit-identical on 2,794 positions; bit 4 bit-identical up to 8 columns **(14/10)** | <!-- src: CHANTIER-QWEN38.md 14/10 17:10 and 17:53 -->
| Small dense Q8_0 MMQ tiles capped at 16 columns (`MMQJ=16`, up to 128 columns) | bit-exact, speed gate green **(16/10)**; +0.3-0.6 % with 1 agent expected by the analysis **(15/10)** | <!-- src: CHANTIER-QWEN38.md 15/10 14:49; 16/10 08:51 -->
| Multi-slot verification at 9-16 columns (`BU4=7`): small F32 matrices as passes of ≤ 8 columns instead of hipBLASLt (bit 1), `ssm_out` of the cohort in one pass (bit 2), reduced draft vocabulary kept for multi-slot drafts (bit 4) | 4 agents **45.4 → 55.2 tok/s (+21.8 %)**; without bit 1 +3.9 % (inconclusive); bit 1 changes the arithmetic, its KL check was inconclusive and it was accepted by decision **(16/10)** | <!-- src: CHANTIER-QWEN38.md 16/10 11:38 -->
| QSA block top-k in one kernel instead of a full hipCUB sort (`QSATOPK=3`), same set as the stable radix sort | exact; 4 agents ×1.000 **(19/10)** | <!-- src: CHANTIER-QWEN38.md 19/10 11:16 -->
| MoE vector kernel grouped by expert for passes of 5-16 columns (`MOEVEC=1`; structure idea from Gufo 0.15, MIT, no code copied) | 4 agents ×1.032 (cycle +4.39 %); 0 wrong rows in 269,824 checks; 6 of 32 single-agent output hashes change **(20/10)** | <!-- src: CHANTIER-QWEN38.md 19/10 22:02; 20/10 00:20 and 08:12 -->

### 🎯 MTP and speculative decoding

| change | effect |
|---|---|
| `--spec-type draft-mtp,ngram-simple` (MTP head plus n-gram lookup) | decode **46.9 → 79.1 tok/s (+92 %)** **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| `--spec-draft-n-max 2` instead of 3 | MTP decode on ordinary text +5.3 % at 0, +7.0 % at 8k (acceptance 66.9 % against 52.7 %) **(09/10)** | <!-- src: CHANTIER-QWEN38.md 09/10 22:10 -->
| One draft width for the whole cohort, also when sampling (`--spec-uniform-draft 2 --spec-uniform-draft-sampling`; policy concept from Gufo's `MtpBatchController`, MIT, no code copied) | aggregate decode at T=0.6: 3 agents 31.7 → 46.7 tok/s (+47 %), 4 agents 38.9 → 59.5 (+53 %) **(11/10)** | <!-- src: CHANTIER-QWEN38.md 11/10 04:20 and 05:06 -->
| Q4_K_M draft head (`-md`) and a 69,632-token output list for it: 65,536 frequent tokens + 4,096 French ones (`LLAMA_MTP_HEAD_IDS`, `--spec-draft-mtp-vocab 69632`) | MTP decode **+8.16 % [+6.42 ; +9.86] at T=0, +8.86 % [+6.77 ; +11.32] at T=1**; exact in distribution (law test), not bit-exact at T=0 **(13/10)**. The 4,096 French tokens raise the coverage of generic French from 82.6-85.2 % to 97.4-98.0 %, code unchanged **(11/10)**. Re-checked: a Q8_0 head gains 4.5 pts of acceptance but loses 10-15 % at depth 0 **(22/10)**; Q4_K_M 43.95 / 43.60 tok/s against 43.17 / 45.21 for the older Q8_0 head (+1.35 GB) and 41.91 / 42.84 for the newer one: Q4_K_M kept **(05/11)** | <!-- src: CHANTIER-QWEN38.md 11/10 02:15, 13/10 13:42, 22/10 23:30; PILOTAGE_GUFO_2610/JOURNAL.md 05/11 23:10 -->
| Draft catch-up after big batches takes the target's hidden rows on the GPU (`MTPCU`) | MTP decode +1.29 %, bit-exact **(12/10)** | <!-- src: CHANTIER-QWEN38.md 12/10 19:52 -->
| Draft catch-up batches store K/V only when the head attends densely (`MTPKVM=1`) | exact; 4 agents +1.42 % **(18/10)** | <!-- src: CHANTIER-QWEN38.md 18/10 17:02 -->
| Draft cap by depth (`NMXD=4096:12288:3:2`: 3 between 4,096 and 12,288 tokens, 2 elsewhere) | +12.3 % at 8k (43.15 against 38.42 tok/s, acceptance 60.8 against 56.0 %), +0.6 % at 0, 0.0 % at 20k, −1.4 % at 32k (control drift −0.7 %); same output at depth 0 **(23/10)**. A fixed cap of 3 gave +13.1 % at 8k but −2.6 % at 32k and −10.0 % at 0 **(22/10)**; a cap of 4 lost 6.7 % at 8k in the same window, 21.5 % in a 2k-16k window **(29/10)** | <!-- src: CHANTIER-QWEN38.md 22/10 23:30, 23/10 13:55; PILOTAGE_GUFO_2610/JOURNAL.md 29/10 17:35 -->
| Exact speculative sampling at T > 0 (`LEVIATHAN`): the draft is drawn from q over 64 candidates, accepted with min(1, p/q), exact residual; derived from Gufo's `mtp_sampling.hpp` (MIT, licence in `common/leviathan-LICENSE`) | 1 agent +14.9 / +14.3 / +10.8 % on three seed sets (about +13 %), acceptance 0.56-0.58 → 0.63-0.66, 18 of 18 prompt × seed pairs faster **(26/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 26/10 12:40 and 13:04 -->
| Prompt-lookup copy after the MTP draft on a sampled match of ≥ 24 tokens, at most 8 tokens (`COPIE24`, adapted from Gufo's `PromptLookup`, MIT) | +1.9 %, then +3.9 % on a second seed set (about +3 %), acceptance 0.63 → 0.75 **(26/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 26/10 15:12 and 15:37 -->
| Copies inside multi-slot cohorts (`COPIE24_COHORTE=1`, adapted from Gufo's `PromptLookup` / `MtpBatchController`, MIT) and at most 16 verification columns per pass (`DRAFTLOAD=16`) | 8 agents: DRAFTLOAD 73.2 → 77.1 tok/s (+5.3 %), COHORTE +1.6 %, both 72.0 → 82.2 (+14 %) **(26/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 26/10 19:03 and 20:50 -->
| The four draft-path levers above, measured on the production configuration (4 slots, unified KV cache) | decode at 32k **39.9 → 45.4 tok/s (+12.2 %)**, +7.0 % at 131k, neutral at 0 and 8k, prefill +3.6 % at 32k, same memory; texts differ from 8k on; next-token KL against the previous production 0.00000 (code) and 0.00001 (long4k, max 0.00039), top-1 40/40 on each **(29/10)**. Put in production under a 24-hour watch instead of after a full law gate (see known issues) | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 29/10 14:45, 15:57 and 19:20 -->
| Third MTP draft token with one agent, kept when an online estimate of the head's confidence says it pays (`MTP_W3`, with `RS_SEQ=3`; idea from Gufo's `MtpLengthController`, MIT, and hipfire's `native_rate`, Apache-2.0; no code copied) | decode **51.3 → 54.7 tok/s (+6.6 %)**, acceptance 81.9 → 82.8 %, drafts per step 2.11 → 2.86, tokens per step 2.72 → 3.37, quality 16/16 against 15/16; law gate on 80 positions: top-1 agreement 100 %, **KL 0.000000** **(06/11)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 06/11 09:25 -->

### 📖 Prefill

| change | effect |
|---|---|
| In-batch checkpoints (`LLAMA_CKPT_INBATCH`): recurrent-state snapshots taken inside the chunked prefill instead of cutting the prompt | prefill +1.6 % at 0, +6.7 % at 8k, +6.5 % at 32k; quality: see correctness **(09/10)** | <!-- src: CHANTIER-QWEN38.md 09/10 23:23 -->
| Recycled checkpoint buffers, checkpoints built after the first token, template analysis cached once (`PPFLOOR=38`); incremental tokenization of warm turns (`WARMHOST`) | bit-exact; prefill +3 to +4 % **(14/10)**; host time of a warm turn 42.2 → 16.6 ms at 20k, 86.9 → 28.8 ms at 60k, 12 of 12 replays **(14/10)** | <!-- src: CHANTIER-QWEN38.md 14/10 17:31 and 19:44 -->
| Chunked GDN and in-batch checkpoints from 64 tokens instead of 256 (`PPSMALL` bit 0) | short-turn TTFT ×0.860, 1 agent ×1.009; KL check inconclusive, accepted by decision **(16/10)** | <!-- src: CHANTIER-QWEN38.md 16/10 11:38 and 11:43 -->
| mmb (BF16 WMMA) kernels from 256 rows instead of 512 (`PPSMALL=3`, `MMB_MIN_T=256`): under 512 tokens the routed experts fell back to compact 4-bit MMQ, 4-5× more expensive per token | prefill **+6.0 % at 384 tokens, +19.7 % at 640**, +0.2 / +0.3 % at 2,048 / 8,192, +0.0 % for 600 new tokens on a 32k context, neutral at 16k and 32k; T=0 text identical on 20 prompts (146 comparisons) **(31/10)**. Threshold 128 rejected (unstable, one prompt diverged); on 16/10 bit 1 at that threshold had given short-turn TTFT ×0.677 but 1 agent ×0.850. Re-test with `llama-bench`: pp1024 604.4 ± 3.9 against 568.7 ± 22.3 without mmb, pp2048 587.5 against 541.2 **(07/11)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 31/10 13:25, 14:35, 14:55 and 07/11 21:30; CHANTIER-QWEN38.md 16/10 11:38 -->
| Native context: `-c 262144` with 2 checkpoints and a 1 GiB RAM prompt cache | prefill against `-c 131072`: +6.0 % at 0, +1.6 % at 32,768, +1.2 % at 122,880; decode ×0.98-1.00; 204,800 tokens served at 1,037 tok/s prefill and 38.0 tok/s decode; lowest MemAvailable 10.59 GB **(23/10)** | <!-- src: CHANTIER-QWEN38.md 23/10 19:50 -->

### 🗂️ Prompt cache, checkpoints and agents

| change | effect |
|---|---|
| Prompt cache on disk (`--cache-dir`, 16 GiB; upstream commits by Local Dev) | resume after a restart: 30k **18.877 → 1.063 s**, 60k **38.709 → 1.128 s**; 30k after eviction 18.799 → 1.014 s; prefix reused at 100 %; decode −0.16 % **(23/10)** | <!-- src: CHANTIER-QWEN38.md 23/10 12:16 -->
| RAM prompt cache evicted by cost (`EVIC_COST`): the entry cheapest to recompute leaves first, the biggest recent session stays, a state that costs more than it gives back is refused | with `--cache-ram 3072 --ctx-checkpoints 4`: resume of a 60k session 40.2 → 2.10 s, exact, 1 agent ×1.000, 4 agents ×1.003, 0 errors in the long session; about 1.7-2.1 GB of RAM **(20/10)**. Since 23/10 the RAM cache is 1 GiB with 2 checkpoints and the disk cache serves resumes; 3 GiB gave no gain on the production configuration **(29/10)** | <!-- src: CHANTIER-QWEN38.md 20/10 01:54 and 13:59; PILOTAGE_GUFO_2610/JOURNAL.md 29/10 14:45 -->
| Checkpoint spacing applied before the checkpoint list is full (`CKPT_EAGER`) | a long session no longer piles up one 112 MiB checkpoint per user message; with 4 agents the full re-prefills, 40 % of the gap measured on 16/10, are gone **(18/10)** | <!-- src: CHANTIER-QWEN38.md 18/10 03:40 -->
| Surplus checkpoint buffers freed off the server thread, asymmetric pool margin (`CKFREE=3`) | host buffers only, bit-exact; gate green, 4 agents included **(18/10)** | <!-- src: CHANTIER-QWEN38.md 18/10 17:02 and 18:58 -->
| A short prompt tail or the first chunk of a warm turn gets its own iteration while other slots generate (`TOURPRIO=3`; same idea as Gufo's scheduler, MIT) | admitted by decision: its own gate component was not green **(16/10)** | <!-- src: CHANTIER-QWEN38.md 16/10 08:51 -->
| SSD keep-alive while PLE rows are in use: one 4 KiB `O_DIRECT` read every 70 ms (`PLE_KEEPALIVE=70`) | PLE gather p90 7.1-7.3 → 1.4 ms; 1 agent +0.5 to +2 %; output unchanged **(14/10)** | <!-- src: CHANTIER-QWEN38.md 14/10 21:47 -->
| MTP catch-up of verification batches read on the GPU and queued early, sampler snapshot without its 3 MB scratch (`CYCLEFUSE=11`) | bit-exact; speed gain not resolved by the bench (about 0.3 ms per token estimated) **(15/10)** | <!-- src: CHANTIER-QWEN38.md 14/10 18:02; 15/10 02:01 -->

### ✅ Correctness and fidelity

| issue | impact | fix |
|---|---|---|
| Prompt cut into extra passes by the context checkpoints | cold long4k: top-1 −4.64 pts, ΔKL +0.127 (z 9.8) against no checkpoints **(09/10)** | in-batch checkpoints: same output as no checkpoints, checkpoints kept for the prompt cache **(09/10)** | <!-- src: CHANTIER-QWEN38.md 09/10 20:40 and 23:23 -->
| In-batch checkpoints first went to production without the server-side fix of the prompt cut | multi-batch unit test NMSE 1.2e+303 (restored state wrong); KL 0.0203 at 16k | fixed runtime: NMSE 0 everywhere, snapshot identical to the normal kernel byte for byte **(10/10)** | <!-- src: CHANTIER-QWEN38.md 10/10 09:00 and 09:20 -->
| Non-finite logits (NaN) in a row to sample | server crash | `NANGUARD`: output discarded, context replayed from token 0 (70 tokens in 0.24 s); idea from Strata #1892 (MIT), rewritten **(11/10)** | <!-- src: CHANTIER-QWEN38.md 11/10 21:50 -->
| KV cache really full with speculative slots: the batch was cut inside a slot's drafts (upstream #24840) | HTTP 500 "speculative batch index … is not inside the current sub-batch", or every slot cleared | `KVFULL`: no cut inside a slot's block, the biggest generating slot is freed cleanly; before 189 KV-full events and 16 HTTP 500, after 0 such errors, 11 clean refusals, the other sessions continue **(20/10)** | <!-- src: CHANTIER-QWEN38.md 19/10 18:42; 20/10 00:51 -->
| Replay of a speculative checkpoint at T > 0 reused a random draw the verification had already spent (defect "M1") | output distribution not exact on that path | `REPLAY_RNG`: fresh draw at replay; GPU oracle: M1 found only without it, 749 of 764 restorations evaluated; T=0 unchanged **(20/10)** | <!-- src: CHANTIER-QWEN38.md 19/10 22:49; 20/10 11:24 -->
| The draft's pending hidden row carried over from a slot's previous request | position 0 of a new request paired with the last row of the previous one (no content leak, a seed-like perturbation) | `MTPRESET=1`: zero row at position 0; identical to a fresh process (20/20 with 1 agent, 6/6 on 4 slots, 8/8 with 4 agents); speed unchanged **(20/10)** | <!-- src: CHANTIER-QWEN38.md 20/10 11:03, 13:00 and 18:54 -->
| LEVIATHAN with tools: a slot alternating eligible and non-eligible requests re-initialized its sampler chain | crash `GGML_ASSERT ... llama_sampler_chain_backend_init() called twice` at the first request with tools | chain no longer re-initialized (`5ec0c83d2`) **(26/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 26/10 22:20 and 22:50 -->
| LEVIATHAN on several slots: the backend top-k ids stayed a non-contiguous view while the values were materialized | wrong token ids, float bits read as ids ("failed to add token 1108731602 to the batch") when sampler chains with different k share a pass | ids made contiguous (`9b20971f0`); 3 runs × 8 agents with the top-k checker: 0 report, 0 invalid token, 0 crash, 30/30 **(27/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 27/10 04:26 and 05:52 -->
| LM head requantized to Q6_K at load (switchover 7, 11/10) | +1.4 % speed, long4k ΔKL +0.0013, top-1 agreement 99.17 % (< 99.5 %) | removed on 13/10: quality first | <!-- src: CHANTIER-QWEN38.md 11/10 12:20; 13/10 19:50 -->

KL divergence against the BF16 reference logits (`nitinpanj/Qwen3.8-Flash-Next-reference-logits`): code 0.120 (top-1
91.3 %), long4k 0.289 (85.9 %) on the first HIP runtime (08/10). On the production runtime, full-vocabulary F32 logits:
code 0.1207 / top-1 91.3 % in prefill and 0.1702 / 89.7 % in decode; long4k, healthy zone, 0.0880 / 93.2 % in prefill and
0.1257 / 90.9 % in decode, reproducible to the bit between two runs (08/11). Paired against Gufo 0.31 on the same weights,
ΔKL (us − Gufo) with 95 % intervals: code prefill +0.0115 [−0.012 ; +0.036], code decode +0.0243 [−0.008 ; +0.067],
long4k prefill −0.0072 [−0.0126 ; −0.0026] (z −1.71, under the superiority threshold z ≤ −3 set beforehand), long4k
decode −0.0052 [−0.016 ; +0.006]: no measurable difference (08/11). <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 29/10 19:20, 08/11 03:45 and 10:45 -->

On 5 November the model file was replaced by Unsloth's "V3.1" UD-Q4_K_XL. On 8 November it turned out to hold the same
1,224 tensors as the previous file (only the offsets differ): the +2.4 to +5.0 % of prefill measured on 5 November was
noise. <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 05/11 22:50 and 08/11 03:45 -->

### 🗑️ Measured and rejected

| change | result |
|---|---|
| draft `--spec-draft-n-max 4` | decode −59 % **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| KV cache in q8_0 | −30 / −59 % **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| MMV chunking | −9 % **(08/10)** | <!-- src: launch_qwen38_flash_next.sh (production) header, 08/10 -->
| halo-box decode fusions `#67` | −35 to −43 % on Kyojin's harness **(09/10)** | <!-- src: CHANTIER-QWEN38.md 09/10 00:20 -->
| Dense weights requantized to Q4_K; Halogen's quantization | top-1 −9 / −5.5 pts and −4.4 / −8.4 pts **(09/10)** | <!-- src: CHANTIER-QWEN38.md 09/10 13:40 -->
| HIP graphs | 399 of 400 graphs replayed, same output, no gain: the time is GPU, not launches **(09/10)**; neutral again with concurrent regions **(01/11)** | <!-- src: CHANTIER-QWEN38.md 09/10 09:15; PILOTAGE_GUFO_2610/JOURNAL.md 01/11 03:20 -->
| 8 routed experts per token instead of 10 | code ΔKL +0.32, top-1 −9.4 pts **(16/10)** | <!-- src: CHANTIER-QWEN38.md 16/10 08:51 -->
| RS-REPLAY (recurrent state rebuilt from compact inputs at rollback) | 71.2 against 73.2 tok/s with 8 agents, T=0 identity not held; reverted **(26/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 26/10 19:03 and 19:07 -->
| MoE wave64 kernel at 2-4 columns (`MOEVEC=3`) | −6.6 % with 8 agents **(27/10)**; in the noise and not bit-exact with 1 agent **(06/11)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 27/10 06:25; 06/11 13:10 -->
| 4 context checkpoints instead of 2 | helps only on a deep prefix break (29,990 → 8,209 re-prefilled tokens); cold prefill +1.3-2.8 % slower; the text of the restored request differed **(29/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 29/10 14:45 -->
| RAM prompt cache of 3 GiB instead of 1 | no gain on the production configuration: only the last context is reused **(29/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 29/10 14:45 -->
| Dense mmb tile driven by the weight footprint (`MMB_DENSE_TILE=1`) | no difference at 0, 32k and 131k **(29/10)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 29/10 16:45 -->
| QSA prefill kernel from 128 queries (`QSAPF`) | −4 % at 320 tokens, slower at 160 **(08/11)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 08/11 03:45 -->
| Model split across two machines over RPC | +1.74 ms per token per offloaded layer (−4.5 % per layer), deep prefill −19 to −30 % **(07/11)** | <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 07/11 19:30 -->

### ⚠️ Known issues

- The four draft-path levers of 29/10 (LEVIATHAN, COPIE24, COPIE24_COHORTE, DRAFTLOAD) went into production under watch
  rather than after the law gates. The gates run on 27/10 on LEVIATHAN + COPIE24: identity and tool calls passed; the
  replay stress failed with an HTTP 500 ("proposal alignment") in the COPIE24 arm; the long-context coverage was
  incomplete; DRAFTLOAD and COPIE24_COHORTE were not covered. A fallback for incomplete COPIE24 proposals was written the
  same evening (`f0b579d52`, its regression test fails before and passes after) but is not in the production commit
  `9b20971f0`. <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 27/10 17:20, 18:02 and 20:44; 29/10 15:57 -->
- Greedy (T=0) outputs depend on the batch shape (width of the verification passes, slot layout), so drafting changes
  greedy texts while keeping the output distribution. A 16k oracle found `-ub 8192` and `-ub 2048` deterministic and
  finite, with this shape sensitivity (top-1 95.3 %, KL 0.071; 89.1 % with NMXD) and not the logit corruption of
  upstream #28211; to be followed separately (27/10). <!-- src: PILOTAGE_GUFO_2610/JOURNAL.md 27/10 20:12; CHANTIER-QWEN38.md 23/10 15:35 -->
- Accepted by decision rather than by a green gate: `BU4` bit 1 and `PPSMALL` bit 0 (arithmetic changed, KL check
  inconclusive) and `TOURPRIO`. <!-- src: CHANTIER-QWEN38.md 16/10 08:51, 11:38 and 11:43 -->

### 🧰 Shipped here

`runtime/COMMITS.md` (the 180 commits), `launch/qwen38-flash-next.sh` (the production launcher, every switch
commented), `bench/check_launcher_parity.py` (launcher check against a production launcher).

## Vulkan runtime (late August → 1 October 2026, tag v1-vulkan)

Everything below sits on top of [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), branch
`vulkan/qwen4exp-rocmfpx` (commit `322e5cdf4cc`), which brought Qwen3.8-Flash-Next (`qwen4exp`) and the ROCmFPx types
to llama.cpp's Vulkan backend, including sparse attention (QSA) and the per-layer n-gram table read from SSD. The 57
patches in `runtime/patches/` are the complete delta.

The work was done over five weeks by AI coding agents (DeepSeek, GPT, Claude, and the local Qwen3.8 itself) under my
direction, on one Strix Halo box. This is a list of the main changes, with the measured effect where one was recorded.
Figures marked **(30/09)** were measured on the final runtime; the others are the numbers recorded when the change was
adopted, and later re-measurements are given when they disagree.

### ✅ Correctness and fidelity

| change | effect |
|---|---|
| Hyper-connection injection scale `2·sigmoid`: the graph applied `SCALE(2.0)` twice (`4·sigmoid`) and the fused `hc_combine` kernel reproduced both | KL divergence vs the full-precision reference **0.116** at `2·sigmoid`; on the same binary 1.5σ gives 0.315 and 2.5σ 0.220; the previous `4·sigmoid` build (older Vulkan backend) measured 0.478 **(30/09)** |
| Recurrent state: per-slot rollback planes (upstream #28123) + host-side zeroing of a new sequence's state row | the Vulkan graph optimizer read the state before zero-filling it in all 36 delta-net layers, so a request started from the previous one's state: on the 5th repeated request the first-token probability fell to 0.48 where the fixed runtime gives 0.9992. Fixed: repeated requests give the same greedy output and match the healthy runtime to the 4th digit **(30/09)** |
| MTP drafting hygiene: pending hidden state reset at sequence start (open upstream PR #28333); drafted tokens after an end-of-generation token are no longer accepted | no drafted tokens past the end of a turn (they forced a state restore on the next turn) **(30/09)** |
| Delta-net q/k normalisation `x·rsqrt(Σx² + ε)` as in the official implementation (upstream #28068) | parity with the reference implementation |
| QSA sparse-attention fixes: pooled-key shadowing (segfault), indexer keys written below the budget, indexer sums by chunks (upstream #28023), `seq_cp` of indexer keys (#27941) | long-context runs without crashes |
| Grammar-compatible backend sampling + reasoning budget with backend sampling | tool calls and reasoning budgets stay correct with `--backend-sampling` (with the default sampler chain the backend path was only really taken once the 30/09 grammar fast path landed) |
| Context checkpoint created on slot restore | a restored slot (pi's `kv-restore`) reuses its cache instead of re-reading the whole prompt |

### ⚡ Agent-loop speed (30/09 – 01/10)

| change | effect |
|---|---|
| Scheduler reservation kept across requests that use the same backend sampler chain (a server sets and clears the chain on every request, and each set or clear re-created the scheduler) | re-reservations **13 → 1** over 12 tool turns; first token after a short tool result **0.62 → 0.40 s**; a 49-token result on a 10.6k context **863 → 636 ms**; identical outputs **(01/10)** |
| Streaming loads (`MOVNTDQA`, AVX-512/AVX2/SSE4.1) for CPU reads of device memory | the write-combined carve-out read at ~0.4 GB/s with `memcpy`; state checkpoint read **516 → 41 ms**, first token after a short tool result **1.51 → 0.58 s** |
| Grammar fast path: draw without the constraint, check the single token, full-vocabulary pass only on rejection (same semantics as upstream's default `grammar_first = false`) | the default sampler chain cannot run on the GPU, so every verified position applied the tool-call grammar to all 248,320 tokens; tool-call decode **25 → 48–55 tok/s**, identical output at temperature 0 (diagnosed and written by the local Qwen3.8 through pi) |
| Reuse of an identical context checkpoint instead of reading the state back again | saved ~0.5 s per reuse before the streaming-load fix; ~40 ms now |
| Per-request MTP parameters (`speculative.n_max`, `.p_min`, `.n_min`) | tuning without restarting the server |

Agent loop, 12 shell commands, pi's request parameters: **11.1 → 16.2 tokens per wall-clock second**, one short turn
**3.06 → 1.38 s** (from the 30/09 afternoon's v2 candidate to the 01/10 runtime).

### 🧪 Adaptive MTP draft length (01/10, off by default)

`LLAMA_MTP_N_MAX_ANSWER=N` caps drafts at N tokens outside the `<think>` block while `--spec-draft-n-max` applies inside
it (the server follows the block from the prompt, then token by token). Measured with 5 against 4 on reasoning text,
paired rounds: **+5.6 % at 32k** (8/8 rounds), +2.1 % at 8k, +0.9 % at 16k; code text −3 % (noisy). The gain only shows
at long context, so the launcher leaves it off.

### 🏎️ Decode and prefill kernels (Vulkan)

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

### 🗑️ Measured and rejected (so you don't pay for them twice)

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

### 🧯 Fixed in the launcher on 1 October

| issue | impact | fix |
|---|---|---|
| `--cache-ram -1` (unlimited host prompt cache) | each conversation the single slot evicts is copied to host RAM; at a 138k-token agent context the server reached 25 GB of host memory on top of the ~100 GB model, and the kernel OOM killer killed the server and the desktop session | `--cache-ram 6144` |

### 🔍 Found by our own code review, fixed on 01/10

| issue | impact | fix |
|---|---|---|
| Host-side zeroing of the recurrent state called `ggml_backend_tensor_memset`, which asserts on backends whose buffers do not implement it (OpenCL, Hexagon, one SYCL buffer type) | recurrent/hybrid models aborted on those backends; Vulkan and CPU were not affected | host clearing only for host and Vulkan buffers; the others keep the in-graph clear |
| Checkpoint reuse matched on position only: with two checkpoints one token apart and a new prompt diverging exactly at the second one, a stale checkpoint could be kept | wrong recurrent state after a later restore; that exact layout was never seen in our agent-loop logs | checkpoints that cover the first diverging token are erased |
| No `MFENCE` before the streaming loads | none observed (a blocking fence wait sits between the GPU write and the read) | `MFENCE` added, as Mesa does |
| A per-request `speculative.n_max` below the launch `n_min` disabled drafting | slower request, no wrong output | `n_min` is clamped to `n_max` |
| `__builtin_cpu_supports` in the streaming-load path needs compiler-rt on Windows clang (MSVC target) | link error on that toolchain | no streaming path on Windows |

The grammar fast path follows upstream's default semantics (sample, check, fall back on rejection): at temperature > 0
with truncating samplers the distribution is the rejection-sampling one, not grammar-first; token probabilities reported
for accepted positions are the unconstrained ones; `mirostat` updates twice on a rejected position. All three are
upstream behaviours.

### 🧰 Tools shipped here

`bench/`: the agent-turn benchmark and the tool-call speed probe used for the numbers above.
