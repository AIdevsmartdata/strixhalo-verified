# Commits of the HIP runtime

The production runtime of 8 November 2026 is commit `9b20971f0`. It sits on top of
[Aristo94/GSQHalo.cpp](https://github.com/Aristo94/GSQHalo.cpp) at `5fc881b114c1ea130f5df6a30a98be2f8d397de6`. Below are
the 180 commits between the two, merges excluded, oldest first, as printed by

```
git log --reverse --no-merges --format='%h %ad %an — %s' --date=short 5fc881b114c1ea130f5df6a30a98be2f8d397de6..9b20971f0
```

Most messages are in French. The authors `pi-agent`, `pi`, `Claude…` / `claude`, `opus…` / `OPUS relecteur` /
`repairs-opus`, and the short local names `h`, `integ`, `ngrp`, `g`, `gdnlazy` and `mtp1g` are AI coding agents working
under the direction of Kévin Rémondière. The other names (Local Dev, Nathan Wilson, David Hauf, Gaetan Puleo, Laurent
Zuijdwijk, Michal Zurkowski, who also appears as delgon, Piotr Wilkin, and `agent`, an author identity of
[halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp)) are upstream contributors whose commits were
taken over. Seven upstream commits appear twice: they were taken over on two branches that were merged later.

```
6deebd45f 2026-08-30 Local Dev — server : add --cache-disk for a persistent prompt cache
f2431a23a 2026-09-14 Local Dev — server : rename --cache-disk to --cache-dir
0466407e1 2026-09-30 Nathan Wilson — vulkan : top-k n-ary search never selected +inf
bd48eb070 2026-09-30 David Hauf — hip: lower the MMB gate to 32 rows for qwen4exp contexts
686c1154f 2026-09-30 David Hauf — tests : MMB small-batch gate and indexer score cases at the qwen4exp shapes
1b899d039 2026-09-30 Nathan Wilson — vulkan : keep masked keys from leaking NaN/inf through flash attention
6402d9500 2026-09-30 David Hauf — hip: comment and help-text cleanup for the MMB small-batch gate
fca93e879 2026-10-01 David Hauf — hip: clarify the MMB opt-in, F32 gate and --mmb off wording
72495fe78 2026-10-01 David Hauf — hip : drop the F32 HC norm output only when the gate mix takes the gate GEMM
7021b01c9 2026-10-01 David Hauf — tests : MMB_QUANT_HC with the down projection and mixed HC weight types
72c10168d 2026-10-01 David Hauf — hip : reword the HC gate-mix comments
5295c001d 2026-10-02 David Hauf — hip : drop the F32 HC norm output only when the fused gate mix takes its stream mix
5b2effebc 2026-10-02 David Hauf — tests : MMB_QUANT_HC when the fused gate mix refuses: shapes, gate weight types, reader rule
4a8c81953 2026-10-02 Gaetan Puleo — hip : default build target to gfx1151
a3646f6e6 2026-10-02 Laurent Zuijdwijk — server : /dashboard live stats page and /dashboard/stats endpoint
51dc1da38 2026-10-02 Laurent Zuijdwijk — server : dashboard reports lazy_mode (strix) instead of ple_on_disk
e8b4cfd13 2026-10-02 Laurent Zuijdwijk — server : link the chat UI and the /dashboard page to each other
5d68f8ec2 2026-10-03 Gaetan Puleo — tests : use _putenv_s for QSA_FAST_MAX on Windows
55b6a52c3 2026-10-02 David Hauf — hip : keep the F32 HC gate output when no BF16 xn copy is expected
668cc7db2 2026-10-02 David Hauf — tests : MMB_QUANT_HC for the layer-0 mix without combine-norm
754021ed1 2026-10-03 Michal Zurkowski — hip : let routed experts take the measured mmb weight types, dense GEMMs keep the Q8_0 / IQ4_NL gate
1b116585c 2026-10-06 Piotr Wilkin — hip : mmb per-slice weight decoders for the K-quant and low-bit i-quant types
d8cdd7a9d 2026-10-06 Piotr Wilkin — hip : reshape the routed small tile so every wave issues WMMA
c17d9dee4 2026-10-06 Piotr Wilkin — hip : keep fused gate_up experts on MMQ
f0945c708 2026-10-06 Piotr Wilkin — hip : admit the measured low-bit expert types to the routed mmb gate
6f8903b74 2026-10-02 David Hauf — hip : keep the F32 HC gate output when no BF16 xn copy is expected
88c135ce2 2026-10-02 David Hauf — tests : MMB_QUANT_HC for the layer-0 mix without combine-norm
4780c7297 2026-10-03 Michal Zurkowski — hip : let routed experts take the measured mmb weight types, dense GEMMs keep the Q8_0 / IQ4_NL gate
ba38ed332 2026-10-06 Piotr Wilkin — hip : mmb per-slice weight decoders for the K-quant and low-bit i-quant types
d8691ccd3 2026-10-06 Piotr Wilkin — hip : reshape the routed small tile so every wave issues WMMA
dd290602e 2026-10-06 Piotr Wilkin — hip : keep fused gate_up experts on MMQ
2caabdfc0 2026-10-06 Piotr Wilkin — hip : admit the measured low-bit expert types to the routed mmb gate
183decc63 2026-10-08 pi-agent — hip : size the routed MoE tile from the AMD tokens-per-expert heuristic
244e38a71 2026-10-08 pi-agent — hip : admit the served expert types to the routed compact MoE tiling
8ad9dc8ff 2026-10-09 h — perf logger
6352e9583 2026-10-06 Piotr Wilkin — cuda : route small-row wide-batch matmuls to the matvec kernel on AMD
283b332df 2026-09-29 agent — CUDA: enable the fused hc_mix_reduce at decode (qwen4exp)
f9f9cf947 2026-09-29 agent — CUDA: enable the GDN decode-chain fusion for qwen4exp (rms_norm+scale q/k norm)
07c6bbec4 2026-09-29 Gaetan Puleo — CUDA: apply the qwen4exp GDN output gate in the decode norm kernel
7f15299cd 2026-09-20 delgon — ggml-cuda : round growing pool requests up to a power of two
095c4203b 2026-10-09 integ — integ 09/10 12h: P(v1 concat+cpy, v2 glu_nc, v3 gdn_verify WIP) + O(v1 hc_upmix, mmv_group_f32)
bc9e1fff0 2026-10-10 pi-agent — integ4 : prod INB2 (095c4203b + INB.patch corrige de Q) + lot2 de P (etat L3b valide, MMB_HCD_SHADOW OFF) + L3 de DEC (f6f8a6db8, GGML_CUDA_CPY_KERNEL) ; arbre identique a lot2inb2/02f74ba3e plus cpy.cu seul
b5141073d 2026-10-10 pi-agent — qwen4exp : tête de brouillon MTP sur un sous-ensemble du vocabulaire (--spec-draft-mtp-vocab N + LLAMA_MTP_HEAD_IDS=<ids fréquents>)
2d5a327a6 2026-10-10 pi-agent — hip : __shfl_xor_sync over DPP row_xmask / permlanex16 on RDNA3/RDNA4
e87eb3ec1 2026-10-10 pi-agent — KA-3: fusion de la chaine de score de l'indexeur au decode (GGML_OPUS_IDXSCORE=1, defaut OFF)
57aca9827 2026-10-10 pi-agent — OPUS-P1 : copies d'entrees de split groupees (GGML_OPUS_P1, defaut OFF) : H2D groupe + noyau de dispersion, plafond d'entrees releve pour les petites entrees
af85c9b71 2026-10-10 pi-agent — OPUS-P1 : diagnostic GGML_OPUS_P1_LOG=2 (liste des entrees de split)
9f03ca6f6 2026-10-10 pi-agent — OPUS-P1 revue : le plafond releve n est evalue (ggml_nbytes) que si le plafond d'origine de 30 est atteint, pour ne pas ajouter de surcout hote au chemin OFF
3fb39d2a9 2026-10-10 pi-agent — V1LONG: LOT LONG (P1 sparse-pack, P4 qsa incremental, D1 topk wg1) + gate de profondeur GGML_OPUS_LONG_MIN_KV (defaut 16384) + maitre GGML_OPUS_LONG; defaut OFF
513c334e9 2026-10-10 pi-agent — V1LONG (revue): D1 top-k wg1 seulement pour le TOP_K marque op_params[0]==1 (selection de blocs complets, ids retries en ordre croissant) ; le top-k token-level de repli (images, multi-flux) nourrit le noyau de decodage dans l'ordre des ids => chemin d'origine
17b8977bc 2026-10-10 opus-V1HCK — KA-1 : GGML_OPUS_PREQ (preq ssm_out/attn_output + R lignes par onde shexp down), defaut OFF
48ba59239 2026-10-10 opus-V1HCK — KA-1 : garde y_op>3 (compat V1GDN)
25b77bdb2 2026-10-10 pi-agent — V1MHC : GGML_OPUS_Q8PROD=1 dedoublonnage des quantize_q8_1 MMVQ d'une meme activation (cache Q8_1 arme par anticipation, calque sur mmq_y_cache)
38c7d0f49 2026-10-10 pi-agent — OPUS V1GDN: GGML_OPUS_GDN=1 : norme gatee GDN dans le prologue d'ssm_out (ordre d'accumulation rejoue, y_op 4) + conv-state lu en place (sans GET_ROWS)
bcdd98d3d 2026-10-10 pi-agent — OPUS V1GDN revue : conv-state sur place seulement pour n_tokens==1 (le prefill T>=256 lit l'historique apres la copie de queue : corruption)
3b7594bd6 2026-10-10 pi-agent — OPUS KA-4 (V1MOE3): queue de FFN en un noyau (wred + shared_gate_mul_add + hc_combine_norm), GGML_OPUS_FFNTAIL=1, defaut OFF
6d20da66a 2026-10-10 pi-agent — OPUS KA-4: variante FULL sans gardes de colonne (n_embd = kp*512)
7826dfc52 2026-10-10 pi-agent — OPUS KA-4: correction du lancement FULL
2a4129b92 2026-10-10 OPUS relecteur — OPUS KA-4 relecture: la reshape de block_out entre dans le sous-graphe (sinon can_fuse refuse toujours), couches weighted-down non deplacees, dependances d'allocation anti-alias
62880065d 2026-10-10 pi-agent — G11 : chargements anticipes (technique 7e4621c de Gufo, MIT) dans hc_combine_norm, hc_up_mix, wred, sgma-gate, qsa wmma ; GGML_OPUS_G11=1 (defaut OFF)
7f0b81f9a 2026-10-10 pi-agent — G11 : ajout de la partie 32 (echelles Q4_K/Q5_K sans branche dans mul_mat_vec_q4_columns_rdna3_5)
a2058ef4b 2026-10-10 opus — VE-1: noyaux q4_columns Q8_0 (PF hc_down, gate shexp) etendus a 2 et 3 colonnes derriere GGML_OPUS_VE1 (defaut OFF)
4082787df 2026-10-10 opus — VE-1b: groupe Q8_0 multi-colonnes (qkv+gate GDN, wk+wv) derriere GGML_OPUS_VE1_GRP (defaut OFF)
7dbe04758 2026-10-10 pi-agent — V2MTPB : MU-1 brouillon uniforme multi-slot (--spec-uniform-draft off|min|N), actif seulement >=2 slots gloutons, defaut OFF; env repli
0c8a012cb 2026-10-10 pi-agent — V2MTPB relecture : ne jamais tronquer un brouillon de rejeu (spec_is_replay restait arme sur un brouillon neuf non verifie) ; un slot sans place de brouillon ne force plus la cohorte a 0
29b0d3a00 2026-10-10 pi-agent — V3BF16 (KA-8): jumeaux BF16 exacts des petits poids F32 (routeur, ssm_alpha/beta, hc inject), LLAMA_OPUS_BF16=1, defaut OFF
fdbaabf76 2026-10-10 pi-agent — V3PLE: PF-4 lecture PLE anticipee (LLAMA_OPUS_PLE_EARLY, LLAMA_PLE_ADVISE, LLAMA_PLE_EVICT), defaut OFF
b98ff6141 2026-10-10 pi-agent — V3PLE relecture: hook serveur update_slots deplace apres la restauration de checkpoint (n_past reel)
66b5ec0c3 2026-10-10 pi-agent — V3HEADQ6: requantification de output.weight au chargement (LLAMA_OPUS_HEAD_QTYPE / LLAMA_OPUS_DRAFT_HEAD_QTYPE), defaut OFF
9f5ae837a 2026-10-10 pi-agent — SP-3 : longueur de brouillon MTP par probabilite cumulee (LLAMA_OPUS_SP3), journal de calibration (LLAMA_OPUS_SP3_LOG)
88c30a799 2026-10-10 pi-agent — SP-3 : vidage du journal CSV tous les 16 lignes (un arret par signal ne perd plus le journal du test de fumee)
c9c6a427c 2026-10-10 Claude — V3Q8FULL : lignes q8_1 ecrites par les producteurs (combine_norm, hc_mix_reduce, norme GDN), interrupteur GGML_OPUS_Q8FULL=1 defaut OFF
7bce61b81 2026-10-10 Claude — V3Q8FULL : lecteurs a >= 2 colonnes seulement (a 1 colonne le prologue fq quantifie deja)
bc35da9e2 2026-10-10 pi-agent — V3Q8FULL (relecture adverse) : lookup reserve au flux 0 (multi-flux), garde nelements=S*H_v*T sur la norme GDN
1b5c98641 2026-10-10 pi-agent — integ5-essai : correction d'accolade (fusion KA-4 + Q8FULL dans ggml-cuda.cu)
12d36378e 2026-10-10 pi-agent — V6CHAUD : commutateurs OPUS modifiables a chaud (registre + GET/POST /opus/flags)
00668814e 2026-10-11 pi-agent — V8MULTI : --spec-uniform-draft-sampling (MTPB aussi a T>0, loi inchangee) + --spec-uniform-adapt (largeur de cohorte W in 0..N selon l'acceptation recente, uniform_ctl.h); defauts OFF
3ad453fa0 2026-10-11 pi-agent — V9MMVQ : GGML_OPUS_MMVQ_MULTI=<n> verification multi-slot par tranches sur le noyau vectoriel MMVQ (MoE mul_mat_id tranches de 4 jetons, dense colonnes equilibrees); defaut OFF
c0f5f38e6 2026-10-11 pi-agent — V7RQ: requantification au chargement par motifs de noms (LLAMA_OPUS_REQUANT), generalise HEADQ6, defaut OFF
ce9f0f978 2026-10-11 pi-agent — V7RQ: recapitulatifs au niveau WARN (les INFO du chargement sont filtres dans srv.log)
c6d9cc030 2026-10-11 pi-agent — V10CAP: GGML_OPUS_MMVQ_CAP, plafonds MMVQ RDNA3.5 de la PR amont #28613 (MIT, SimonTeixidor, head 6617683e43f9) derriere interrupteur (0=OFF exact, 1=toute largeur, N=ne11>=N)
a018539c7 2026-10-11 pi-agent — PLEOFF: lignes PLE chargees hors du chemin critique (1: rechauffement du cache de lignes depuis le brouillon MTP ; 2: collecte asynchrone derriere une porte d'entree attendue avant le premier lecteur)
d9e52d4ce 2026-10-11 pi-agent — SMALLFUS bit 1: top-k 1 ligne (brouillon MTP) par noyau 1 groupe + tri des k ids (ordre identique au tri radix stable)
95fb8f6fc 2026-10-11 ngrp — NGRP: groupes 2..4 colonnes avec segments F32/BF16 (alpha/beta, routeur+shexp, inject+hc_down) derriere GGML_OPUS_NGRP
d463838d3 2026-10-11 pi-agent — SMALLFUS bit 2: IMRoPE dans la fusion rms_norm+mul+rope(+view+set_rows) (bloc 128 pour les 128 colonnes)
62ce02dc9 2026-10-11 pi-agent — SMALLFUS bit 4: tri par rang (512 fils) dans qsa_expand_complete_blocks_512
ab25f974b 2026-10-11 pi-agent — SAMPGPU: pre-filtre top-K sur GPU pour l'echantillonnage hote (LLAMA_OPUS_SAMPGPU=K, defaut OFF)
ddb3f826c 2026-10-11 pi-agent — SAMPGPU: top-K en deux noyaux (16 tranches par ligne puis fusion) au lieu d'un bloc par ligne
e4e709847 2026-10-11 pi-agent — SAMPGPU: llama.h, declaration deplacee hors du commentaire de la grammaire paresseuse
e9a69662d 2026-10-11 gdnlazy — GDNSNAP: stores non temporels pour les instantanes de retour arriere GDN (GGML_OPUS_GDNSNAP=1|2)
a1cbbeb20 2026-10-11 g — GDNSNAP: noyau scalaire (verification n=3)
718cbc75e 2026-10-11 g — GDNSNAP: fix pragma
37027834b 2026-10-11 pi-agent — INTEG7: registre a chaud etendu (NGRP, SMALLFUS, GDNSNAP, MMVQ_CAP/MULTI/CHUNK, PLEOFF, SAMPGPU) + remise a zero des marques backend au changement de generation
1b32c7ab1 2026-10-11 pi-agent — CDFIX: LLAMA_OPUS_DISK_IDLE (ecritures du tiroir differees pendant la generation, ioprio idle, sans fsync)
a6af1d4af 2026-10-11 pi-agent — CDFIX: pas de copie des lignes pendant la generation (disk_hold_now), plafond 120 s
204a70ea5 2026-10-11 pi-agent — SMALLFUS: bit 2 = norme+rope IMRoPE sans set_rows, bit 8 = avec set_rows (isolation de la derive d'integ7)
d84567f48 2026-10-11 ngrp — NGRP: reutiliser le q8_1 du producteur (Q8FULL) / du cache Q8PROD au lieu de requantifier (le TOUS perdait ce que ces leviers economisent)
40874dde7 2026-10-11 ngrp — NGRP bit 8 : ancien comportement (requantifier) pour l'A/B a chaud
cc76eb805 2026-10-11 ngrp — VE1_GRP: GGML_OPUS_VE1_REUSE=1 reutilise le q8_1 du producteur / cache Q8PROD au lieu de requantifier (bit-exact)
dd1943847 2026-10-11 pi-agent — NANGUARD: LLAMA_OPUS_NANGUARD=1 jette la sortie a logits non finis, vide le contexte du slot et le relit depuis 0 (idee Strata #1892, MIT, reecrit) ; LLAMA_OPUS_NANTEST pour le banc
2f33b8c6c 2026-10-11 pi-agent — NANGUARD: le rejeu ne rappelle pas update_prompt_start (assertion t_start) ; NANTEST accepte une liste N1,N2
fddf7cc60 2026-10-11 pi-agent — P1FIX: GGML_OPUS_P1 au chargement seulement (refus de commutation a chaud + WARN serveur)
15f11b2ca 2026-10-11 pi-agent — SYNC1: LLAMA_OPUS_SYNC1=1 lit la ligne d'echantillonnage en une synchronisation (port de ggml-org/llama.cpp#30196)
40a2bb1de 2026-10-11 mtp1g — MTP1G: rattrapage MTP replie dans le graphe du brouillon 1 (LLAMA_OPUS_MTP1G=1, mono-slot)
8ae1082c4 2026-10-12 repairs-opus — MTP1G: report de rattrapage valide seulement si la memoire du brouillon n'a pas bouge (correctif 'failed to initialize batch' a pos=0 : report perime de la requete precedente vide dans ctx_dft fraichement efface) ; begin() garde la queue courte du prompt
08da084a9 2026-10-12 repairs-opus — POOL: GGML_CUDA_POOL_CAP_MB (plafond souple par pool legacy : vidage des blocs en cache avant de grossir, sans arrondi) + ggml_backend_cuda_pool_trim (get_proc_address) + LLAMA_OPUS_POOL_IDLE_TRIM=<ms> cote serveur (purge apres <ms> de repos de tous les slots, minuteur qui reveille la boucle) ; defauts OFF
dc336a5df 2026-10-12 repairs-opus — PLEHOT: port de V4PLEHOT (4f7e427bf) sur INTEG7 : cache de lignes PLE dedie associatif verrouille (LLAMA_PLE_CACHE_MB, alias LLAMA_PLE_HOT_MB ; LRU 8 voies, mlock), trace LLAMA_PLE_TRACE, LLAMA_PLE_IOPRIO ; rechauffement PLEOFF branche sur le cache associatif ; un seul cache dedie par processus ; defauts OFF
9c92d1b2c 2026-10-10 pi-agent — V5GDNCH : GGML_OPUS_PPSMALL (defaut OFF) bit0 = GDN par blocs + INB des 64 jetons (seuil 256->64, plafond instantanes 255->63, garde ngram-simple), bit1 = noyaux mmb des GGML_OPUS_MMB_MIN_T lignes (512->128)
6016e7e42 2026-10-10 claude — MTPCU : rattrapage MTP du prefill sans aller-retour hote (h_nextn residente GPU), derriere LLAMA_OPUS_MTPCU=1
bfe781444 2026-10-10 claude — MTPCU : repli hote par tranche si la source device est refusee
3979417f6 2026-10-10 pi-agent — MTPCU : h_nextn de la cible est 3-D [n_embd,hc,n] (levier inerte) : test par nombre d'elements + vues 2-D ; ggml_backend_view_init sur les vues (repli generique)
412ed3210 2026-10-12 pi-agent — DISKCACHE2: cache disque sobre (defauts OFF = comportement actuel) : LLAMA_OPUS_DISK_NOCKPT=1|2 (etat de fin seul | + dernier point de controle du dernier prompt), LLAMA_OPUS_DISK_MAXMBPS=<Mo/s> (seau a jetons de l'ecrivain par morceaux de 1 Mio, leve des qu'on attend le tiroir : persist, slot sortant sur file pleine, restauration, arret), LLAMA_OPUS_DISK_MINTOK=<n> (seuil aussi pour les runs en croissance), LLAMA_OPUS_DISK_EVICTONLY=1 (ecriture seulement a la sortie du slot : autre conversation, make_room/purge, /strix/persist)
98e1756c9 2026-10-12 pi-agent — MEMG: LLAMA_OPUS_MEMSTAT=<s> (ligne [MEMSTAT] : tampons modele/contexte/calcul cible+brouillon, pools HIP, points de controle par slot, cache de prompts, MemAvailable/GTT/RSS ; lecture seule) ; LLAMA_OPUS_CKPT_EAGER=1 (pas minimal des points de controle aussi quand la liste n'est pas pleine) ; LLAMA_OPUS_CKPT_TOTAL_MB=<n> (plafond global en octets, evince d'abord le plus ancien du slot inactif le plus charge). Defauts OFF = code d'origine.
8d2121e9e 2026-10-12 pi-agent — PFX 1/2: report de V12PFX (58b55c897, integ6) sur integ7-essai-1011 : LLAMA_OPUS_PFX=1 (point de controle appris sur le prefixe partage), defaut OFF
f8efa0bf1 2026-10-12 pi-agent — PFX 2/2: alignement du point sur 64 jetons (grille des blocs GDN), election d'un meneur et attente des suiveurs pour une salve de sous-agents (opus_pfx_front : READY/PENDING/BURST), reprise de l'etat du meneur par le cache RAM
4a62bc780 2026-10-12 pi-agent — PFX 3/3: lecture des interrupteurs par le registre GGML_OPUS_INT (integ7 n'a pas opus_ple_env), interrupteurs inscrits (categorie c, une lecture), reprise possible de tout point de controle du prefixe commun (READY)
8e147fa47 2026-10-12 pi-agent — server : EVIC, RAM prompt cache eviction policy behind LLAMA_OPUS_EVIC_DEDUP and LLAMA_OPUS_EVIC_COST
7e463215b 2026-10-12 pi-agent — tests : EVIC policy test, byte limit makes the token limit unreachable in update()
3c9943774 2026-10-12 pi-agent — server : EVIC, admission test in percent (ADMIT, default 200) and cache fill in the log lines
6d51ce6df 2026-10-12 pi-agent — tests : EVIC policy test, burst replay, replay printing modes, sizes of M3/A3/A4 from the log
d8d7210e0 2026-10-12 pi-agent — server : EVIC, evic_transport replaces evic_force (a state saved to be loaded back at once keeps the protection, skips only the admission test, no dedup, no coverage credit)
34e995b65 2026-10-12 pi-agent — tests : EVIC policy test in the order of the real server (take before save), origin witness at 5120 MiB, transport state keeps the protection, PROT_S=0/30 fallback, --replay-burst LIMIT
df9ad9aab 2026-10-13 pi-agent — INTEG8 : crochet EVIC<->PFX (EVIC/pfx_hook.patch) : evic_transport autour de rdy->prompt_save() dans la reprise READY de PFX
e950025ad 2026-10-14 Claude — LEV9 : INTEG8 + NGRB (LLAMA_OPUS_RS_SEQ, NGRAM_ADAPT) + MOE16 (GGML_OPUS_MMID_MAXB) + PFIL (LLAMA_OPUS_PFCAP), interrupteurs OFF par defaut
34151c3cd 2026-10-14 pi-agent — PLEPREF : lecture anticipee effective des lignes PLE dans le cache de pages (LLAMA_OPUS_PLEPREF, OFF par defaut)
e0c8334a0 2026-10-14 pi-agent — PLEPREF revue : garde has_media (abandon sur requete image), sonde STATS sans I/O (cachestat), priorite des requetes de verification dans la file
e6fd3a9b9 2026-10-14 pi-agent — MIDB : decodage a 5-16 jetons par pas (GGML_OPUS_MIDB, masque, OFF par defaut)
7273d8e08 2026-10-14 Claude — PPFLOOR : plancher des petits prefills et des tours chauds (TAILSMALL N, CKPOOL B, CKDEFER B, TRACE, MMVF16 N+, TPLCACHE B), interrupteur LLAMA_OPUS_PPFLOOR OFF par defaut
a30f1ccc1 2026-10-14 pi-agent — PPFLOOR revue : retrait de MMVF16 (changement de precision interdit), garde CKDEFER (slot vide/rembobine par NANGUARD ou fils libere), tache d'origine conservee pour les points de controle differes, travaux differes vides sur echec de decode
920363588 2026-10-14 pi-agent — MIDB : q4_columns 4 lignes par onde pour NC 9..16, mode de controle GGML_OPUS_MIDB_CHECK
f5e26c9f2 2026-10-14 pi-agent — MIDB revue : budget du mode de controle par nombre de colonnes
bd442e77d 2026-10-14 pi-agent — CYCLESTATS : sonde interne de chronometrage des cycles de decodage et des tours d'agent (LLAMA_OPUS_CYCLE_STATS, OFF par defaut, classe B)
0d1c074f9 2026-10-14 pi-agent — CYCLESTATS revue : synchro GPU internes de llama_process (v_sync), cout de l'ecriture du resume, OFF sans horloge au cache de prompt, jalons HTTP sans TLS perime
338c8e0fb 2026-10-14 pi-agent — WARMHOST : tokenisation incrementale bit-exacte des tours chauds + un rendu de gabarit en moins (Qwen3-Coder) + tables builtins par reference ; outil CPU opus-hostbench ; interrupteur LLAMA_OPUS_WARMHOST (0 par defaut)
5a25bf99f 2026-10-15 pi-agent — INTEG10 : patch INTEG10 (OFF par defaut)
f0eedcc7c 2026-10-15 pi-agent — INTEG11 : patch INTEG11 (OFF par defaut)
2b0e02634 2026-10-15 pi-agent — PLEDOUBLE : patch .. (OFF par defaut)
cf932c987 2026-10-16 pi-agent — PDMMQJ : patch PDREBASE (OFF par defaut)
2a38f1edb 2026-10-18 pi-agent — INTEG12 : patch INTEG12 (OFF par defaut)
db93cc073 2026-10-18 pi-agent — INTEG13 : patch sorties (OFF par defaut)
19cc6a88c 2026-10-19 pi-agent — INTEG15 : patch combinaison (OFF par defaut)
6ff3b70a9 2026-10-19 pi-agent — INTEG15KV : patch patch (OFF par defaut)
604272a8e 2026-10-20 pi-agent — INTEG16KV : patch patch (OFF par defaut)
dbc76718c 2026-10-20 pi-agent — INTEG18C : patch build (OFF par defaut)
fce5050dc 2026-10-23 pi — NMXD : plafond de brouillon pilote par la profondeur KV (LLAMA_OPUS_NMXD=lo:hi:n_hi:n_lo, OFF par defaut)
e0a98dc74 2026-10-26 Claude (pilote) — ROWTILE : down Q5_1 des experts routes a 3 colonnes, GGML_OPUS_MOE_Q5_1_RPB=2|4|8 (defaut 2 = prod) — row8.patch de Kimi (24/10), paquet runtime-opt/pilots/row8-hip24
eeb34c00a 2026-10-26 Claude (pilote) — QSA etroit : GGML_OPUS_QSA_GROUP (groupes de 2/4 tetes) et GGML_OPUS_QSA_IDS4 (ids par int4), OFF par defaut — patch S6 (Sol, 26/10), idee Gufo #626 (MIT)
487d00152 2026-10-26 Claude (pilote) — MOEVEC32 : MOEVEC jusqu'a 32 colonnes (GGML_OPUS_MOEVEC_MAX 17..32 explicite ; defaut 16 = prod) — patch S3 (Sol, 26/10) ; commute avec ROWTILE (mmvq.cu identique dans les deux ordres)
b6f03f20a 2026-10-26 Claude (pilote) — Q8WAVE : petits denses Q8_0 de 9 a 24 colonnes en vagues de 8 (OFF par defaut) — patch 02 de Kimi K3.1 (26/10), idee Gufo mmvq.hip.cpp (MIT)
d326fcbeb 2026-10-26 Claude (pilote) — BF16_LOAD8 : chargements BF16 de 8 octets (MMVF, groupes FQ/NGRP), GGML_OPUS_BF16_LOAD8=1, OFF par defaut — patchs 00+01 de S4 (Sol, 26/10), idee Gufo #681 (MIT)
5d64229f6 2026-10-26 Claude (pilote) — ENVCACHE : fige les tests getenv du backend au premier acces, GGML_OPUS_ENVCACHE=1 (au chargement), OFF par defaut — patch 02 de S4 (Sol, 26/10), idee llama.cpp #30327 (MIT)
e04bafd85 2026-10-26 Claude (pilote) — DRAFTPOL : controleur de longueur de brouillon adaptatif par requete (table de couts JSON, EMA d'acceptation par position), LLAMA_OPUS_DRAFTPOL=1, OFF par defaut — patch S9 (Sol, 26/10), idee Gufo mtp_policy (MIT) ; registre fusionne a la main
19ee00968 2026-10-26 Claude (pilote) — Q8WAVE : plafond de colonnes GGML_OPUS_Q8WAVE_MAXCOLS (defaut 24 = comportement du patch) ; criblage du 26/10 : gain a 12-16 colonnes, perte a 20-24
630867a95 2026-10-26 Claude (pilote) — LEVIATHAN : echantillonnage speculatif exact pour les brouillons MTP a T>0 (brouillon tire dans q sur 64 candidats, acceptation min(1,p/q), residu exact), LLAMA_OPUS_LEVIATHAN=1, OFF par defaut — patch S10 (Sol, 26/10), idee Gufo mtp_sampling.hpp (MIT, licence incluse)
a2f8e358a 2026-10-26 Claude (pilote) — GDN gather-concat : fusion de copie exacte (36 lancements par passe), OFF par defaut — patch S11 (Sol, 26/10)
928812e5d 2026-10-26 Claude (pilote) — RS-REPLAY : etat de depart + entrees compactes, rejeu exact au retour arriere (LLAMA_OPUS_RSREPLAY), OFF par defaut — patch K3 (Kimi K3.1, 26/10)
af593a37b 2026-10-26 Claude (pilote) — A3 : politique de speculation complete (cout des propositions, rejeux RS2, largeur par fente en cohorte), OFF par defaut — patch A3 (GPT-6-Astra, 26/10)
a39464a05 2026-10-26 Claude (pilote) — RS-REPLAY : acces direct a op_params (les aides ggml_*_op_params_i32 sont internes a ggml) — correction de compilation du pilote
0a7db159b 2026-10-26 Claude (pilote) — Revert "RS-REPLAY : acces direct a op_params (les aides ggml_*_op_params_i32 sont internes a ggml) — correction de compilation du pilote"
6babf4f50 2026-10-26 Claude (pilote) — Revert "RS-REPLAY : etat de depart + entrees compactes, rejeu exact au retour arriere (LLAMA_OPUS_RSREPLAY), OFF par defaut — patch K3 (Kimi K3.1, 26/10)"
887c824cc 2026-10-26 Claude (pilote) — RS-REPLAY v2 (K3c) : entrees de rejeu initialisees, verification n_replay a la reutilisation des graphes, gardes et repli RS2 — Kimi K3.1 (26/10)
6a5876c6a 2026-10-26 Claude (pilote) — COPIE24 : regles de copie lookup facon Gufo (correspondance >= 24 en echantillonne, plafond 8, sources bornees), OFF par defaut, compatible LEVIATHAN — Kimi K3.1 (26/10), idee Gufo ngram.cpp (MIT)
42fed6b4b 2026-10-26 Claude (pilote) — RS-REPLAY v2 : acces direct a op_params (aides internes a ggml) — correction de compilation du pilote
73fe63a87 2026-10-26 Claude (pilote) — S12 : instrumentation des portes de loi LEVIATHAN (audit grammaire/logits, mutant RNG, RS2 force), OFF par defaut — patch S12 (Sol, 26/10) ; hunk common.cpp fusionne a la main apres le bloc A3
c71100e58 2026-10-26 Claude-pilote — COPIE24_COHORTE : copies de lookup en cohorte multi-fente a largeur egale (1 remplacement, 2 apres MTP0, 3 elargissement au cout), OFF par defaut — Kimi K5 (26/10)
9750efa4e 2026-10-26 Claude-pilote — Revert "RS-REPLAY v2 : acces direct a op_params (aides internes a ggml) — correction de compilation du pilote"
0b15afd4e 2026-10-26 Claude-pilote — Revert "RS-REPLAY v2 (K3c) : entrees de rejeu initialisees, verification n_replay a la reutilisation des graphes, gardes et repli RS2 — Kimi K3.1 (26/10)"
7c8f00d1c 2026-10-26 Claude-pilote — RS-REPLAY v3 (K3d) : verification et rejeu par le meme GDN (K=n_rs_seq+1), etat final seul dans le cache ; points de controle avec etat de base + journal compact pendant un rollback ; op_params directs — Kimi K3.1 (26/10)
594ce1403 2026-10-26 Claude-pilote — S12b : audit des copies pour les portes de loi du candidat complet (traces MTP/copie separees), OFF par defaut — Sol (26/10)
d04fa7da0 2026-10-26 Claude-pilote — Revert RS-REPLAY v3 (K3d) : aucun gain a U8 (71,2 contre 73,2 t/s), identite T=0 non tenue ; retire de l'integration pour les portes de loi S12b (ggml.h attendu par le kit)
0d2896b08 2026-10-26 Claude-pilote — S12c : audit des logits avec position et role (cible/brouillon) pour aligner la porte d'identite en classe N — Sol (26/10)
bbbe4ea34 2026-10-26 Claude-pilote — PLE_PRECHAUFFE : prechauffe verrouillee des lignes PLE chaudes au chargement (classe B), OFF par defaut — Sol S18 (26/10) ; avenant pilote : plancher MemAvailable reglable LLAMA_OPUS_PLE_PRECHAUFFE_PLANCHER_GO (defaut 30)
5ec0c83d2 2026-10-26 Claude-pilote — S10b : LEVIATHAN — plus de reinitialisation d'une chaine deja initialisee quand une fente alterne requetes eligibles/non eligibles (plantage avec outils, GGML_ASSERT llama_sampler_chain_backend_init called twice) — Sol (26/10)
f2125464d 2026-10-26 Claude-pilote — S10c : garde et diagnostic du lot LEVIATHAN multi-fente (jeton hors vocabulaire isole avant l'ajout au lot, fente et provenance journalisees) — Sol (26/10) ; cause non etablie
829270ed2 2026-10-26 Claude-pilote — K6 : multi-fente — LLAMA_OPUS_SEQGAPS (regroupe les fentes trouees, K/V compactes) et GGML_OPUS_VPLANS (plans HIP par disposition), OFF par defaut — Kimi K3.1 (26/10)
195ef4772 2026-10-27 Claude-pilote — S12d : audit compact des portes de loi (alignement par position, NUM chronometre), OFF par defaut — Sol (26/10)
ef1972941 2026-10-27 Claude-pilote — LEVIATHAN_MONO : LEVIATHAN reserve aux passes a une seule sequence (contournement du jeton de brouillon aberrant en multi-fente avec outils), OFF par defaut — pilote (27/10)
b25dbc8ea 2026-10-27 Claude-pilote — K7 : mono — MoE wave64 2-4 colonnes Q4_K/Q5_K (GGML_OPUS_MOEVEC=3) et HC up+mix 2-4 colonnes (GGML_OPUS_HCUP_NC=1), OFF par defaut, idee Gufo (MIT) — Kimi K3.1 (27/10)
dbd3383f1 2026-10-27 Claude-pilote — K9 : instrument LLAMA_OPUS_TOPK_CHECK (controle GPU+hote des candidats top-k, oracle CPU, vue SYNC1), non fatal, OFF par defaut — Kimi K3.1 (27/10)
9b20971f0 2026-10-27 Claude-pilote — K10 : top-k backend — ids materialises (ggml_cont) comme les valeurs ; corrige les identifiants faux / bits de flottant quand des chaines a k differents partagent une passe (LEVIATHAN multi) — Kimi K3.1 (27/10)
```
