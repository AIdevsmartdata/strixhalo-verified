#!/usr/bin/env python3
"""probe_agent_turns.py <port> <tag> <out_dir> [n_turns=12] [ctx=8000] [mode=fidele|sans_raisonnement|args_reserialises]
AGENT-TURN BENCHMARK: what ONE turn of a tool loop costs, the way pi (or any coding agent) drives the server.
Decode speed alone does not tell it: before the first token of each turn the server creates checkpoints, may re-reserve
its graph and processes the new tokens. This measures that delay (TTFT) and the agentic rate (tokens generated / wall clock).

Deterministic scenario: system prompt + project notes (ctx tokens, French text, kept as measured), tools bash/read/edit/write
(pi-like schemas); the user asks for K commands, ONE per tool call; tool results are synthetic (short / medium / long).
Request parameters = pi's, captured on 14/09: stream, reasoning_effort (env TOURS_EFFORT, default medium),
thinking_budget_tokens 1024, temperature 0, top_k 20, top_p 0.95, min_p 0.05, seed 42, id_slot 0; cache_prompt = True.
Modes (how the history is sent back):
  fidele              reasoning_content + tool_calls returned exactly as received (what pi does)
  sans_raisonnement   reasoning_content dropped from the history (harnesses that discard reasoning)
  args_reserialises   tool_call arguments re-serialised with json.dumps(separators=(',',':')) (harnesses that re-parse)
Output lines (labels in French): TOUR = one turn (ttft, mur = wall clock, prompt_n / cache_n / pred_n, outil = tool call
present, bon = correct command); BILAN_TOURS = summary (commandes justes = correct commands, jetons generes = tokens
generated, tok/s AGENTIQUE = tokens per wall-clock second, decodage seul = decode only). NON_MES = not measured."""
import json, os, sys, time, urllib.request
PORT, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 12
CTX = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
MODE = sys.argv[6] if len(sys.argv) > 6 else "fidele"
os.makedirs(OUT, exist_ok=True)
CORPUS = open(os.environ["TOURS_CORPUS"], encoding="utf-8").read() if os.environ.get("TOURS_CORPUS") else "".join(
    "Project note %d: the build uses Vulkan, tests run with ctest, logs go to build/logs, configuration lives in config/*.yaml.\n" % i for i in range(4000))


def post_json(path, body, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:%s%s" % (PORT, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def n_jetons(txt):
    return len(post_json("/tokenize", {"content": txt})["tokens"])


TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to the last 2000 lines or 50KB.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Bash command to execute"},
                                                         "timeout": {"type": "number", "description": "Timeout in seconds (optional)"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read the contents of a file. Supports text files and images. For text files, output is truncated to 2000 lines or 50KB. Use offset/limit for large files.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                                                         "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                                                         "limit": {"type": "number", "description": "Maximum number of lines to read"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Edit a file by replacing exact text. The oldText must match exactly (including whitespace). Use this for precise, surgical edits.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                                                         "oldText": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
                                                         "newText": {"type": "string", "description": "New text to replace the old text with"}}, "required": ["path", "oldText", "newText"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                                                         "content": {"type": "string", "description": "Content to write to the file"}}, "required": ["path", "content"]}}},
]

# --- prefixe : prompt systeme + notes de projet, a la taille demandee
ratio = len(CORPUS[:120000]) / max(1, n_jetons(CORPUS[:120000]))
notes = CORPUS[:int(max(0, CTX - 600) * ratio)]
notes = notes[:notes.rfind("\n")] if "\n" in notes else notes
SYSTEME = ("You are an expert coding assistant operating inside a terminal coding harness. You help users by reading files, executing commands, "
           "editing code, and writing new files.\n\nAvailable tools: bash, read, edit, write.\n\nGuidelines:\n- Use bash for file operations like ls, grep, find\n"
           "- Use read to examine files before editing\n- Use edit for precise changes (old text must match exactly)\n- Use write only for new files or complete rewrites\n"
           "- Be concise in your responses\n- Call exactly one tool per turn and wait for its result\n\n# Project Context\n\n" + notes)
ETAPES = ["echo etape-%02d && date +%%s" % (i + 1) for i in range(K)]
CONSIGNE = ("Execute EXACTEMENT les commandes suivantes avec l'outil bash, UNE commande par appel d'outil, dans l'ordre, sans rien ajouter. "
            "Attends le resultat de chaque commande avant de lancer la suivante. Apres la derniere, reponds seulement : FINI.\n\n"
            + "\n".join("%d. %s" % (i + 1, c) for i, c in enumerate(ETAPES)))
LIGNE = "2026-09-30T12:%02d:%02d worker[%d] INFO  batch=%04d items=%d elapsed_ms=%d status=ok checksum=%08x\n"
TAILLES = ["court", "court", "moyen", "court", "long", "court", "moyen", "court", "court", "long", "court", "moyen"]


def resultat(i):
    t = TAILLES[i % len(TAILLES)]
    n = {"court": 0, "moyen": 12, "long": 90}[t]
    corps = "".join(LIGNE % ((i * 7 + j) % 60, (j * 13) % 60, 4000 + j, i * 100 + j, 50 + (j * 37) % 900, 3 + (j * 11) % 400, (i * 2654435761 + j * 40503) & 0xffffffff) for j in range(n))
    return t, "etape-%02d\n17592%05d\n%s" % (i + 1, 40000 + i * 61, corps)


def chat_stream(messages, nom):
    body = {"model": "qwen3.8-flash-next-125b", "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "store": False,
            "max_completion_tokens": 2048, "tools": TOOLS, "reasoning_effort": os.environ.get("TOURS_EFFORT", "medium"), "thinking_budget_tokens": 1024, "temperature": 0,
            "top_k": 20, "top_p": 0.95, "min_p": 0.05, "seed": 42, "cache_prompt": True, "id_slot": 0}
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % PORT, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None
    contenu, raison = [], []; tcs = {}; fin = None; tm = {}; usage = {}
    brut = open(os.path.join(OUT, "%s_tours_%s.sse" % (TAG, nom)), "w", encoding="utf-8")
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip(); brut.write(payload + "\n")
            if payload == "[DONE]": break
            try: d = json.loads(payload)
            except Exception: continue
            if d.get("timings"): tm = d["timings"]
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices") or []:
                de = ch.get("delta") or {}
                if ttft is None and (de.get("content") or de.get("reasoning_content") or de.get("tool_calls")): ttft = time.time() - t0
                if de.get("content"): contenu.append(de["content"])
                if de.get("reasoning_content"): raison.append(de["reasoning_content"])
                for tc in de.get("tool_calls") or []:
                    k = tc.get("index", 0)
                    cur = tcs.setdefault(k, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    cur["id"] = cur["id"] or tc.get("id") or ""
                    f = tc.get("function") or {}
                    cur["function"]["name"] += f.get("name") or ""
                    cur["function"]["arguments"] += f.get("arguments") or ""
                if ch.get("finish_reason"): fin = ch["finish_reason"]
    brut.close()
    return {"content": "".join(contenu), "reasoning": "".join(raison), "tool_calls": [tcs[k] for k in sorted(tcs)], "finish": fin,
            "timings": tm, "usage": usage, "ttft": ttft, "wall": time.time() - t0}


def msg_assistant(rep):
    m = {"role": "assistant", "content": rep["content"] or None}
    if rep["reasoning"] and MODE != "sans_raisonnement": m["reasoning_content"] = rep["reasoning"]
    if rep["tool_calls"]:
        tcs = json.loads(json.dumps(rep["tool_calls"]))
        if MODE == "args_reserialises":
            for tc in tcs:
                try: tc["function"]["arguments"] = json.dumps(json.loads(tc["function"]["arguments"]), separators=(",", ":"), ensure_ascii=False)
                except Exception: pass
        m["tool_calls"] = tcs
    return m


messages = [{"role": "system", "content": SYSTEME}, {"role": "user", "content": [{"type": "text", "text": CONSIGNE}]}]
print("TOURS %s mode=%s K=%d ctx vise=%d : prompt systeme de %d jetons, schema d'outils %d car." % (TAG, MODE, K, CTX, n_jetons(SYSTEME), len(json.dumps(TOOLS))), flush=True)
lignes = []; n_ok = 0
for i in range(K + 2):
    rep = chat_stream(messages, "t%02d" % i)
    tm = rep["timings"]; g = lambda k: tm.get(k, "NON_MES")
    tc = rep["tool_calls"][0] if rep["tool_calls"] else None
    valide = False; cmd = None
    if tc:
        try: a = json.loads(tc["function"]["arguments"]); cmd = a.get("command"); valide = tc["function"]["name"] == "bash" and isinstance(cmd, str)
        except Exception: valide = False
    attendu = ETAPES[i] if i < K else None
    bon = valide and attendu is not None and cmd.strip() == attendu
    n_ok += bool(bon)
    taille = resultat(i)[0] if i < K else "-"
    L = {"tour": i, "ttft": rep["ttft"], "wall": rep["wall"], "prompt_n": g("prompt_n"), "cache_n": g("cache_n"), "pred_n": g("predicted_n"),
         "prompt_ms": g("prompt_ms"), "pred_ms": g("predicted_ms"), "pred_tps": g("predicted_per_second"), "draft_n": g("draft_n"), "draft_acc": g("draft_n_accepted"),
         "fin": rep["finish"], "outil": bool(tc), "valide": valide, "bonne_commande": bon, "raison_car": len(rep["reasoning"]), "resultat_precedent": resultat(i - 1)[0] if 0 < i <= K else "-"}
    lignes.append(L)
    print("TOUR %s %02d ttft=%s mur=%.2fs | prompt_n=%s cache_n=%s pred_n=%s | prompt_ms=%s pred_ms=%s (%s tok/s) | arret=%s outil=%s bon=%s raison=%d car. | resultat precedent=%s"
          % (TAG, i, "%.2fs" % rep["ttft"] if rep["ttft"] is not None else "NON_MES", rep["wall"], L["prompt_n"], L["cache_n"], L["pred_n"],
             "%.0f" % L["prompt_ms"] if isinstance(L["prompt_ms"], (int, float)) else L["prompt_ms"], "%.0f" % L["pred_ms"] if isinstance(L["pred_ms"], (int, float)) else L["pred_ms"],
             "%.1f" % L["pred_tps"] if isinstance(L["pred_tps"], (int, float)) else L["pred_tps"], rep["finish"], bool(tc), bon, len(rep["reasoning"]), L["resultat_precedent"]), flush=True)
    messages.append(msg_assistant(rep))
    if not tc:
        print("TOURS %s : plus d'appel d'outil au tour %d (contenu=%r)" % (TAG, i, rep["content"][:80]), flush=True)
        break
    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": resultat(i)[1] if i < K else "commande hors liste"})
json.dump({"mode": MODE, "lignes": lignes, "messages": messages}, open(os.path.join(OUT, "%s_tours.json" % TAG), "w"), ensure_ascii=False)

# --- bilan : on ecarte le tour 0 (prefill du prefixe entier)
import statistics as st
T = [l for l in lignes[1:] if isinstance(l["pred_n"], int) and l["ttft"] is not None]
if T:
    par = {}
    for l in T: par.setdefault(l["resultat_precedent"], []).append(l)
    gen = sum(l["pred_n"] for l in T); mur = sum(l["wall"] for l in T); dec = sum(l["pred_ms"] for l in T if isinstance(l["pred_ms"], (int, float))) / 1000
    print("BILAN_TOURS %s mode=%s : %d tours mesures, commandes justes %d/%d | jetons generes %d en %.1fs de mur = %.1f tok/s AGENTIQUE (decodage seul : %.1f tok/s) | TTFT median %.2fs | part du mur hors decodage %.0f %%"
          % (TAG, MODE, len(T), n_ok, K, gen, mur, gen / mur, gen / dec if dec else float("nan"), st.median(l["ttft"] for l in T), 100 * (mur - dec) / mur), flush=True)
    for t in ("court", "moyen", "long"):
        if t in par:
            x = par[t]
            print("BILAN_TOURS %s   apres un resultat %-5s : n=%d, jetons nouveaux med %s, TTFT med %.2fs (min %.2f, max %.2f), mur med %.2fs, pred_n med %s"
                  % (TAG, t, len(x), st.median(l["prompt_n"] for l in x if isinstance(l["prompt_n"], int)), st.median(l["ttft"] for l in x), min(l["ttft"] for l in x), max(l["ttft"] for l in x),
                     st.median(l["wall"] for l in x), st.median(l["pred_n"] for l in x)), flush=True)
else:
    print("BILAN_TOURS %s mode=%s : NON_MES (aucun tour exploitable)" % (TAG, MODE), flush=True)
