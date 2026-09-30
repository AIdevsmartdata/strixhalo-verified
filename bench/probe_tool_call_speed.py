#!/usr/bin/env python3
"""probe_tool_call_speed.py <port> <tag> <out_dir> [reps=2] -- decode speed INSIDE a long tool call (grammar active).
The model is asked for the same ~60-line module twice (the two outputs differ slightly: ~530 vs ~740 tokens):
  F  as a write(path, content) tool call, the code in content   (the tool-call grammar is active over all the code)
  G  as plain text in a markdown block, no tools                (reference)
Measures decode tok/s, tokens per step and ms per step. Thinking is disabled to isolate code generation.
On a runtime that applies the grammar to the whole vocabulary at every verified position, F runs at about half of G.
Output line (labels in French): LONG <tag> bras=<arm> r<rep>: tok/s, pred_n, pas = steps, ms/pas = ms per step,
jetons/pas = tokens per step, arret = finish reason, appel_outil = tool call present, code = characters of code."""
import json, os, sys, time, urllib.request
PORT, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(OUT, exist_ok=True)
WRITE = [{"type": "function", "function": {"name": "write", "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file to write"},
                                                           "content": {"type": "string", "description": "Content to write to the file"}}, "required": ["path", "content"]}}}]
TACHE = ("Ecris un module Python d'environ 60 lignes, avec docstrings, qui implemente une file de priorite a base de tas binaire "
         "(classe TasBinaire avec les methodes inserer, extraire_min, voir_min, __len__ et une fonction de test).")


def une(bras, rep):
    if bras == "F":
        msgs = [{"role": "user", "content": TACHE + " Enregistre-le avec l'outil write dans /tmp/tas.py. N'ecris rien d'autre."}]
    else:
        msgs = [{"role": "user", "content": TACHE + " Donne uniquement le code, dans un seul bloc markdown python, sans aucun texte autour."}]
    b = {"messages": msgs, "max_tokens": 1500, "temperature": 0, "seed": 42, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    if bras == "F": b["tools"] = WRITE
    r = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % PORT, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as resp: d = json.loads(resp.read())
    dt = time.time() - t0; tm = d.get("timings", {}); ch = d["choices"][0]; m = ch.get("message", {})
    json.dump(d, open(os.path.join(OUT, "%s_long_%s_%d.json" % (TAG, bras, rep)), "w"), ensure_ascii=False)
    n, acc = tm.get("predicted_n"), tm.get("draft_n_accepted")
    pas = (n - acc) if isinstance(n, int) and isinstance(acc, int) else None
    tc = m.get("tool_calls") or []
    taille = len(json.loads(tc[0]["function"]["arguments"]).get("content", "")) if tc else len(m.get("content") or "")
    print("LONG %s bras=%s r%d : %s tok/s, pred_n=%s, pas=%s, %s ms/pas, %s jetons/pas | arret=%s appel_outil=%s code=%d car. (%.1fs)"
          % (TAG, bras, rep, "%.1f" % tm["predicted_per_second"] if tm.get("predicted_per_second") else "NON_MES", n, pas,
             "%.1f" % (tm["predicted_ms"] / pas) if pas else "NON_MES", "%.2f" % (n / pas) if pas else "NON_MES", ch.get("finish_reason"), bool(tc), taille, dt), flush=True)


for rep in range(REPS):
    for bras in ("G", "F"): une(bras, rep)
