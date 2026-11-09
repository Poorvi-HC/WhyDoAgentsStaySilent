"""Causal checks on individual agents (Block 4).

resample: re-run the exact same decision k times -> P(escalate | context).
ablate:   change one channel post (remove it, or strip it to its bare verdict) and re-run k times.
"""
import json, os
from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .backends import parse_json
from .cells import CELLS
from .run import label_decision
from .scenario import documents


def _rebuild_user(rec, posts, evidence_keys):
    cell = CELLS[rec["cell"]]
    route = cell.get("route", "dedicated")
    docs = documents(evidence_keys, "dedicated" if route.startswith("dedicated") else route)
    tools = P.TOOLS_BY_ROUTE[route]
    n_others = rec["meta"].get("n_others", 0)
    if cell["mode"] in ("alone", "aware"):
        return P.decide_prompt(docs, n_others, [], self_name=None, channel=False, tools=tools)
    return P.decide_prompt(docs, n_others, posts, self_name=rec["agent"], channel=True, tools=tools)


def _run_k(backend, system, user, k, workers):
    def one(_):
        out = backend.chat(system, user)
        if out.get("error"):
            return None  # infrastructure failure: excluded from the rate
        j, _ = parse_json(out.get("content", ""))
        return bool(label_decision(j).get("escalated"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = [x for x in ex.map(one, range(k)) if x is not None]
    return (sum(res) / len(res) if res else float("nan")), res


def pick_targets(calls_path, judged_path=None, n_escalated=15, n_declined=10,
                 cells=("C3-T", "C3-T-adv", "C5-T", "C3-M")):
    """Escalators from channel cells whose channel contains at least one other live post, plus
    agents whose private intent was 'escalate' but who did not escalate (considered-but-declined)."""
    from .run import load_valid
    _, all_dec, all_posts = load_valid(calls_path)
    posts = {(r["run_id"], r["agent"]): str((r.get("parsed") or {}).get("intent", "")).lower()
             for r in all_posts if isinstance(r.get("parsed"), dict)}
    dec = [r for r in all_dec if r["cell"] in cells]
    esc = [r for r in dec if (r.get("decision") or {}).get("escalated") and len(r["meta"].get("posts", [])) >= 2]
    declined = [r for r in dec if not (r.get("decision") or {}).get("escalated")
                and posts.get((r["run_id"], r["agent"])) == "escalate"]
    return esc[:n_escalated], declined[:n_declined]


def already_done(out_path):
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                j = json.loads(l)
                done.add((j.get("run_id"), j["agent"]))
            except Exception:
                pass
    return done


def analyse_agent(rec, backend, out_path, k=10, workers=8, evidence_keys=None):
    """Resample, then ablate each other agent's post one at a time (remove, and verdict-only).
    Skips agents already analysed, so it can be re-run after a crash."""
    from .run import repair_tail
    repair_tail(out_path)
    if (rec.get("run_id"), rec["agent"]) in already_done(out_path):
        return None
    docs = evidence_keys  # passed through; _rebuild_user picks route-specific docs
    base_posts = rec["meta"].get("posts", [])
    results = []
    p, _ = _run_k(backend, rec["system"], _rebuild_user(rec, base_posts, docs), k, workers)
    results.append(dict(variant="resample", p_escalate=p))
    for i, post in enumerate(base_posts):
        if post["author"] == rec["agent"]:
            continue
        removed = base_posts[:i] + base_posts[i + 1:]
        p, _ = _run_k(backend, rec["system"], _rebuild_user(rec, removed, docs), k, workers)
        results.append(dict(variant="remove", post_index=i, author=post["author"], text=post["text"], p_escalate=p))
        verdict = dict(post, text=("I'm escalating Rx-07." if "escalat" in post["text"].lower()
                                   else "I'd let Rx-07 go through."))
        swapped = base_posts[:i] + [verdict] + base_posts[i + 1:]
        p, _ = _run_k(backend, rec["system"], _rebuild_user(rec, swapped, docs), k, workers)
        results.append(dict(variant="verdict_only", post_index=i, author=post["author"], p_escalate=p))
    out = dict(run_id=rec.get("run_id"), cell=rec["cell"], trial=rec["trial"], agent=rec["agent"],
               original_label=(rec.get("decision") or {}).get("label"), k=k, results=results)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(out) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return out
