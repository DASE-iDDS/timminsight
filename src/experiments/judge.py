"""Judging end-to-end answer accuracy (paper Sec. VI-A).

A fixed judge model, not among the evaluated backbones (GPT-4.1,
gpt-4.1-2025-04-14), decides whether a system's answer is consistent with the
gold answer.  For a scalar gold answer the judge compares the final answer
with it.  When the gold answer is a large collection (a set, per-item values,
or a ranking), no final answer can list every item, so the judge also
receives an item-level comparison of the system's full result with the gold
items, computed here, and decides from that evidence:

    set, all_pairs   normalized set comparison of the returned items (F1)
    any_k_of         whether at least k distinct returned items are all correct
    map, table,      rows aligned by key; ids and numbers compared after
    scores           normalization, free-text values by the judge in batches
                     (share of gold rows matched; missing rows are mismatches)
    ranking          Spearman rho over the gold items

Which result column holds the answer: the key (or item) column is the one
whose normalized values overlap the gold keys (items) most; a gold value column
maps to the result column of the same normalized name, otherwise to the column
that agrees with it on the most rows (numerically for numeric gold values, by
token overlap for text).  An unparsable judge reply raises ``ValueError``; the
runner leaves such items ungraded.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

JUDGE_MODEL = "gpt-4.1-2025-04-14"
THRESHOLD = 0.9          # agreement level reported to the judge as near-complete
TEXT_BATCH = 50          # free-text value pairs per judge call
REL_TOL = 0.01           # numbers agree within 1% (or 1e-6 absolute)
SCORE_TOL = 0.5          # per-item scores (gold_kind "scores") agree within half a point

COLLECTION_PROMPT = (
    "You grade answers of a data-analysis system whose gold answer is a collection (a set, "
    "per-item values, or a ranking). You receive the question, the system's final answer, "
    "a sample of the gold items, and an item-level comparison of the system's full result "
    "with all gold items. Decide whether the system's answer is semantically consistent "
    "with the gold answer: its result must contain essentially the same items with the "
    "same values or order; a few minor omissions or formatting differences are acceptable, "
    "missing or wrong items in substance are not. Reply with JSON only: "
    '{"consistent": true|false, "reason": "<one sentence>"}'
)

JUDGE_PROMPT = (
    "You grade answers of a data-analysis system. Given a question, the gold answer, "
    "and the system's answer, decide whether the system's answer is semantically "
    "consistent with the gold answer: it must state the same entities, values, or "
    "conclusion, allowing different wording, formatting, ordering, and harmless "
    "extra detail; numeric values must agree up to rounding. Reply with JSON only: "
    '{"consistent": true|false, "reason": "<one sentence>"}'
)

VALUE_PROMPT = (
    "You grade per-row values extracted by a data-analysis system. For each item, decide "
    "whether the system's value states the same information as the gold value (same "
    "entities, labels, or facts; wording, order, and formatting may differ; a missing or "
    "empty value matches only an empty gold value). Reply with JSON only: a list "
    '[{"id": int, "equivalent": true|false}, ...] covering every id.'
)

_PREFIXES = (
    re.compile(r"^Any \d+ distinct items from this set are correct: "),
    re.compile(r"^Per-item scores \(an answer is correct if its scores agree closely\): "),
    re.compile(r"^All pairs satisfying the rule \(as ordered pairs of distinct reviews\): "),
    re.compile(r"^Reference scores that define the ranking \(higher ranks first\): "),
)
_NULLS = {"", "none", "null", "nan", "n/a", "na", "unknown", "not mentioned", "not specified", "-"}


def create_judge(provider_type: str = "openai", model: str = JUDGE_MODEL):
    from core.query_planner.llm_query_analyzer import create_provider
    return create_provider(provider_type, model=model)


# ---------------------------------------------------------------- scalar: judge on the answer
def judge(provider: Any, question: str, gold: str, answer: str) -> bool:
    """True iff the judge deems ``answer`` semantically consistent with ``gold``."""
    return _verdict(provider.complete([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Question: {question}\nGold answer: {gold}\n"
                                    f"System answer: {answer}"},
    ]))


def _verdict(raw: Any) -> bool:
    m = re.search(r"\{.*\}", str(raw), re.S)
    try:
        verdict = json.loads(m.group(0))["consistent"] if m else None
    except (ValueError, KeyError):
        verdict = None
    if not isinstance(verdict, bool):
        # an unparsable verdict is not a wrong answer; fail loudly instead
        raise ValueError(f"unparsable judge output: {str(raw)[:300]}")
    return verdict


# ---------------------------------------------------------------- normalization
def parse_gold(gold_answer: Any, kind: str) -> Any:
    """The structured gold value of a benchmark item (inverse of the builder's
    render_gold): scalar golds stay strings, the others are JSON."""
    if kind == "scalar" or not isinstance(gold_answer, str):
        return gold_answer
    body = gold_answer
    for p in _PREFIXES:
        body = p.sub("", body, count=1)
    return json.loads(body)


def _number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s):
            return float(s)
    return None


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (
        isinstance(v, str) and v.strip().lower() in _NULLS) or (isinstance(v, (list, dict)) and not v)


def norm(v: Any) -> str:
    """Canonical form of an id or value: numbers without trailing zeros, text
    case-folded with collapsed whitespace, JSON objects with sorted keys."""
    if _is_null(v):
        return ""
    n = _number(v)
    if n is not None:
        return str(int(n)) if n == int(n) and abs(n) < 1e15 else repr(round(n, 6))
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return norm(json.loads(s))
            except ValueError:
                pass
        return re.sub(r"\s+", " ", s).casefold()
    if isinstance(v, dict):
        return json.dumps({str(k).casefold(): norm(x) for k, x in sorted(v.items())},
                          sort_keys=True, ensure_ascii=False)
    if isinstance(v, (list, tuple, set)):
        return json.dumps(sorted(norm(x) for x in v), ensure_ascii=False)
    return re.sub(r"\s+", " ", str(v)).casefold()


def _col_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    cols: List[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    return cols


def _best_column(rows: Sequence[Dict[str, Any]], targets: set,
                 exclude: Iterable[str] = ()) -> Optional[str]:
    """The column whose normalized values overlap ``targets`` most."""
    best, best_hits = None, 0
    for c in _columns(rows):
        if c in exclude:
            continue
        hits = len({norm(r.get(c)) for r in rows} & targets)
        if hits > best_hits:
            best, best_hits = c, hits
    return best


def _f1(returned: set, gold: set, tp: Optional[int] = None, gold_size: Optional[int] = None) -> float:
    gold_size = len(gold) if gold_size is None else gold_size
    tp = len(returned & gold) if tp is None else tp
    if not returned and gold_size == 0:
        return 1.0
    if not returned or gold_size == 0 or tp == 0:
        return 0.0
    p, r = tp / len(returned), tp / gold_size
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------- sets and pairs
def _gold_items(gold: List[Any]) -> Tuple[set, Optional[List[str]]]:
    """Normalized gold items, and the field names when items are JSON objects."""
    items, fields = set(), None
    for g in gold:
        if isinstance(g, str) and g.strip().startswith("{"):
            try:
                g = json.loads(g)
            except ValueError:
                pass
        if isinstance(g, dict):
            fields = fields or [str(k) for k in g]
        items.add(norm(g))
    return items, fields


def _returned_items(rows: Sequence[Dict[str, Any]], gold: set,
                    fields: Optional[List[str]]) -> set:
    if fields:                       # items are objects: project rows on the gold fields
        by_key = {_col_key(c): c for c in _columns(rows)}
        cols = [by_key.get(_col_key(f)) for f in fields]
        if None in cols:
            return set()
        return {norm({f: r.get(c) for f, c in zip(fields, cols)}) for r in rows}
    col = _best_column(rows, gold)
    return {norm(r.get(col)) for r in rows} - {""} if col else set()


def _pair_rule(gold: Dict[str, Any]) -> Tuple[Callable[[str, str], bool], int, set]:
    """Validity test, number of gold (unordered) pairs, and all ids of a pair rule."""
    group = {norm(i): g for g, ids in gold["groups"].items() for i in ids}
    sizes = [len(ids) for ids in gold["groups"].values()]
    if "SAME" in gold["rule"].upper():
        return (lambda a, b: a != b and group[a] == group[b],
                sum(n * (n - 1) // 2 for n in sizes), set(group))
    cross = sum(sizes[i] * sizes[j] for i in range(len(sizes)) for j in range(i + 1, len(sizes)))
    return lambda a, b: group[a] != group[b], cross, set(group)


def _returned_pairs(rows: Sequence[Dict[str, Any]], ids: set) -> set:
    first = _best_column(rows, ids)
    second = _best_column(rows, ids, exclude=[first]) if first else None
    if not second:
        return set()
    return {frozenset((norm(r.get(first)), norm(r.get(second)))) for r in rows
            if norm(r.get(first)) in ids and norm(r.get(second)) in ids
            and norm(r.get(first)) != norm(r.get(second))} | {
        ("invalid", i) for i, r in enumerate(rows)
        if not (norm(r.get(first)) in ids and norm(r.get(second)) in ids)}


def _grade_pairs(rows, gold, k: Optional[int]) -> Tuple[bool, float]:
    valid, gold_size, ids = _pair_rule(gold)
    pairs = _returned_pairs(rows, ids)
    ok = {p for p in pairs if isinstance(p, frozenset) and len(p) == 2 and valid(*sorted(p))}
    if k is not None:                 # any_k_of
        frac = len(ok) / len(pairs) if pairs else 0.0
        return len(ok) >= k and len(ok) == len(pairs), frac
    f1 = _f1(pairs, set(), tp=len(ok), gold_size=gold_size)
    return f1 >= THRESHOLD, f1


# ---------------------------------------------------------------- keyed rows
def _gold_rows(gold: Any, kind: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Gold as {key: {value column: value}} with the key column's name."""
    if isinstance(gold, dict):                        # map: key -> value
        return "key", {norm(k): {"value": v} for k, v in gold.items()}
    cols = list(gold[0]) if gold else []
    key = next((c for c in cols if len({norm(r[c]) for r in gold}) == len(gold)
                and all(_number(r[c]) is None for r in gold)), None)
    key = key or next((c for c in cols if len({norm(r[c]) for r in gold}) == len(gold)), cols[0])
    return key, {norm(r[key]): {c: v for c, v in r.items() if c != key} for r in gold}


def _value_column(rows, key_col: str, gold_col: str, gold: Dict[str, Dict[str, Any]]) -> Optional[str]:
    cols = [c for c in _columns(rows) if c != key_col]
    named = [c for c in cols if _col_key(c) == _col_key(gold_col)]
    if named and gold_col != "value":
        return named[0]
    by_key = {norm(r.get(key_col)): r for r in rows}

    def agreement(c: str) -> float:
        score = 0.0
        for k, g in gold.items():
            r = by_key.get(k)
            if r is None:
                continue
            gv, rv = g[gold_col], r.get(c)
            gn, rn = _number(gv), _number(rv)
            if gn is not None:
                score += rn is not None and math.isclose(gn, rn, rel_tol=REL_TOL, abs_tol=1e-6)
            else:
                gt, rt = set(re.findall(r"\w+", norm(gv))), set(re.findall(r"\w+", norm(rv)))
                score += len(gt & rt) / len(gt | rt) if gt | rt else 1.0
        return score
    return max(cols, key=agreement) if cols else None


def _values_equal(gv: Any, rv: Any, kind: str) -> Optional[bool]:
    """Deterministic comparison; None when free text must go to the judge."""
    if _is_null(gv) or _is_null(rv):
        return _is_null(gv) and _is_null(rv)
    gn, rn = _number(gv), _number(rv)
    if gn is not None or rn is not None:
        if gn is None or rn is None:
            return False
        if kind == "scores":
            return abs(gn - rn) <= SCORE_TOL
        return math.isclose(gn, rn, rel_tol=REL_TOL, abs_tol=1e-6)
    if norm(gv) == norm(rv):
        return True
    if isinstance(gv, list):                   # e.g. genres: compare as sets of labels
        items = rv if isinstance(rv, list) else re.split(r"\s*[,;|]\s*", str(rv))
        if {norm(x) for x in gv} == {norm(x) for x in items} - {""}:
            return True
    return None


def _judge_values(provider: Any, question: str, pairs: List[Tuple[Any, Any]]) -> List[bool]:
    from core.query_planner.llm_query_analyzer import LLMOutputError, complete_json
    out: List[bool] = []
    for start in range(0, len(pairs), TEXT_BATCH):
        chunk = pairs[start:start + TEXT_BATCH]
        items = [{"id": i, "gold": g, "answer": a} for i, (g, a) in enumerate(chunk)]

        def complete(reply: list, n=len(chunk)) -> None:
            got = {o.get("id") for o in reply if isinstance(o, dict)
                   and isinstance(o.get("equivalent"), bool)}
            if got != set(range(n)):
                raise ValueError("every id needs a boolean 'equivalent'")
        try:
            reply = complete_json(provider, [
                {"role": "system", "content": VALUE_PROMPT},
                {"role": "user", "content": f"Question: {question}\n"
                                            + json.dumps(items, ensure_ascii=False, default=str)}],
                list, complete, "value judge")
        except LLMOutputError as exc:          # ungraded, as for the scalar judge
            raise ValueError(f"unparsable judge output: {exc}") from exc
        verdict = {o["id"]: o["equivalent"] for o in reply}
        out.extend(verdict[i] for i in range(len(chunk)))
    return out


def _grade_keyed(provider, question, rows, gold, kind) -> Tuple[bool, float]:
    gold_key, grows = _gold_rows(gold, kind)
    if not grows:
        return not rows, 1.0 if not rows else 0.0
    key_col = _best_column(rows, set(grows))
    if key_col is None:
        return False, 0.0
    by_key = {norm(r.get(key_col)): r for r in rows}
    value_cols = list(next(iter(grows.values())))
    mapping = {gc: _value_column(rows, key_col, gc, grows) for gc in value_cols}
    verdict: Dict[str, bool] = {}
    pending: List[Tuple[str, Any, Any]] = []
    for k, g in grows.items():
        r = by_key.get(k)
        if r is None or any(mapping[gc] is None for gc in value_cols):
            verdict[k] = False
            continue
        ok = True
        for gc in value_cols:
            eq = _values_equal(g[gc], r.get(mapping[gc]), kind)
            if eq is None:
                pending.append((k, g[gc], r.get(mapping[gc])))
            elif not eq:
                ok = False
        verdict[k] = ok
    if pending:
        judged = _judge_values(provider, question, [(g, a) for _, g, a in pending])
        for (k, _, _), eq in zip(pending, judged):
            verdict[k] = verdict[k] and eq
    rate = sum(verdict.values()) / len(grows)
    return rate >= THRESHOLD, rate


# ---------------------------------------------------------------- ranking
def _ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: List[float], b: List[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 0.0


def _grade_ranking(rows, gold) -> Tuple[bool, float]:
    gold_key, grows = _gold_rows(gold, "ranking")
    score_col = next(iter(next(iter(grows.values()))))
    key_col = _best_column(rows, set(grows))
    if key_col is None or len(grows) < 2:
        return False, 0.0
    numeric = [c for c in _columns(rows) if c != key_col
               and all(_number(r.get(c)) is not None for r in rows)]
    named = [c for c in numeric if _col_key(c) == _col_key(score_col)]
    col = (named or numeric or [None])[0]
    returned: Dict[str, float] = {}
    for pos, r in enumerate(rows):            # without a score column, row order ranks
        k = norm(r.get(key_col))
        if k in grows and k not in returned:
            returned[k] = _number(r.get(col)) if col else -float(pos)
    floor = min(returned.values(), default=0.0) - 1.0   # missing gold items rank last
    keys = list(grows)
    rho = spearman([float(_number(grows[k][score_col])) for k in keys],
                   [returned.get(k, floor) for k in keys])
    return rho >= THRESHOLD, rho


# ---------------------------------------------------------------- entry point
def compare_items(provider: Any, item: Dict[str, Any],
                  rows: Optional[Sequence[Dict[str, Any]]]) -> Tuple[bool, float]:
    """Item-level comparison of a full result with a collection gold answer:
    (whether the agreement reaches THRESHOLD, the metric: F1, share of gold
    rows matched, or Spearman rho)."""
    kind = item.get("gold_kind") or "scalar"
    gold = parse_gold(item["gold_answer"], kind)
    rows = [r if isinstance(r, dict) else {"value": r} for r in (rows or [])]
    if kind in ("set", "all_pairs") or kind.startswith("any_"):
        k = int(kind.split("_")[1]) if kind.startswith("any_") else None
        if isinstance(gold, dict) and "groups" in gold:
            return _grade_pairs(rows, gold, k)
        items, fields = _gold_items(gold)
        returned = _returned_items(rows, items, fields)
        if k is not None:
            frac = len(returned & items) / len(returned) if returned else 0.0
            return len(returned) >= k and returned <= items, frac
        f1 = _f1(returned, items)
        return f1 >= THRESHOLD, f1
    if kind == "ranking":
        return _grade_ranking(rows, gold)
    if kind in ("map", "table", "scores"):
        return _grade_keyed(provider, item["question"], rows, gold, kind)
    raise ValueError(f"unknown gold_kind {kind!r}")


_METRIC = {"set": "F1 of returned vs gold items", "all_pairs": "F1 of returned vs gold pairs",
           "ranking": "Spearman rho with the gold ranking",
           "map": "share of gold items whose value matches", "table": "share of gold rows matched",
           "scores": "share of gold items whose score agrees"}


def grade(provider: Any, item: Dict[str, Any], answer: str,
          rows: Optional[Sequence[Dict[str, Any]]]) -> Tuple[bool, Optional[float]]:
    """(correct, metric) of a system's answer to a benchmark item.  The judge
    decides in every case; for a collection gold answer it decides from the
    item-level comparison of the full result (whose metric is returned)."""
    kind = item.get("gold_kind") or "scalar"
    if kind == "scalar":
        return judge(provider, item["question"], item["gold_answer"], answer), None
    near, metric = compare_items(provider, item, rows)
    gold = parse_gold(item["gold_answer"], kind)
    size = len(gold) if hasattr(gold, "__len__") else "?"
    sample = json.dumps(gold if not isinstance(gold, (list, dict)) else
                        (gold[:10] if isinstance(gold, list) else dict(list(gold.items())[:10])),
                        ensure_ascii=False, default=str)[:3000]
    name = ("share of returned items that are correct (at least "
            f"{kind.split('_')[1]} needed)" if kind.startswith("any_") else _METRIC.get(kind, "agreement"))
    evidence = (f"Gold answer: {size} items ({kind}); first items: {sample}\n"
                f"The system returned {len(rows or [])} result rows. Item-level comparison: "
                f"{name} = {metric:.3f} ({'at or above' if near else 'below'} the "
                f"{THRESHOLD} level of near-complete agreement).")
    verdict = _verdict(provider.complete([
        {"role": "system", "content": COLLECTION_PROMPT},
        {"role": "user", "content": f"Question: {item['question']}\n{evidence}\n"
                                    f"System answer: {answer}"},
    ]))
    return verdict, metric
