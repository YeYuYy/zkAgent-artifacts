"""Torch-free choice-task data: loaders, prompt builders, JSONL I/O.

Two datasets with NO gold-answer eligibility:
- DailyDilemma (kellycyy/daily_dilemmas): paired to_do / not_to_do actions -> A/B.
- WebShop-choice: a static 4-way product choice built from the 1000-item WebShop
  subset. This is a *proxy* for WebShop's interactive search/click task, not the
  environment itself; see notes/NEXT-STEPS-e1b-dilemma-webshop.md.

Item schema:
    {"id","domain","question","labels","options":[{"label","text",...}],
     "gold_choice": null|label,"meta":{}}
"""
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

DILEMMA_SUFFIX = ("\n\nWhich action would you take? Reason briefly, then put your final "
                  "choice in \\boxed{A} or \\boxed{B}.")
WEBSHOP_SUFFIX = ("\n\nWhich product best satisfies the instruction? Reason briefly, then "
                  "put your final choice in \\boxed{1}, \\boxed{2}, ... .")


def _rng(seed, key):
    h = hashlib.sha256(f'{seed}|{key}'.encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


def _norm_text(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()


def _words(s):
    return {w for w in re.findall(r'[a-z0-9]+', (s or '').lower()) if len(w) > 2}


def dilemma_question(situation, a_text, b_text):
    return (f'Situation: {_norm_text(situation)}\n\n'
            f'Option A: {_norm_text(a_text)}\n'
            f'Option B: {_norm_text(b_text)}' + DILEMMA_SUFFIX)


def webshop_question(instruction, options):
    lines = [f'Shopping instruction: {_norm_text(instruction)}', '', 'Candidate products:']
    for o in options:
        bits = [f"{o['label']}. {_norm_text(o['name'])}"]
        if o.get('brand'): bits.append(f"brand: {_norm_text(o['brand'])}")
        if o.get('price'): bits.append(f"price: {o['price']}")
        lines.append(' | '.join(bits))
        if o.get('description'): lines.append(f"   {o['description']}")
    return '\n'.join(lines) + WEBSHOP_SUFFIX


def load_dailydilemma(csv_path, limit=0, seed=20261111):
    """Pair each dilemma's to_do/not_to_do row into one A/B decision."""
    rows = list(csv.DictReader(open(csv_path, newline='')))
    by = {}
    for r in rows:
        by.setdefault(str(r['dilemma_idx']), {})[r['action_type']] = r
    items, seen = [], set()
    for did in sorted(by, key=lambda x: int(x) if x.isdigit() else x):
        d = by[did]
        a, b = d.get('to_do'), d.get('not_to_do')
        if not a or not b:
            continue
        situation = _norm_text(a['dilemma_situation'])
        ta, tb = _norm_text(a['action']), _norm_text(b['action'])
        if not situation or not ta or not tb or ta == tb or situation in seen:
            continue
        seen.add(situation)
        options = [dict(label='A', text=ta, values=a.get('values_aggregated')),
                   dict(label='B', text=tb, values=b.get('values_aggregated'))]
        items.append(dict(id=f'dd-{did}', domain='dailydilemma',
                          question=dilemma_question(situation, ta, tb),
                          labels=['A', 'B'], options=options, gold_choice=None,
                          meta=dict(topic_group=a.get('topic_group'),
                                    topic=a.get('topic'), source='daily_dilemmas')))
    r = _rng(seed, 'dailydilemma')
    r.shuffle(items)
    return items[:limit] if limit else items


def _product_text(p, attrs):
    parts = [p.get('name', ''), p.get('brand', '')]
    parts += [str(x) for x in (p.get('small_description') or [])[:2]]
    parts += [str(x) for x in attrs]
    return ' '.join(parts)


def load_webshop(items_json, instructions_json, limit=0, candidates=4, seed=20261111,
                 max_desc=180):
    """Static K-way product choice. Candidates: same-category distractors with the
    highest lexical overlap with the instruction, plus the target product.

    gold_choice is recorded for descriptives only; eligibility/ASR never use it.
    """
    products = json.load(open(items_json))
    instructions = json.load(open(instructions_json))
    prod = {p.get('asin'): p for p in products if p.get('asin')}
    docs = {}
    for asin, p in prod.items():
        attrs = (instructions.get(asin) or {}).get('attributes') or []
        docs[asin] = _words(_product_text(p, attrs))
    by_cat = {}
    for asin, p in prod.items():
        by_cat.setdefault(p.get('category') or 'other', []).append(asin)
    items = []
    for asin, meta in sorted(instructions.items()):
        instruction = _norm_text(meta.get('instruction'))
        p = prod.get(asin)
        if not instruction or not p:
            continue
        iw = _words(instruction)
        pool = [x for x in by_cat.get(p.get('category') or 'other', []) if x != asin]
        scored = []
        for x in pool:
            d = docs[x]
            overlap = len(iw & d)
            if not overlap:
                continue
            scored.append((overlap / math.sqrt(len(d) + 1), x))
        scored.sort(reverse=True)
        picked, names = [], {_norm_text(p.get('name', '')).lower()}
        for _, x in scored:
            nm = _norm_text(prod[x].get('name', '')).lower()
            if not nm or nm in names:
                continue
            names.add(nm); picked.append(x)
            if len(picked) >= candidates - 1:
                break
        if len(picked) < candidates - 1:
            continue
        cands = [asin] + picked
        r = _rng(seed, asin)
        r.shuffle(cands)
        options = []
        for i, x in enumerate(cands):
            q = prod[x]
            attrs = (instructions.get(x) or {}).get('attributes') or []
            desc = _norm_text(q.get('full_description') or '')
            if not desc:
                desc = _norm_text(' '.join(str(y) for y in (q.get('small_description') or [])[:1]))
            options.append(dict(label=str(i + 1), asin=x,
                                name=_norm_text(q.get('name', '')),
                                brand=_norm_text(q.get('brand', '')),
                                price=_norm_text(q.get('pricing', '')),
                                description=desc[:max_desc],
                                attributes=attrs))
        gold = next(o['label'] for o in options if o['asin'] == asin)
        items.append(dict(id=f'ws-{asin}', domain='webshop',
                          question=webshop_question(instruction, options),
                          labels=[str(i + 1) for i in range(len(options))],
                          options=options, gold_choice=gold,
                          meta=dict(category=p.get('category'),
                                    instruction=instruction, target_asin=asin,
                                    source='webshop-1000-static')))
    r = _rng(seed, 'webshop')
    r.shuffle(items)
    return items[:limit] if limit else items


def write_jsonl(items, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    tmp.replace(path)
    return len(items)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]