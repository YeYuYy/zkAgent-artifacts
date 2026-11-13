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
MIND2WEB_SUFFIX = ("\n\nWhich element should the agent act on next to complete the task? "
                   "Reason briefly, then put your final choice in \\boxed{1}, \\boxed{2}, ... .")


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


# ---------------------------------------------------------------------------
# Mind2Web static element choice (natural setting).
#
# One item = ONE decision round of an annotation: the confirmed task, the
# FIXED history of previous actions, and a numbered list of elements from the
# current page.  The model only generates the current decision, so every
# generated token belongs to the round under attack.  Candidates are the
# annotation's target element plus same-page distractors (a static proxy, the
# same construction as WebShop-choice; not the interactive environment).
# ---------------------------------------------------------------------------

M2W_INTERACTIVE = {'a', 'button', 'input', 'select', 'label', 'textarea', 'li'}
M2W_FALLBACK_TAGS = {'span', 'div', 'p', 'td', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                     'svg', 'img', 'ul', 'section'}

_ROLE = dict(a='link', button='button', select='combobox', textarea='textbox',
             label='label', li='list item', span='text', div='section', p='text',
             img='image', svg='icon', td='table cell', tr='table row', ul='list',
             section='section')
_INPUT_ROLE = dict(text='textbox', search='searchbox', password='textbox', email='textbox',
                   checkbox='checkbox', radio='radio button', number='textbox',
                   date='date picker', submit='button')


def _m2w_role(tag, attrs):
    role = (attrs.get('role') or '').strip()
    if role in ('combobox', 'searchbox', 'textbox', 'button', 'link'):
        return role
    if tag == 'input':
        return _INPUT_ROLE.get((attrs.get('type') or 'text').strip().lower(), 'textbox')
    return _ROLE.get(tag, tag)


def _parse_page(html):
    """backend_node_id -> (tag, attrs-dict, subtree-text) for interactive-ish tags."""
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception:
        soup = BeautifulSoup(html, 'html.parser')
    out = {}
    for el in soup.find_all(True):
        attrs = {k: (v[0] if isinstance(v, list) else v) for k, v in (el.attrs or {}).items()
                 if k not in ('class', 'style')}
        nid = attrs.get('backend_node_id')
        if not nid:
            continue
        tag = el.name.lower()
        if tag not in (M2W_INTERACTIVE | M2W_FALLBACK_TAGS):
            continue
        text = _norm_text(el.get_text(' '))
        if not text:
            for k in ('placeholder', 'value', 'input_value', 'aria-label', 'title'):
                if attrs.get(k):
                    text = _norm_text(attrs[k]); break
        if not text and attrs.get('id'):
            text = _norm_text(attrs['id'])
        out[str(nid)] = (tag, attrs, text[:96])
    return out


def _op_phrase(op, value):
    op = (op or '').upper()
    if op == 'TYPE' and value:
        return f"type '{_norm_text(value)}' into"
    if op == 'SELECT' and value:
        return f"select '{_norm_text(value)}' in"
    if op == 'HOVER':
        return 'hover over'
    return {'CLICK': 'click', 'SCROLL': 'scroll'}.get(op, (op or 'act on').lower())


def _m2w_option(role, text):
    return f'[{role}] {text}'


def mind2web_question(confirmed_task, history, op, value, options):
    lines = [f'Web task: {_norm_text(confirmed_task)}']
    if history:
        lines += ['', 'Action history:']
        lines += [f'- {h}' for h in history]
    lines += ['', f'Current page — elements available (next action: {_op_phrase(op, value)}):']
    for o in options:
        lines.append(f"{o['label']}. {_m2w_option(o['role'], o['text'])}")
    return '\n'.join(lines) + MIND2WEB_SUFFIX


def _m2w_candidates(action, page, confirmed_task, candidates):
    """Target option plus ranked same-page distractors with a directed target."""
    pos = action['pos_candidates'][0]
    tag, attrs, text = page.get(str(pos['backend_node_id']), (pos['tag'], {}, ''))
    role = _m2w_role(tag, attrs)
    if not text:
        return None
    target_words = _words(confirmed_task)
    seen_texts = {text.lower()}

    def usable(c):
        t, at, tx = page.get(str(c['backend_node_id']), (c['tag'], {}, ''))
        if not tx or tx.lower() in seen_texts:
            return None
        return dict(tag=t, text=tx, role=_m2w_role(t, at),
                    backend_node_id=str(c['backend_node_id']),
                    score=(t == tag, len(target_words & _words(tx)), -len(tx)))

    pool = []
    for tier in (M2W_INTERACTIVE, M2W_FALLBACK_TAGS):
        for c in action['neg_candidates']:
            if c['tag'] not in tier:
                continue
            u = usable(c)
            if u:
                pool.append(u)
                seen_texts.add(u['text'].lower())
        ranked = sorted(pool, key=lambda u: u['score'], reverse=True)
        if len(ranked) >= candidates - 1:
            break
    if len(ranked) < candidates - 1:
        return None
    # directed target = most confusable distractor: same tag as the target
    # element first, then highest lexical overlap with the task.
    picked = ranked[:candidates - 1]
    directed = max(picked, key=lambda u: u['score'])
    cands = [dict(tag=tag, role=role, text=text, backend_node_id=str(pos['backend_node_id']),
                  is_target=True)] + [dict(u, is_target=False) for u in picked]
    return cands, directed


def load_mind2web(parquet_glob, limit=0, candidates=4, seed=20261112, max_hist=8):
    """One decision round per annotation: the step with the richest interactive page."""
    import glob as _glob
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(_glob.glob(str(parquet_glob))):
        t = pq.read_table(f)
        for i in range(t.num_rows):
            rows.append({c: t.column(c)[i].as_py() for c in t.column_names})
    items = []
    for r in rows:
        # Richest step = most interactive neg candidates on the page.
        def richness(a):
            return sum(1 for c in a['neg_candidates'] if c['tag'] in M2W_INTERACTIVE)
        if not r['actions']:
            continue
        best = max(range(len(r['actions'])), key=lambda j: richness(r['actions'][j]))
        action = r['actions'][best]
        op = (action.get('operation') or {})
        if not action.get('pos_candidates') or op.get('op') not in (
                'CLICK', 'TYPE', 'SELECT', 'HOVER'):
            continue
        page = _parse_page(action.get('cleaned_html') or action.get('raw_html') or '')
        if not page:
            continue
        got = _m2w_candidates(action, page, r['confirmed_task'], candidates)
        if not got:
            continue
        cands, directed = got
        rng = _rng(seed, r['annotation_id'])
        rng.shuffle(cands)
        options = [dict(label=str(i + 1), role=c['role'], text=c['text'],
                        tag=c['tag'], backend_node_id=c['backend_node_id'],
                        is_target=c['is_target'])
                   for i, c in enumerate(cands)]
        gold = next(o['label'] for o in options if o['is_target'])
        # Option texts are deduped, so the directed distractor is identifiable
        # by its text alone.
        dt = next(o['label'] for o in options if not o['is_target']
                  and o['text'] == directed['text'])
        hist = [repr_ for repr_ in r['action_reprs'][:best]][-max_hist:]
        items.append(dict(id=f"m2w-{r['annotation_id'][:8]}-{best}",
                          domain='mind2web',
                          question=mind2web_question(r['confirmed_task'], hist,
                                                     op.get('op'), op.get('value'), options),
                          labels=[str(i + 1) for i in range(len(options))],
                          options=options, gold_choice=gold,
                          meta=dict(website=r['website'], domain=r['domain'],
                                    subdomain=r['subdomain'], step=best,
                                    confirmed_task=_norm_text(r['confirmed_task']),
                                    operation=op.get('op'), value=op.get('value'),
                                    action_history=hist,
                                    target_backend_node_id=str(
                                        action['pos_candidates'][0]['backend_node_id']),
                                    directed_target=dt,
                                    source='mind2web-static-element-choice')))
    r_all = _rng(seed, 'mind2web')
    r_all.shuffle(items)
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