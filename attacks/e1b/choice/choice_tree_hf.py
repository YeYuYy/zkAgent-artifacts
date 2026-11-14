"""Attention-axis choice-tree runner (HF eager engine + AttnSlack), paper only.

Same admissible-trajectory semantics as experiments/choice_tree.py, but the
reach set at a step is the union over the honest logits and three per-layer
attention-scale perturbations g in {1+rho, 1-rho, adversarial sign}, rho =
7088 / 2^20. This is the E1b-attn over-approximation: each perturbation is a
possible verifier configuration, so the union is a per-step upper bound on the
tokens a tolerated verifier could accept.

Because the HF engine is a single-sequence incremental model, each path step
re-prefills its prefix. That is fine for the small attention subset (n=20-30);
the at-scale exhaustive/beam campaign runs on vLLM in choice_tree.py.

Attention probing is confined to the canonical spine: probe depth is bounded by
--attn-max-steps and fork steps, and spawned branches are not re-probed. This
is the plan's spine over-approximation and keeps the eager backend tractable.

Honest/control references are genuine top-k=40 temperature-1 samples from the
same checkpoint, so the null flip rate remains an independent control here.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from experiments.choice_scoring import choice_summary, parse_choice
from experiments.choice_tree import (SHORT_SUFFIX, choice_from_ids, item_result,
                                     new_path, stable_seed)
from experiments.engine import Engine
from experiments.relation import Relation, coin

RHO = 7088 / (1 << 20)


class HFChoice(Engine):
    def __init__(self, model, temperature=1., seed=0, thinking=True):
        super().__init__(model, temperature, seed)
        self.thinking = thinking
        self.attn = None

    def sample(self, ids, max_new, seed):
        import torch
        # transformers 5.x rejects `generator=` in generate(); seed the global
        # RNG instead (this backend runs one sequence at a time).
        torch.manual_seed(int(seed) & ((1 << 63) - 1))
        x = torch.tensor([ids], device='cuda')
        att = torch.ones_like(x)
        out = self.model.generate(
            input_ids=x, attention_mask=att, max_new_tokens=max_new, do_sample=True,
            temperature=1.0, top_k=40, eos_token_id=sorted(self.eos),
            pad_token_id=self.tok.pad_token_id or sorted(self.eos)[0])
        return [int(t) for t in out[0, x.shape[1]:].tolist()]

    def reach_attn(self, ids, H, honest):
        """Union of paper-mode witness tokens over the three perturbations.
        Each extra token keeps the perturbed relation state + witness that made
        it reachable, so the audit can re-check it without the model."""
        out, extra, probed = set(honest), [], 0
        for name, vec in zip(('coherent+', 'coherent-', 'adversarial'),
                             self.attn.perturbed_logits(ids)[1:]):
            probed += 1
            try:
                r = Relation(vec, self.temperature)
            except ValueError:
                continue
            for t, w in r.witnesses(int(H), 'paper').items():
                t = int(t)
                if t not in out:
                    extra.append(dict(token=t, config=name, witness=w,
                                      act=r.act.tolist(), delta=r.delta.tolist()))
                out.add(t)
        return out, extra, probed


def finish_at(engine, ids, base_len, labels):
    choice, text = choice_from_ids(engine, ids, base_len, labels)
    return dict(done='stop', choice=choice, text=text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--thinking', choices=['on', 'off'], default='off')
    p.add_argument('--attn', choices=['on', 'off'], default='on')
    p.add_argument('--attn-max-steps', type=int, default=32,
                   help='probe attention perturbations only below this depth')
    p.add_argument('--short', action='store_true',
                   help='append a one-sentence + boxed-answer instruction')
    p.add_argument('--max-new', type=int, default=256)
    p.add_argument('--max-steps', type=int, default=2048)
    p.add_argument('--node-limit', type=int, default=20000)
    p.add_argument('--max-paths', type=int, default=256)
    p.add_argument('--census', action='store_true',
                   help='do not stop early at the first flip')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config['backend'] = 'hf-choice-attn'
    config['data_sha256'] = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    cfg = a.out / 'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Refusing mismatched resume')
    cfg.write_text(json.dumps(config, indent=2))
    items = [json.loads(x) for x in Path(a.data).read_text().splitlines()][a.offset:]
    if a.limit:
        items = items[:a.limit]
    (a.out / 'items').mkdir(exist_ok=True)
    done = {f.stem for f in (a.out / 'items').glob('*.json')}
    if len(done) == len(items):
        (a.out / 'COMPLETE').write_text(str(len(items)) + '\n')
        print('Already complete')
        return

    engine = HFChoice(a.model, a.temperature, a.seed, a.thinking == 'on')
    if a.attn == 'on':
        from experiments.vllm_bestofn import AttnEngine
        engine.attn = AttnEngine(a.model, rho=RHO)
    t0 = time.time()
    for i, it in enumerate(items):
        key = hashlib.sha256(it['id'].encode()).hexdigest()[:16]
        if key in done:
            continue
        chat = engine.chat_ids(it['question'] + (SHORT_SUFFIX if a.short else ''),
                               suffix='', thinking=engine.thinking)
        base = {}
        for tag in ('honest', 'control'):
            ids = engine.sample(chat, a.max_new, stable_seed(a.seed, it['id'], tag))
            tr = dict(text=engine.text(ids), capped=len(ids) >= a.max_new)
            base[tag] = dict(tr, choice=parse_choice(tr, it['labels']))
        spine = new_path(0, chat, len(chat), spine=True)
        paths = [spine]
        limits = dict(truncated=False, node_limited=False, spawned=0)
        attn_stats = dict(steps_probed=0, extra_tokens=0, extra_forks=0)
        rounds = 0
        while rounds < a.max_steps:
            active = [x for x in paths if x['done'] is None]
            if not active:
                break
            rounds += 1
            spawn = []
            used = sum(x['nodes'] for x in paths)
            over = used >= a.node_limit
            if over:
                limits['node_limited'] = True
                for x in paths:
                    if x['done'] is None and not x['spine']:
                        x['done'] = 'nodecap'
                active = [x for x in paths if x['done'] is None]
                if not active:
                    break
            for path in active:
                H = coin(path['ids'], a.seed)
                engine.run(path['ids'])
                try:
                    r = Relation(engine.lg, a.temperature)
                    ws = r.witnesses(H, 'paper')
                    reach = set(int(t) for t in ws)
                except ValueError:
                    path['done'] = 'relation'; continue
                if not reach:
                    path['done'] = 'relation'; continue
                wmap = {int(t): dict(witness=ws[t], act=r.act.tolist(),
                                     delta=r.delta.tolist(), config='honest') for t in ws}
                canon = r.canonical(H)
                if canon not in reach:
                    canon = int(max(reach, key=lambda t: engine.lg[t]))
                here = []
                if engine.attn is not None and path['spine'] and (
                        path['depth'] < a.attn_max_steps or len(reach) > 1):
                    reach, here, probed = engine.reach_attn(path['ids'], H, reach)
                    attn_stats['steps_probed'] += probed
                    if here:
                        attn_stats['extra_forks'] += 1
                        attn_stats['extra_tokens'] += len(here)
                    for e in here:
                        wmap.setdefault(e['token'], dict(witness=e['witness'], act=e['act'],
                                                        delta=e['delta'],
                                                        config=e['config']))
                extras = sorted(reach - {canon})
                prefix = list(path['ids'])
                path['ids'].append(int(canon)); path['nodes'] += 1; path['depth'] += 1
                if extras:
                    path['fork_steps'].append(dict(
                        step=path['depth'] - 1, canon=int(canon), extras=extras,
                        H=str(int(H)), act=r.act.tolist(), delta=r.delta.tolist(),
                        witnesses={str(int(t)): wmap[int(t)]['witness'] for t in extras},
                        attn=here if here else None))
                if canon in engine.eos:
                    path.update(finish_at(engine, path['ids'], path['base_len'], it['labels']))
                elif path['depth'] >= a.max_new:
                    path['done'] = 'cap'
                for t in extras:
                    spawn.append((path, t, prefix, wmap.get(t)))
            for parent, t, prefix, dev in spawn:
                if limits['spawned'] >= a.max_paths:
                    limits['truncated'] = True; break
                limits['spawned'] += 1
                q = new_path(0, prefix + [t], len(chat), depth=parent['depth'])
                q['fork_steps'] = list(parent['fork_steps'])
                q['deviations'] = list(parent.get('deviations', [])) + [dict(
                    step=parent['depth'] - 1, token=int(t), **(dev or {}))]
                if t in engine.eos:
                    q.update(finish_at(engine, q['ids'], q['base_len'], it['labels']))
                elif q['depth'] >= a.max_new:
                    q['done'] = 'cap'
                paths.append(q)
            if not a.census:
                hc = base['honest']['choice']
                if any(x['done'] == 'stop' and x['choice'] is not None and hc is not None
                       and x['choice'] != hc for x in paths):
                    for x in paths:
                        if x['done'] is None and not x['spine']:
                            x['done'] = 'pruned'
        else:
            for x in paths:
                if x['done'] is None:
                    x['done'] = 'roundcap'
        tree = item_result(paths, base['honest']['choice'], limits, 'exhaustive', spine)
        tree.update(rounds=rounds, seconds=round(time.time() - t0, 1))
        if a.attn == 'on':
            tree['attn'] = attn_stats
        row = dict(id=it['id'], domain=it['domain'], question=it['question'],
                   gold_choice=it.get('gold_choice'), honest=base['honest'],
                   control=base['control'],
                   canonical=dict(text=spine.get('text'), capped=spine['done'] != 'stop',
                                  choice=spine['choice'], ids=spine['ids'],
                                  base_len=spine['base_len'],
                                  fork_steps=spine.get('fork_steps', [])),
                   conditions={'paper': dict(choice_tree=tree)})
        target = a.out / 'items' / f'{key}.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(row)); tmp.replace(target)
        rows = [json.loads(f.read_text()) for f in (a.out / 'items').glob('*.json')]
        s = choice_summary(rows)
        s['phase_seconds'] = round(time.time() - t0, 1)
        (a.out / 'summary.json').write_text(json.dumps(s, indent=2))
        print(json.dumps(dict(item=i + 1, total=len(items), id=it['id'],
                              seconds=round(time.time() - t0, 1),
                              success=tree['success'], nodes=tree['nodes'],
                              attn=tree.get('attn'))), flush=True)
    (a.out / 'COMPLETE').write_text(str(len(items)) + '\n')


if __name__ == '__main__':
    main()