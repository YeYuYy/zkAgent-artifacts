"""Diagnose forced-answer label probabilities at a real prefix boundary."""
import argparse
import json
from pathlib import Path

from experiments.choice_tree import (ChoiceEngine, FORCED_CUE, SHORT_SUFFIX,
                                     stable_seed)


class A:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--max-new', type=int, default=128)
    a = p.parse_args()
    it = [json.loads(x) for x in Path(a.data).read_text().splitlines()][a.index]
    cfg = A()
    cfg.model, cfg.temperature, cfg.seed = a.model, 1., a.seed
    cfg.thinking, cfg.max_seqs, cfg.max_new, cfg.rollout = 'off', 64, a.max_new, a.max_new
    e = ChoiceEngine(cfg)
    labels = it['labels']
    lids = e.label_ids(labels)
    print('labels', labels, 'label_ids', lids,
          {lab: [e.text([t]) for t in ids] for lab, ids in lids.items()}, flush=True)
    chat = e.chat(it['question'] + SHORT_SUFFIX)
    ids, _ = e.sample([chat], [stable_seed(a.seed, it['id'], 'honest')], a.max_new)[0]
    print('sample text:', e.text(ids)[:300], flush=True)
    cue = e.cue_ids
    allowed = sorted({t for lab in labels for t in lids[lab]})
    for k in sorted({len(ids) // 2, max(len(ids) - 2, 1), len(ids)}):
        pref = chat + ids[:k]
        lp = e.next_logprobs([pref + cue])[0]
        top = sorted(lp.items(), key=lambda x: -x[1])[:10]
        print(f'\n-- prefix k={k} last_text={e.text(ids[max(0, k-6):k])!r}')
        print('top10', [(t, e.text([t]), round(v, 3)) for t, v in top], flush=True)
        print('label hits', {lab: {t: lp.get(t) for t in ids_} for lab, ids_ in lids.items()},
              flush=True)
        if allowed:
            SP = e.SP
            sp = SP(temperature=1., max_tokens=1, logprobs=max(len(allowed), 1),
                    allowed_token_ids=allowed, detokenize=False)
            o = e.llm.generate([{'prompt_token_ids': pref + cue}], [sp], use_tqdm=False)[0]
            d = o.outputs[0].logprobs
            got = {int(t): float(v.logprob) for t, v in d[0].items()} if d else None
            print('allowed logprobs', {t: (e.text([t]), got.get(t)) for t in allowed}
                  if got else got, flush=True)


if __name__ == '__main__':
    main()