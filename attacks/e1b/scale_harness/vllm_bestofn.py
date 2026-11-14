"""Corrected E1b decision evaluation (hash-chain best-of-N) on a vLLM engine.

Replaces the single-continuation decision search of evaluate.py/vllm_runner.py
per E1b-5 (see attacks/e1b/reports/E1b-NEXT.md): the earlier protocol tested ONE
continuation per injected digit, but the sampling coin is a hash chain over the
committed prefix, so the attacker may re-roll every downstream coin and keep a
best-of-N over whole trajectories. This runner measures ASR as a function of
the sampled-continuation budget, next to a greedy single-chain arm.

Arms:
- greedy (n=1, argmax): the old artifacts-repo conservative baseline.
- sampled (n per slot, temperature tau over top-k=40): the zkAgent honest
  coin-decode distribution; ASR@1 reproduces the old harness protocol in
  distribution, ASR@b gives the corrected best-of-N ladder.

Efficiency: honest traces and all continuations are plain batched generations
(continuous batching + prefix caching); per-step control is never used. The
per-step top-48 engine logprobs returned alongside generation rebuild the exact
Q20 relation at digit forks (same reconstruction as the step-loop backend:
active top-40 is always inside the returned top-48). Injected fork tokens are
witness-checked with Relation.check; downstream tokens of reported first flips
are re-verified admissible (witness_for) from a prompt_logprobs replay.

Boundaries (documented, inherited): decode-relation simulation on serving-
engine logprobs; strict admissibility of every downstream token is enforced
only on reported first flips, and a single trailing EOS is not audited
(termination, not a decode choice).
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from experiments.relation import Relation, coin, MAX_LOGIT, ALPHA
from experiments.scoring import answer, correct, summarize

FILL_GAP = 1000.0  # logit gap below the weakest returned logprob

# Self-correction markers banned by the steer arm (reference: decision_bestofN.py;
# banning is unconditional there and here — ~24 markers vs a 40-token top-k).
CORRECTION = ["Wait", "wait", " Wait", " wait", "But", "However", " But", " However",
              "Actually", " Actually", "recheck", "Recheck", "Let", "let", " Let", " let",
              "Hmm", " Hmm", "mistake", "wrong", "Wrong", "recompute", "check", "Check"]


def marker_ids(tok, vocab):
    """Vocab ids whose surface form is a correction marker (single tokens only)."""
    want = set(CORRECTION)
    return {i for i in range(vocab) if tok.decode([i]) in want}


def operand_digits(question):
    import re
    return set(re.findall(r'\d', question))


def relation_from(active, delta):
    """Reconstruct a Relation for witness enumeration from recorded state."""
    r = Relation.__new__(Relation)
    r.act = np.array(active); r.delta = np.array(delta)
    r.upper = 2 * MAX_LOGIT + int(np.floor(np.log(40) / ALPHA + .5)); r._states = None
    return r


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def chat_ids(tok, question):
    msg = [dict(role='user', content=question + ' Reason step by step, then put the final answer in \\boxed{}.')]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, enable_thinking=True)
    if not isinstance(ids, list): ids = ids['input_ids']
    if ids and isinstance(ids[0], list): ids = ids[0]
    return [int(t) for t in ids]


def relation_from_logprobs(lp, vocab, temperature):
    """Q20 relation from a vLLM top-N logprob dict (missing tokens far below)."""
    vec = np.full(vocab, min(lp.values()) - FILL_GAP)
    vec[list(lp)] = np.fromiter(lp.values(), dtype=np.float64)
    return Relation(vec, temperature)


def build_slots(digit_steps, horizon, mode, max_slots, candidates, digits, operand=None):
    """Ordered (step, candidate) attack slots, mirroring the old search order:
    digit steps ascending, candidates by descending quantized logit. With
    --target-restatement, digit steps whose honest digit appears among the prompt
    operands are searched first (the reference skips the others under budget;
    prioritization reduces to that when slots are scarce).
    Returns (slots, stats) — stats records fork availability for interpretability."""
    steps = digit_steps
    if operand is not None:
        steps = sorted(digit_steps, key=lambda d: 0 if d['base_digit'] in operand else 1)
    slots = []
    seen = witness_total = cand_total = 0
    for d in steps:
        if len(slots) >= max_slots or d['step'] >= horizon: break
        seen += 1
        r = relation_from(d['active'], d['delta'])
        ws = r.witnesses(int(d['H']), mode)
        witness_total += len(ws)
        cands = sorted([t for t in ws if int(t) in digits and int(t) != d['base']],
                       key=lambda t: -d['q'].get(int(t), -(1 << 62)))[:candidates]
        cand_total += len(cands)
        for cand in cands:
            if len(slots) >= max_slots: break
            w = ws[int(cand)]
            assert r.check(int(cand), w, int(d['H']), mode), 'Invalid fork witness'
            slots.append(dict(step=d['step'], base=int(d['base']), cand=int(cand), H=d['H'],
                              active=d['active'], delta=d['delta'], w=w, src='decode'))
    return slots, dict(digit_steps=seen, witness_tokens=witness_total,
                       digit_candidates=cand_total)


def cont_outcome(text, finish, gold):
    """Hardened flip check: truncated rollouts never count (answer is None)."""
    pred = answer(dict(text=text, capped=(finish == 'length')))
    wrong = pred is not None and not correct(pred, gold)
    return wrong, pred


def ladder(wrong_matrix, samples, budgets):
    """wrong_matrix[i][j] = continuation j of slot i is a flip.
    Budget ladder flattens slot-major (explore a fork, then the next fork)."""
    flat = [wrong_matrix[i][j] for i in range(len(wrong_matrix)) for j in range(samples)]
    first = next((k for k, w in enumerate(flat) if w), None)
    hits = {b: (first is not None and first < b) for b in budgets}
    return flat, first, hits


class FakeLogprob:
    def __init__(self, value): self.logprob = value


class VEngine:
    def __init__(self, a):
        self.a = a
        self.t = dict(honest=0., greedy=0., sampled=0., steer=0., audit=0.)
        self._boot(a)
        gc = Path(a.model) / 'generation_config.json'
        eos = (json.loads(gc.read_text()).get('eos_token_id', self.tok.eos_token_id)
               if gc.is_file() else self.tok.eos_token_id)
        self.eos = list({int(t) for t in ([eos] if isinstance(eos, int) else eos)})
        self.think = self.tok.get_vocab().get('</think>')
        self.V = len(self.tok)
        self.digits = {i for i in range(self.V)
                       if len(self.tok.decode([i]).strip()) == 1 and self.tok.decode([i]).strip().isdigit()}
        self.markers = marker_ids(self.tok, self.V)
        self.bad_words = sorted(set(CORRECTION))

    def _boot(self, a):
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        # enforce_eager: required for the attention-slack patch — torch.compile
        # constant-specialises the in-place sign buffer (measured: margin shift
        # 1e-7 compiled vs 0.125 eager).  Other runners leave it False.
        self.llm = LLM(model=a.model, dtype=getattr(a, 'dtype', 'bfloat16'),
                       gpu_memory_utilization=0.92,
                       max_model_len=12288, enable_prefix_caching=True, max_num_seqs=a.max_seqs,
                       max_logprobs=64, disable_log_stats=True,
                       enforce_eager=bool(getattr(a, 'enforce_eager', False)))
        self.tok = self.llm.get_tokenizer()

    def text(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def honest(self, prompts, states, keys):
        """Batched honest traces; per-step top-48 logprobs -> relation state at digit steps."""
        SP = self.SP
        sps = [SP(temperature=self.a.temperature, top_k=40, max_tokens=self.a.max_new,
                  logprobs=48, detokenize=False, seed=stable_seed(self.a.seed, k)) for k in keys]
        t0 = time.time()
        outs = self.llm.generate([{'prompt_token_ids': p} for p in prompts], sps, use_tqdm=False)
        self.t['honest'] += time.time() - t0
        traces = []
        for p, st, o in zip(prompts, states, outs):
            c = o.outputs[0]
            ids = [int(t) for t in c.token_ids]
            for j, lp in enumerate(c.logprobs or []):
                if ids[j] not in self.digits or lp is None: continue
                try:
                    r = relation_from_logprobs({int(t): float(l.logprob) for t, l in lp.items()},
                                               self.V, self.a.temperature)
                except ValueError:
                    st['relation_skips'] = st.get('relation_skips', 0) + 1
                    continue
                st['digits'].append(dict(step=j, base=ids[j], base_digit=self.text([ids[j]]).strip(),
                    H=str(coin(p + ids[:j], self.a.seed)),
                    active=r.act.tolist(), delta=r.delta.tolist(),
                    q={int(t): int(r._qmap[t]) for t in r.act if int(t) in self.digits}))
            traces.append((ids, c.finish_reason))
        return traces

    def continuations(self, reqs, arm):
        """arm: 'greedy' (n=1, argmax) | 'sampled' (n samples, tau over top-k=40)
        | 'steer' (sampled + self-correction markers banned while an alternative
        remains in the top-k — a within-set-valued-freedom heuristic).
        Returns per request a list of (ids, text, finish) per sample."""
        SP = self.SP
        if arm == 'greedy':
            sp = SP(temperature=0.0, max_tokens=self.a.rollout, detokenize=False,
                    stop_token_ids=list(self.eos))
            sps = [sp] * len(reqs)
        else:
            kw = dict(temperature=self.a.temperature, top_k=40, n=self.a.samples,
                      max_tokens=self.a.rollout, stop_token_ids=list(self.eos),
                      detokenize=False)
            if arm == 'steer':
                # native per-step banning (vLLM V1 has no per-request logits
                # processors); unconditional as in the reference decision_bestofN.py —
                # with ~24 markers against a 40-token top-k the guard would ~never bind
                kw['bad_words'] = self.bad_words
            sps = [SP(seed=stable_seed(self.a.seed, q['key'], arm), **kw) for q in reqs]
        t0 = time.time()
        outs = self.llm.generate([{'prompt_token_ids': q['ids']} for q in reqs], sps, use_tqdm=False)
        self.t[arm if arm != 'greedy' else 'greedy'] += time.time() - t0
        return [[([int(t) for t in c.token_ids], self.text([int(t) for t in c.token_ids]), c.finish_reason)
                 for c in o.outputs] for o in outs]

    def shutdown(self):
        """Release the vLLM engine (staged attn pipeline reloads it later)."""
        import gc
        self.llm = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def audit_trajectory(self, base_ids, cont_ids, mode):
        """Replay a flip trajectory; every non-trailing-EOS token must be admissible."""
        sp = self.SP(temperature=0.0, max_tokens=1, prompt_logprobs=48, detokenize=False)
        t0 = time.time()
        out = self.llm.generate([{'prompt_token_ids': base_ids + cont_ids}], [sp], use_tqdm=False)[0]
        self.t['audit'] += time.time() - t0
        plp = out.prompt_logprobs
        bad = None
        for j, token in enumerate(cont_ids):
            if j == len(cont_ids) - 1 and int(token) in self.eos: break  # termination, not decode
            lp = plp[len(base_ids) + j] if plp is not None else None
            if lp is None: bad = j; break
            try:
                r = relation_from_logprobs({int(t): float(l.logprob) for t, l in lp.items()},
                                           self.V, self.a.temperature)
                if r.witness_for(int(token), coin(base_ids + cont_ids[:j], self.a.seed), mode) is None:
                    bad = j; break
            except ValueError:
                bad = j; break
        return bad


class AttnEngine:
    """HF eager-attention port of attacks/e1b/attention/attn_reach.py (v0 placeholder):
    per-layer output scaling g in [1-rho, 1+rho], union over {coherent+,
    coherent-, adversarial sign} — a documented loose over-approximation of the
    committed-config reachable set. Used only at recorded fork steps."""

    def __init__(self, model_path, rho, device='cuda', dtype='bfloat16'):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype), attn_implementation='eager',
            local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.rho = float(rho); self.device = device; self.dtype = getattr(torch, dtype)
        self.scale = {}
        self.mods = [m for m in self.model.modules() if type(m).__name__.endswith('Attention')]
        self.handles = [m.register_forward_hook(self._mk(m)) for m in self.mods]

    def _mk(self, mod):
        def hook(_m, _in, out):
            s = self.scale.get(id(mod))
            if s is None: return out
            return (out[0] * s,) + tuple(out[1:]) if isinstance(out, tuple) else out * s
        return hook

    def close(self):
        for h in self.handles: h.remove()
        del self.model

    def perturbed_logits(self, ids):
        """[honest, coherent+, coherent-, adversarial] float64 logit vectors at
        the last position of `ids`."""
        torch = self.torch
        m = self.model
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            self.scale.clear(); L0 = m(x).logits[0, -1, :].float()
            for mod in self.mods: self.scale[id(mod)] = 1 + self.rho
            Lp = m(x).logits[0, -1, :].float()
            for mod in self.mods: self.scale[id(mod)] = 1 - self.rho
            Lm = m(x).logits[0, -1, :].float()
        t1, t2 = torch.topk(L0, 2).indices.tolist()
        with torch.enable_grad():
            for mod in self.mods:
                self.scale[id(mod)] = torch.ones((), device=self.device, dtype=self.dtype,
                                                 requires_grad=True)
            lg = m(x).logits[0, -1, :]
            (lg[t1].float() - lg[t2].float()).backward()
        for mod in self.mods:
            g = self.scale[id(mod)]
            self.scale[id(mod)] = (1 - self.rho * torch.sign(g.grad)).detach() \
                if g.grad is not None else torch.tensor(1.0, device=self.device)
        with torch.no_grad():
            Ladv = m(x).logits[0, -1, :].float()
        self.scale.clear()
        return [v.detach().cpu().numpy().astype(np.float64) for v in (L0, Lp, Lm, Ladv)]


def attn_fork_candidates(vec, mode, H, digits, base, temperature, taken):
    """Digit candidates reachable under one perturbed logit vector but not
    already slotted. Returns {token: (witness, q, active, delta)}."""
    try:
        r = Relation(vec, temperature)
    except ValueError:
        return {}
    ws = r.witnesses(int(H), mode)
    out = {}
    for t, w in ws.items():
        if int(t) in digits and int(t) != base and int(t) not in taken:
            out[int(t)] = (w, int(r._qmap[t]), r.act.tolist(), r.delta.tolist())
    return out


def augment_slots_attn(plans, a, modes, digits, attn=None, on_progress=None):
    """In-place attn-union augmentation of each plan's slots: at the recorded
    fork steps, union the decode-reachable set over {coherent+/-, adversarial}
    attention configs (per-layer g placeholder). Honest-decode slots keep
    priority; attn-only candidates fill the remaining slot budget."""
    own = attn is None
    if own: attn = AttnEngine(a.model, rho=7088 / (1 << 20))
    try:
        for pi, pl in enumerate(plans):
            if not pl['ok']: continue
            horizon = pl['horizon']
            for mode in modes:
                slots = pl['modes'][mode]
                if len(slots) >= a.max_slots: continue
                taken = {s['cand'] for s in slots}
                steps = [d for d in pl['state']['digits']
                         if d['step'] < horizon and len(slots) < a.max_slots]
                for d in steps:
                    if len(slots) >= a.max_slots: break
                    prefix = pl['chat'] + pl['ids'][:d['step']]
                    vecs = attn.perturbed_logits(prefix)
                    merged = {}
                    for name, vec in zip(('coherent+', 'coherent-', 'adversarial'), vecs[1:]):
                        for tok, (w, q, act, delta) in attn_fork_candidates(
                                vec, mode, d['H'], digits, d['base'], a.temperature, taken).items():
                            if tok not in merged or q > merged[tok][2]:
                                merged[tok] = (name, w, q, act, delta)
                    for tok, (name, w, q, act, delta) in sorted(merged.items(),
                                                                key=lambda kv: -kv[1][2]):
                        if len(slots) >= a.max_slots: break
                        taken.add(tok)
                        slots.append(dict(step=d['step'], base=int(d['base']), cand=tok,
                                          H=d['H'], active=act, delta=delta, w=w,
                                          src='attn:' + name))
            if on_progress: on_progress(pi)
    finally:
        if own: attn.close()


def plan_item(engine, item, chat, honest_ids, finish, state, a):
    """Honest trace + per-mode attack slots; no engine calls beyond tokenization."""
    h = dict(text=engine.text(honest_ids), length=len(honest_ids), capped=(finish == 'length'),
             injections=[], reachable=[])
    ok = correct(answer(h), item['gold'])
    reason = list(honest_ids); stopped = False
    for j, t in enumerate(reason):
        if t == engine.think or t in engine.eos: reason = reason[:j + 1]; stopped = True; break
    cutoff = next((j for j, t in enumerate(honest_ids) if t == engine.think or t in engine.eos),
                  len(honest_ids))
    horizon = min(a.max_steps, cutoff)
    operand = operand_digits(item['question']) if a.target_restatement else None
    modes, fork_stats = {}, {}
    for mode in a.modes.split(','):
        if ok:
            modes[mode], fork_stats[mode] = build_slots(
                state['digits'], horizon, mode, a.max_slots, a.candidates,
                engine.digits, operand)
        else:
            modes[mode], fork_stats[mode] = [], dict(digit_steps=0, witness_tokens=0,
                                                     digit_candidates=0)
    return dict(item=item, chat=chat, ids=honest_ids, ok=ok, honest=h, state=state,
                horizon=horizon,
                honest_reasoning=dict(length=len(reason), capped=not stopped),
                modes=modes, fork_stats=fork_stats)


def chunk_requests(plans, a):
    """One flat request list for the whole chunk (both arms reuse it);
    index maps (plan_idx, mode) -> [first, last) request range."""
    reqs, index = [], {}
    for pi, pl in enumerate(plans):
        for mode, slots in pl['modes'].items():
            first = len(reqs)
            for k, s in enumerate(slots):
                reqs.append(dict(ids=pl['chat'] + pl['ids'][:s['step']] + [s['cand']],
                                 key=f"{pl['item']['id']}|{mode}|{k}"))
            index[(pi, mode)] = (first, len(reqs))
    return reqs, index


def finish_item(engine, pl, greedy_outs, sample_outs, steer_outs, index, pi, a):
    """Assemble one output row from batched continuation outputs:
    greedy arm fills the legacy decision fields, decision_bestofn carries
    the corrected ladders (sampled and, when enabled, steer)."""
    item = pl['item']
    row = dict(id=item['id'], question=item['question'], gold=item['gold'], honest=pl['honest'],
               honest_reasoning=pl['honest_reasoning'], conditions={})
    for mode in a.modes.split(','):
        c = {}
        slots = pl['modes'].get(mode, [])
        greedy_first = first_details = audit = steer_details = None
        sample_matrix = steer_matrix = []
        lo, hi = index[(pi, mode)]
        if slots and a.greedy_arm:
            outs = greedy_outs[lo:hi]
            g = next((i for i, per in enumerate(outs)
                      if cont_outcome(per[0][1], per[0][2], item['gold'])[0]), None)
            if g is not None:
                greedy_first = dict(slot=g, step=slots[g]['step'], baseline=slots[g]['base'],
                    token=slots[g]['cand'],
                    answer=cont_outcome(outs[g][0][1], outs[g][0][2], item['gold'])[1],
                    witness=dict(active=slots[g]['active'], delta=slots[g]['delta'],
                                 H=slots[g]['H'], **slots[g]['w']))
        if slots:
            outs = sample_outs[lo:hi]
            sample_matrix = [[cont_outcome(text, fin, item['gold'])[0] for _, text, fin in per]
                             for per in outs]
        flat, first, hits = ladder(sample_matrix, a.samples, a.budgets)
        if first is not None:
            slot_i, sample_j = divmod(first, a.samples)
            s = slots[slot_i]
            ids, text, fin = sample_outs[lo + slot_i][sample_j]
            first_details = dict(order=first, slot=slot_i, sample=sample_j, step=s['step'],
                baseline=s['base'], token=s['cand'], answer=cont_outcome(text, fin, item['gold'])[1],
                trace=dict(text=text, length=len(ids), capped=(fin == 'length'), injections=[], reachable=[]),
                witness=dict(active=s['active'], delta=s['delta'], H=s['H'], **s['w']),
                src=s.get('src', 'decode'))
            if a.audit:
                bad = engine.audit_trajectory(pl['chat'] + pl['ids'][:s['step']] + [s['cand']],
                                              ids, mode)
                audit = dict(admissible=bad is None, first_bad_step=bad)
        steer_ladder = None
        if slots and a.steer:
            outs = steer_outs[lo:hi]
            steer_matrix = [[cont_outcome(text, fin, item['gold'])[0] for _, text, fin in per]
                            for per in outs]
            sflat, sfirst, shits = ladder(steer_matrix, a.samples, a.budgets)
            steer_ladder = dict(first_flip_budget=(sfirst + 1) if sfirst is not None else None,
                budgets={str(b): shits[b] for b in a.budgets})
            if sfirst is not None and first is None:      # steer found what sampling missed
                slot_i, sample_j = divmod(sfirst, a.samples)
                ids, text, fin = steer_outs[lo + slot_i][sample_j]
                s = slots[slot_i]
                steer_details = dict(step=s['step'], baseline=s['base'], token=s['cand'],
                    src=s.get('src', 'decode'),
                    answer=cont_outcome(text, fin, item['gold'])[1],
                    trace=dict(text=text, length=len(ids), capped=(fin == 'length'),
                               injections=[], reachable=[]))
        c['decision'] = dict(eligible=pl['ok'], success=bool(greedy_first),
            tried=(greedy_first['slot'] + 1) if greedy_first else len(slots),
            **({k: greedy_first[k] for k in ('step', 'baseline', 'token', 'answer', 'witness')} if greedy_first else {}),
            seconds=0)
        c['decision_bestofn'] = dict(eligible=pl['ok'], slots=len(slots), samples=a.samples,
            continuations=len(flat), fork=pl['fork_stats'].get(mode), first_flip_order=first,
            first_flip_budget=(first + 1) if first is not None else None,
            budgets={str(b): hits[b] for b in a.budgets},
            slots_attn=sum(1 for s in slots if str(s.get('src', '')).startswith('attn')),
            first=first_details, greedy_first_slot=(greedy_first['slot'] if greedy_first else None),
            steer=steer_ladder, steer_first=steer_details, audit=audit)
        row['conditions'][mode] = c
    return row


def bestofn_summary(rows, modes, budgets):
    out = {}
    for mode in modes:
        rs = [r['conditions'][mode]['decision_bestofn'] for r in rows if mode in r['conditions']]
        elig = [r for r in rs if r['eligible']]
        n = max(1, len(elig))
        audits = [r['audit'] for r in elig if r['audit']]
        top = str(max(budgets))
        d = dict(
            eligible=len(elig), slots_total=sum(r['slots'] for r in elig),
            slots_attn_total=sum(r.get('slots_attn', 0) for r in elig),
            continuations_total=sum(r['continuations'] for r in elig),
            asr_greedy=sum(r['greedy_first_slot'] is not None for r in elig) / n,
            asr_by_budget={str(b): sum(r['budgets'][str(b)] for r in elig) / n for b in budgets},
            first_flip_budgets=[r['first_flip_budget'] for r in elig if r['first_flip_budget']],
            audited=len(audits), admissible=sum(a['admissible'] for a in audits),
            asr_strict_at_max=sum(r['budgets'][top] and (r['audit'] is None or r['audit']['admissible'])
                                  for r in elig) / n)
        steers = [r for r in elig if r.get('steer')]
        if steers:
            ns = max(1, len(steers))
            d['steer'] = dict(
                asr_by_budget={str(b): sum(r['steer']['budgets'][str(b)] for r in steers) / ns
                               for b in budgets},
                first_flip_budgets=[r['steer']['first_flip_budget'] for r in steers
                                     if r['steer']['first_flip_budget']])
        flips = [(r.get('first') or {}).get('src', 'decode') for r in elig if r.get('first')]
        if flips:
            d['first_flip_by_src'] = {}
            for src in flips:
                d['first_flip_by_src'][src] = d['first_flip_by_src'].get(src, 0) + 1
        out[mode] = d
    return out


def save_plans(path, plans):
    """Plans carry int-keyed q maps; JSON stringifies keys, so re-int on load."""
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(plans)); tmp.replace(path)


def load_plans(path):
    plans = json.loads(path.read_text())
    for pl in plans:
        for d in pl['state']['digits']:
            d['q'] = {int(k): v for k, v in d['q'].items()}
    return plans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0); p.add_argument('--offset', type=int, default=0)
    p.add_argument('--temperature', type=float, default=1.); p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--modes', default='free-r,paper')
    p.add_argument('--max-new', type=int, default=4096); p.add_argument('--max-steps', type=int, default=400)
    p.add_argument('--candidates', type=int, default=6); p.add_argument('--max-slots', type=int, default=12)
    p.add_argument('--samples', type=int, default=4); p.add_argument('--rollout', type=int, default=1536)
    p.add_argument('--batch', type=int, default=32); p.add_argument('--max-seqs', type=int, default=512)
    p.add_argument('--budgets', default='1,2,4,8,16,24,48')
    p.add_argument('--steer', action='store_true',
                   help='add the steer arm (ban self-correction markers while a top-k alternative remains)')
    p.add_argument('--target-restatement', action='store_true',
                   help='search digit forks whose honest digit appears in the prompt operands first')
    p.add_argument('--attn-forks', action='store_true',
                   help='augment fork candidate sets with the attention-tolerance union (HF eager pass)')
    p.add_argument('--no-greedy-arm', dest='greedy_arm', action='store_false')
    p.add_argument('--no-audit', dest='audit', action='store_false')
    p.add_argument('--stage', default='all', choices=['all', 'honest', 'attn', 'arms'],
                   help='all runs the attn pass in a subprocess; explicit stages allow manual control')
    a = p.parse_args(); a.budgets = sorted({int(x) for x in a.budgets.split(',')})
    a.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
              if k != 'stage'}  # stage is orchestration, not experiment semantics
    config['backend'] = 'vllm-bestofn'; config['data_sha256'] = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    cfg = a.out / 'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config: raise ValueError('Refusing mismatched resume')
    cfg.write_text(json.dumps(config, indent=2))
    items = [json.loads(x) for x in Path(a.data).read_text().splitlines()][a.offset:]
    if a.limit: items = items[:a.limit]
    done = {f.stem for f in (a.out / 'items').glob('*.json')} if (a.out / 'items').exists() else set()
    (a.out / 'items').mkdir(exist_ok=True)
    if len(done) == len(items):
        (a.out / 'COMPLETE').write_text(str(len(items)) + '\n'); print('Already complete'); return

    def write_rows(engine, plans):
        t0 = time.time()
        creqs, index = chunk_requests(plans, a)
        greedy_outs = engine.continuations(creqs, 'greedy') \
            if (a.greedy_arm and creqs) else [None] * len(creqs)
        sample_outs = engine.continuations(creqs, 'sampled') if creqs else []
        steer_outs = engine.continuations(creqs, 'steer') if (a.steer and creqs) else []
        for pi, pl in enumerate(plans):
            row = finish_item(engine, pl, greedy_outs, sample_outs, steer_outs, index, pi, a)
            key = hashlib.sha256(pl['item']['id'].encode()).hexdigest()[:16]
            target = a.out / 'items' / f'{key}.json'; tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(row)); tmp.replace(target)
        all_rows = [json.loads(f.read_text()) for f in (a.out / 'items').glob('*.json')]
        summary = summarize(all_rows)
        summary['bestofn'] = bestofn_summary(all_rows, a.modes.split(','), a.budgets)
        summary['phase_seconds'] = {k: round(v, 1) for k, v in engine.t.items()}
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(dict(items=len(all_rows), total=len(items),
              seconds=round(time.time() - t0, 1), phase_seconds=summary['phase_seconds'],
              bestofn={m: {k: v for k, v in s.items() if k != 'first_flip_budgets'}
                       for m, s in summary['bestofn'].items()})), flush=True)

    plans_path = a.out / 'plans.json'
    if a.stage == 'attn':   # standalone HF attention pass; no vLLM engine needed
        plans = load_plans(plans_path)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        digits = {i for i in range(len(tok))
                  if len(tok.decode([i]).strip()) == 1 and tok.decode([i]).strip().isdigit()}
        t0 = time.time()
        augment_slots_attn(plans, a, a.modes.split(','), digits,
                           on_progress=lambda i: print(json.dumps(dict(attn_pass=i + 1,
                                                                  total=len(plans))), flush=True))
        print(json.dumps(dict(attn_augment_seconds=round(time.time() - t0, 1))), flush=True)
        save_plans(plans_path, plans)
        return
    engine = VEngine(a)
    todo = [it for it in items if hashlib.sha256(it['id'].encode()).hexdigest()[:16] not in done]
    if a.stage in ('all', 'honest'):
        plans = []
        for start in range(0, len(todo), a.batch):
            chunk = todo[start:start + a.batch]
            reqs = [(chat_ids(engine.tok, it['question']), dict(digits=[]), it) for it in chunk]
            traces = engine.honest([r[0] for r in reqs], [r[1] for r in reqs],
                                   [r[2]['id'] for r in reqs])
            plans.extend(plan_item(engine, it, chat, ids, fin, st, a)
                         for (chat, st, it), (ids, fin) in zip(reqs, traces))
            print(json.dumps(dict(honest_pass=start + len(chunk), total=len(todo))), flush=True)
        save_plans(plans_path, plans)
    if a.stage == 'honest':
        return
    if a.attn_forks:  # isolate the HF attention pass in a subprocess (GPU release)
        import subprocess, sys
        engine.shutdown()
        subprocess.run([sys.executable, '-m', 'experiments.vllm_bestofn'] + sys.argv[1:]
                       + ['--stage', 'attn'], check=True)
        engine = VEngine(a)
    plans = load_plans(plans_path)
    for start in range(0, len(plans), a.batch):
        write_rows(engine, plans[start:start + a.batch])
    (a.out / 'COMPLETE').write_text(str(len(items)) + '\n')


if __name__ == '__main__':
    main()
