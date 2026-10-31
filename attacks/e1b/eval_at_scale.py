#!/usr/bin/env python3
"""At-scale evaluation of zkAgent decode-steering attacks.

Runs on any HF causal LM and a real dataset (GSM8K by default), measuring, under
BOTH decode models (free-r = released implementation, paper = H-bound+grind-Theta):

  decision:  honest accuracy, attacked accuracy, and attack success rate
             (fraction of originally-correct items the attacker flips to wrong,
             via a lookahead search for a reachable single-digit corruption that
             survives the model's self-correction).
  billing:   honest vs attacked billed length, injection rate, inflation factor
             (P2-lexical: inject reachable content-opening tokens to prolong CoT).

This is the scale-up of the per-prompt proofs in attacks/e1b/. On a 15 GB CPU box
use a 1.5 B model and --limit a few; on a GPU box use 8 B+ and the full split.

Faithfulness: quant / shift / mass / free-r reachability are the same primitives
validated bit-for-bit against the C++ oracle (see reach_both.cpp, freer_reach.cpp).
"""
import argparse, csv, re, sys, time
import numpy as np, torch
sys.path.insert(0, "attacks/e1b")
from e1b_length_steer import quant, active_and_shift, reach_freeR, reach_paper, prefix_coin
from transformers import AutoModelForCausalLM, AutoTokenizer

OPENERS = {"Wait","But","However","Although","Though","Alternatively","Actually","Hmm",
           "Also","Moreover","Furthermore","Additionally","Besides","Yet","Still","Instead",
           "Meanwhile","Note","Another","Perhaps","Maybe","Unless","Except","Despite",
           "Whereas","Nevertheless","Nonetheless","Regardless","Anyway","Plus","Interestingly",
           "Notably","Importantly","Similarly","Conversely","Otherwise","Suppose","Consider",
           "Recall","Remember","Looking","Because","Since","Firstly","Secondly","Here","Now","Let"}

def last_int(text):
    m = re.findall(r"-?\d+", text.replace(",", ""))
    return m[-1] if m else None

def load_items(dataset, limit, split):
    if dataset == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"{split}[:{limit}]")
        return [(r["question"], r["answer"].split("####")[-1].strip()) for r in ds]
    if dataset == "math":
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split=f"{split}[:{limit}]")
        def boxed(s):
            m = re.search(r"\\boxed\{([^}]*)\}", s); return m.group(1) if m else None
        return [(r["problem"], boxed(r["solution"])) for r in ds]
    # custom tiny arithmetic set (no download) -- for plumbing validation
    base = [("What is 7 times 8?", "56"), ("What is 9 times 6?", "54"),
            ("What is 13 plus 29?", "42"), ("What is 100 minus 37?", "63"),
            ("What is 12 times 12?", "144"), ("What is 15 times 4?", "60")]
    return base[:limit]

class Harness:
    def __init__(self, a):
        self.a = a
        self.dev = a.device
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(self.dev).eval()
        self.terms = {self.tok.eos_token_id}
        te = self.tok.convert_tokens_to_ids("</think>")
        if isinstance(te, int) and te >= 0: self.terms.add(te)
        self.is_digit = lambda i: (lambda s: len(s)==1 and s.isdigit())(self.tok.decode([int(i)]).strip())
        self.is_opener = lambda i: self.tok.decode([int(i)]).strip() in OPENERS
        self.probe_w = np.load(a.probe)["w"] if getattr(a, "probe", "") else None  # length probe (P2)

    @torch.no_grad()
    def probe_score(self, ids, cand):
        """1-step lookahead predicted remaining length if we emit `cand` (P2 value probe)."""
        probe_ids = torch.cat([ids, torch.tensor([[cand]], device=self.dev)], dim=1)
        h = self.model(probe_ids, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu().numpy()
        return float(self.probe_w[:-1] @ h + self.probe_w[-1])

    def prompt_ids(self, q):
        msg = [{"role":"user","content": q + " Reason step by step, then give the final answer."}]
        return self.tok.apply_chat_template(msg, add_generation_prompt=True,
                 return_tensors="pt", return_dict=True)["input_ids"].to(self.dev)

    @torch.no_grad()
    def logits(self, ids):
        return self.model(ids).logits[0, -1, :].float()

    def reach_mask(self, mode, lg_np, ids):
        # the decode relation operates at temperature tau: logits are scaled by 1/tau
        qint = quant(lg_np / self.a.temperature); act, delta, shift = active_and_shift(qint, self.a.k)
        mask = reach_freeR(delta, shift) if mode == "free-r" \
               else reach_paper(delta, shift, prefix_coin(ids[0].tolist()), ngrid=self.a.ngrid)
        return act, mask

    @torch.no_grad()
    def generate(self, ids, cap, policy=None):
        """policy(step, lg_np, ids, greedy) -> token to emit (or None for greedy)."""
        blen = ids.shape[1]; n = 0; meta = {"inj": 0}
        while n < cap:
            lg = self.logits(ids); lg_np = lg.cpu().numpy(); greedy = int(lg.argmax())
            choose = greedy
            if policy is not None and greedy not in self.terms:
                alt = policy(n, lg_np, ids, greedy, meta)
                if alt is not None: choose = alt
            ids = torch.cat([ids, torch.tensor([[choose]], device=self.dev)], dim=1); n += 1
            if choose in self.terms and n > 3: break
        return self.tok.decode(ids[0, blen:]), meta

    # ---- billing: length steering (P2-lexical openers, or P2-probe value scoring) ----
    def billing(self, q, mode, cap):
        ids = self.prompt_ids(q)
        htext, _ = self.generate(ids.clone(), cap)
        Lh = len(self.tok(htext).input_ids)
        use_probe = self.probe_w is not None and self.a.bill_policy == "probe"
        def pol(n, lg_np, ids, greedy, meta):
            act, mask = self.reach_mask(mode, lg_np, ids)
            if use_probe:
                # among reachable non-terminator candidates (top-N by model prob),
                # pick the one the probe predicts gives the LONGEST continuation.
                order = np.argsort(-lg_np[act])
                cands = [int(act[j]) for j in order
                         if mask[j] and int(act[j]) not in self.terms][:self.a.probe_topn]
                if not cands: return None
                best = max(cands, key=lambda c: self.probe_score(ids, c))
                if best != greedy and self.probe_score(ids, best) > self.probe_score(ids, greedy):
                    meta["inj"] += 1; return best
                return None
            cands = [int(act[j]) for j in range(len(act))
                     if mask[j] and self.is_opener(int(act[j])) and int(act[j]) != greedy]
            if cands: meta["inj"] += 1; return cands[0]
            return None
        atext, meta = self.generate(ids.clone(), cap, pol)
        La = len(self.tok(atext).input_ids)
        return dict(honest_len=Lh, atk_len=La, inj=meta["inj"], infl=La/max(Lh,1), capped=La>=cap)

    # ---- decision: lookahead flip search ----
    @torch.no_grad()
    def rollout(self, ids, n):
        blen = ids.shape[1]; k = 0
        while k < n:
            t = int(self.logits(ids).argmax())
            ids = torch.cat([ids, torch.tensor([[t]], device=self.dev)], dim=1); k += 1
            if t == self.tok.eos_token_id and k > 3: break
        return self.tok.decode(ids[0, blen:])

    @torch.no_grad()
    def decision(self, q, correct, mode):
        ids = self.prompt_ids(q)
        htext = self.rollout(ids.clone(), self.a.max_new)
        hans = last_int(htext); honest_ok = (hans == correct)
        flipped = False; info = {}
        if honest_ok:  # only attack items the model gets right
            cur = ids.clone(); tried = 0
            for step in range(self.a.max_steps):
                if tried >= self.a.max_rollouts: break
                lg = self.logits(cur); lg_np = lg.cpu().numpy(); greedy = int(lg.argmax())
                if greedy in self.terms: break
                if self.is_digit(greedy):
                    act, mask = self.reach_mask(mode, lg_np, cur)
                    cands = [int(act[j]) for j in range(len(act))
                             if mask[j] and self.is_digit(int(act[j])) and int(act[j]) != greedy][:self.a.cand]
                    for c in cands:
                        if tried >= self.a.max_rollouts: break
                        tried += 1
                        probe = torch.cat([cur, torch.tensor([[c]], device=self.dev)], dim=1)
                        ans = last_int(self.rollout(probe, self.a.rollout))
                        if ans is not None and ans != correct:
                            flipped = True
                            info = dict(step=step, inj=f"{self.tok.decode([greedy]).strip()}->{self.tok.decode([c]).strip()}", wrong=ans, tried=tried)
                            break
                    if flipped: break
                cur = torch.cat([cur, torch.tensor([[greedy]], device=self.dev)], dim=1)
            info.setdefault("tried", tried)
        return dict(honest_ok=honest_ok, honest_ans=hans, flipped=flipped, **info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k","math","custom"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32","bfloat16","float16"], help="bfloat16 for 8B on a single GPU")
    ap.add_argument("--models", default="free-r,paper")
    ap.add_argument("--task", default="both", choices=["billing","decision","both"])
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=1.0, help="tau of the sampling decode the attacker exploits")
    ap.add_argument("--ngrid", type=int, default=1500)
    ap.add_argument("--max-new", type=int, default=512)   # honest/rollout length
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--max-rollouts", type=int, default=24)
    ap.add_argument("--cand", type=int, default=6)
    ap.add_argument("--bill-cap", type=int, default=512)
    ap.add_argument("--bill-policy", default="lexical", choices=["lexical","probe"],
                    help="probe = P2 value probe (needs --probe); lexical = content-opener injection")
    ap.add_argument("--probe", default="", help="path to length_probe.npz from train_length_probe.py")
    ap.add_argument("--probe-topn", type=int, default=6, help="reachable candidates scored per step (P2)")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    modes = args.models.split(",")
    H = Harness(args)
    items = load_items(args.dataset, args.limit, args.split)
    print(f"[eval] model={args.model} dataset={args.dataset} n={len(items)} modes={modes} task={args.task}", flush=True)

    rows = []
    agg = {m: dict(correct=0, flipped=0, attackable=0, infl=[], injrate=[]) for m in modes}
    for idx, (q, gold) in enumerate(items):
        t0 = time.time(); row = {"idx": idx, "gold": gold}
        for m in modes:
            if args.task in ("decision","both"):
                d = H.decision(q, gold, m)
                row[f"{m}_honest_ok"] = int(d["honest_ok"]); row[f"{m}_flip"] = int(d["flipped"])
                agg[m]["correct"] += int(d["honest_ok"])
                if d["honest_ok"]:
                    agg[m]["attackable"] += 1; agg[m]["flipped"] += int(d["flipped"])
            if args.task in ("billing","both"):
                b = H.billing(q, m, args.bill_cap)
                row[f"{m}_infl"] = round(b["infl"],2); row[f"{m}_inj"] = b["inj"]
                agg[m]["infl"].append(b["infl"]); agg[m]["injrate"].append(b["inj"]/max(b["atk_len"],1))
        rows.append(row)
        print(f"  [{idx+1}/{len(items)}] {time.time()-t0:.0f}s " +
              " | ".join(f"{m}:flip={row.get(m+'_flip','-')},infl={row.get(m+'_infl','-')}" for m in modes), flush=True)

    print("\n===== SUMMARY =====", flush=True)
    for m in modes:
        a = agg[m]
        line = f"[{m}]"
        if args.task in ("decision","both"):
            acc = a["correct"]/len(items)
            asr = a["flipped"]/max(a["attackable"],1)
            atk_acc = (a["correct"]-a["flipped"])/len(items)
            line += f" honest_acc={acc:.1%} attacked_acc={atk_acc:.1%} flip_success={a['flipped']}/{a['attackable']}={asr:.1%}"
        if args.task in ("billing","both"):
            import statistics as st
            line += f" | median_inflation={st.median(a['infl']):.2f} mean_injrate={st.mean(a['injrate']):.1%}"
        print(line, flush=True)

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"[wrote] {args.csv}", flush=True)

if __name__ == "__main__":
    main()
