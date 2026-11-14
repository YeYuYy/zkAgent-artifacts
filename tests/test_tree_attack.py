"""Stub-based tests for the admissible tree search (no model, no torch).

Vocabulary (id -> text): 1..9 digits '1'..'9', 20 '=', 30 '\\boxed{', 31 '4',
32 '2', 33 '}', 34 '9', 40 '</think>', 50 EOS.  A scripted step_logits_fn maps
each prefix (tuple) to a full-dim logit vector; reach_fn decides reachability
per step, so tests control forks exactly.
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'attacks' / 'e1b' / 'decision'))
import numpy as np
from attacks.e1b.decision.tree_attack import (tree_attack, default_cfg, argmax_in_reach,
                                              Stepper, operand_digits)

V = 60
DECODE = {i: str(i) for i in range(1, 10)}
DECODE.update({20: '=', 30: '\\boxed{', 31: '4', 32: '2', 33: '}', 34: '9',
               40: '</think>', 50: '<eos>'})


def cfg(**over):
    return default_cfg(
        max_steps=40, max_groups=3, max_forks=6, term_ids={50},
        is_digit=lambda i: (lambda s: len(s) == 1 and s.isdigit())(DECODE.get(i, '?')),
        decode=lambda i: DECODE.get(i, '?'), **over)


def vec(top):
    """Full-dim logits; `top` = ordered [(id, logit)] for the head of the row
    (first occurrence wins, so scripted tokens keep their slot)."""
    v = np.full(V, -20.0, dtype=np.float64)
    for t, lg in top:
        if v[t] == -20.0:
            v[t] = lg
    return v


def boxed_answer(text):
    import re
    m = re.findall(r'\\boxed\{([^}]*)\}', text)
    return m[-1] if m else None


class ScriptedWorld:
    """logits per prefix (scripted top token + fixed runners-up); reach per
    prefix (scripted overrides; default: top-k of the row)."""

    def __init__(self, paths=None, reach_over=None, default_reach_size=3):
        self.paths = paths or {}          # tuple(prefix) -> token to place on top
        self.reach_over = reach_over or {}
        self.default_reach_size = default_reach_size
        self.reach_seen = []              # (prefix, reach) for invariant checks

    def logits(self, prefix):
        t = self.paths.get(tuple(prefix))
        if t is not None:
            return vec([(t, -0.1), (34, -0.5), (31, -0.6)])
        return vec([(31, -0.1), (32, -0.2), (33, -0.3)])

    def reach(self, logits, prefix):
        order = list(np.argsort(-logits))[:self.default_reach_size]
        r = self.reach_over.get(tuple(prefix), set(int(t) for t in order))
        self.reach_seen.append((tuple(prefix), set(r)))
        return r


# canonical trajectory: <boxed> 4 2 } <eos>  (= 42); fork at the '2' step where
# '9' is also reachable; the 9-branch concludes \boxed{49}.
CANON = {(99,): 30, (99, 30): 31, (99, 30, 31): 32, (99, 30, 31, 32): 33,
         (99, 30, 31, 32, 33): 50,
         (99, 30, 31, 34): 33, (99, 30, 31, 34, 33): 50}
FORK = {(99, 30, 31): {32, 34}}


def run_attack(world, correct='42', prompt='What is 6 times 7?', **over):
    c = cfg(**over)
    out = tree_attack(world.logits, world.reach, argmax_in_reach, [99], correct,
                      prompt, 400, c, boxed_answer)
    return out, c


class TestTreeAttack(unittest.TestCase):
    def test_invariant_every_emitted_token_in_reach(self):
        world = ScriptedWorld(paths=CANON, reach_over=FORK)
        out, c = run_attack(world)
        st = Stepper(world.logits, world.reach, argmax_in_reach, c, [400])
        ids, states, meta = st.run([99])
        for reach, emit, _dig in states:
            self.assertIn(emit, reach)                 # the core invariant
        self.assertEqual([DECODE[t] for t in ids[1:]],
                         ['\\boxed{', '4', '2', '}', '<eos>'])

    def test_constructed_flip_is_found(self):
        world = ScriptedWorld(paths=CANON, reach_over=FORK)
        out, c = run_attack(world)
        self.assertTrue(out['ok'], out)
        self.assertEqual(out['dprime'], '9')
        self.assertEqual(out['answer'], '49')
        self.assertEqual(out['group'], '2')
        self.assertEqual(out['plan'], 'A')             # single mention: A hits first
        # every prefix the search visited kept emissions inside its reach
        for prefix, reach in world.reach_seen:
            self.assertTrue(reach)

    def test_paper_degenerate_no_forks(self):
        world = ScriptedWorld(paths=CANON, default_reach_size=1)
        out, c = run_attack(world)
        self.assertFalse(out['ok'])
        self.assertEqual(out['reason'], 'exhausted')   # canonical ok, zero forks
        self.assertEqual(out['tried'], [])
        for prefix, reach in world.reach_seen:
            self.assertEqual(len(reach), 1)            # singleton reach, never left

    def test_canonical_mismatch_skips(self):
        world = ScriptedWorld(paths={(99,): 30, (99, 30): 50})  # never concludes
        out, c = run_attack(world)
        self.assertFalse(out['ok'])
        self.assertTrue(out['skipped'])
        self.assertEqual(out['reason'], 'canonical-answer')

    def test_grouping_and_operands(self):
        self.assertEqual(operand_digits('6 times 7 and 42 apples'),
                         {'6', '7', '4', '2'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
