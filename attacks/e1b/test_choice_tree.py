"""Logic tests for the choice-tree runner (no GPU): path bookkeeping, flip
accounting, forced-answer scoring and candidate-level AUC labels."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.choice_tree import (ChoiceEngine, FORCED_CUE, candidate_prompts,
                                     item_result, new_path, spawn_path, stable_seed)


class LP:
    def __init__(self, v):
        self.logprob = v


class FakeTok:
    SURF = {0: 'A', 1: ' A', 2: 'a', 3: ' a', 4: 'B', 5: ' B'}

    def encode(self, s, add_special_tokens=False):
        for i, v in self.SURF.items():
            if v == s:
                return [i]
        return [99]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.SURF.get(int(i), '?') for i in ids)


class TestHelpers(unittest.TestCase):
    def test_stable_seed(self):
        self.assertEqual(stable_seed(1, 'x', 'honest'), stable_seed(1, 'x', 'honest'))
        self.assertNotEqual(stable_seed(1, 'x'), stable_seed(1, 'y'))

    def test_label_ids(self):
        e = ChoiceEngine.__new__(ChoiceEngine)
        e.tok = FakeTok()
        ids = e.label_ids(['A', 'B'])
        self.assertEqual(ids['A'], [0, 1, 2, 3])
        self.assertEqual(ids['B'], [4, 5])

    def test_forced_scores(self):
        e = ChoiceEngine.__new__(ChoiceEngine)
        e.label_lps = lambda prompts, allowed: [
            {0: -0.1, 4: -2.3},   # A dominates
            {0: -2.3, 4: -0.1},   # B dominates
        ]
        out = e.forced_scores(['p1', 'p2'], ['A', 'B'], ['A', 'A'],
                              {'A': [0, 1], 'B': [4, 5]})
        self.assertLess(out[0]['margin'], 0)   # honest A is likely -> no flip
        self.assertGreater(out[1]['margin'], 0)  # non-honest B dominates -> flip
        self.assertAlmostEqual(out[0]['probs']['A'], 1 / (1 + 0.1108), places=3)

    def test_candidate_prompts_include_token(self):
        # (parent, token, prefix, is_canonical): the cue must follow the token,
        # otherwise every candidate of a parent shares a prompt and delta == 0.
        cand = [(None, 7, [1, 2], True), (None, 9, [1, 2], False)]
        ps = candidate_prompts(cand, [0, 1], [100, 101])
        self.assertEqual(ps, [[1, 2, 7, 100, 101], [1, 2, 9, 100, 101]])

    def test_cue(self):
        self.assertIn('\\boxed{', FORCED_CUE)


class TestAudit(unittest.TestCase):
    def test_witness_audit(self):
        import numpy as np
        from experiments.relation import Relation, coin
        from experiments.choice_tree_audit import audit_trace
        r = Relation(np.random.default_rng(0).normal(0, 20, size=500), 1.0)
        prefix = [901, 902]
        seed = next(s for s in range(5000) if len(r.witnesses(coin(prefix, s), 'paper')) > 1)
        H = coin(prefix, seed)
        ws = r.witnesses(H, 'paper')
        canon = int(r.canonical(H))
        fork = dict(step=0, canon=canon, extras=[int(t) for t in ws if t != canon],
                    act=r.act.tolist(), delta=r.delta.tolist(),
                    witnesses={str(int(t)): w for t, w in ws.items() if t != canon})
        good = dict(ids=prefix + [canon, 3], base_len=2, fork_steps=[fork], deviations=[])
        self.assertEqual(audit_trace(good, seed)['failures'], [])
        t = int(next(t for t in ws if t != canon))
        dev = dict(ids=prefix + [t, 3], base_len=2, fork_steps=[fork],
                   deviations=[dict(step=0, token=t, witness=ws[t], act=r.act.tolist(),
                                    delta=r.delta.tolist(), config='honest')])
        self.assertEqual(audit_trace(dev, seed)['failures'], [])
        bad = dict(dev, ids=prefix + [t, 3], deviations=[dict(dev['deviations'][0], token=canon)])
        self.assertTrue(audit_trace(bad, seed)['failures'])


class TestItemResult(unittest.TestCase):
    def _paths(self):
        spine = new_path(0, [1, 2], 2, spine=True)
        spine.update(done='stop', choice='A', nodes=5,
                     cand_scores=[dict(step=0, token=1, margin=0.1, canonical=True)])
        flip = new_path(0, [1, 9, 3], 2, depth=1)
        flip.update(done='stop', choice='B', nodes=3, root_score=1.5,
                    cand_scores=[dict(step=1, token=3, margin=0.4, canonical=True)])
        other = new_path(0, [1, 4], 2)
        other.update(done='stop', choice='A', nodes=2)
        return [spine, flip, other], spine

    def test_flip_and_canonical(self):
        paths, spine = self._paths()
        limits = dict(truncated=False, node_limited=False, spawned=2)
        r = item_result(paths, 'A', limits, 'exhaustive', spine, beam_width=0)
        self.assertTrue(r['eligible'])
        self.assertTrue(r['success'])
        self.assertEqual(r['terminal_paths'], 3)
        self.assertEqual(r['flip_paths'], 1)
        self.assertEqual(r['canonical_choice'], 'A')
        self.assertEqual(r['distinct_choices'], ['A', 'B'])
        self.assertEqual(r['first_flip_depth'], 1)
        self.assertTrue(r['complete'])
        self.assertEqual(sorted(r['forced']['scores'], key=str),
                         sorted([[0.1, False], [0.4, True], [1.5, True]], key=str))

    def test_spawn_carries_deviation_token(self):
        # prefix = parent sequence before its own token; the child must be
        # prefix + [t] with depth one past the parent's step.
        q = spawn_path(0, [7, 8, 9], 42, base_len=2, depth=2)
        self.assertEqual(q['ids'], [7, 8, 9, 42])
        self.assertEqual(q['depth'], len(q['ids']) - q['base_len'])
        self.assertRaises(AssertionError, new_path, 0, [1], 2, 1)

    def test_incomplete_when_pruned(self):
        paths, spine = self._paths()
        paths.append(dict(new_path(0, [1, 2], 2), done='pruned', choice=None, nodes=1))
        limits = dict(truncated=False, node_limited=False, spawned=2)
        r = item_result(paths, 'A', limits, 'exhaustive', spine)
        self.assertFalse(r['complete'])
        self.assertEqual(r['paths_pruned'], 1)

    def test_no_honest_is_ineligible(self):
        paths, spine = self._paths()
        r = item_result(paths, None, dict(truncated=False, node_limited=False, spawned=0),
                        'exhaustive', spine)
        self.assertFalse(r['eligible'])
        self.assertFalse(r['success'])
        self.assertEqual(r['scores'], [])

    def test_beam_block(self):
        paths, spine = self._paths()
        r = item_result(paths, 'A', dict(truncated=False, node_limited=False, spawned=0),
                        'beam', spine, beam_width=4)
        self.assertEqual(r['beam']['width'], 4)
        self.assertEqual(r['beam']['tokens'], r['nodes'])
        self.assertIn('beam', r)

    def test_final_margins(self):
        paths, spine = self._paths()
        spine['final_margin'] = -1.0
        paths[1]['final_margin'] = 2.0
        paths[1]['beam_score'] = 1.5
        r = item_result(paths, 'A', dict(truncated=False, node_limited=False, spawned=0),
                        'beam', spine, beam_width=4)
        self.assertEqual(r['final'], [[0.0, -1.0, False], [1.5, 2.0, True]])

    def test_rollout_meta(self):
        paths, spine = self._paths()
        paths[1]['rollout'] = 3
        paths[1]['ndev'] = 2
        r = item_result(paths, 'A', dict(truncated=False, node_limited=False, spawned=0),
                        'rollout', spine)
        # rows are [rollout idx, any-flip, directed-hit, depth, ndev]; no
        # directed target here, so the directed column is None.
        self.assertEqual(r['rollout_meta'], [[-1, False, None, 0, 0],
                                             [3, True, None, 1, 2],
                                             [-1, False, None, 0, 0]])

    def test_directed_result(self):
        paths, spine = self._paths()
        paths[1]['choice'] = 'B'
        r = item_result(paths, 'A', dict(truncated=False, node_limited=False, spawned=0),
                        'beam', spine, beam_width=4, directed_target='B')
        self.assertTrue(r['directed_eligible'])
        self.assertTrue(r['directed_success'])
        self.assertEqual(r['directed_hits'], 1)
        # honest choice IS the target -> directed flip is vacuous
        r2 = item_result(paths, 'B', dict(truncated=False, node_limited=False, spawned=0),
                         'beam', spine, beam_width=4, directed_target='B')
        self.assertFalse(r2['directed_eligible'])
        self.assertFalse(r2['directed_success'])


if __name__ == '__main__':
    unittest.main()