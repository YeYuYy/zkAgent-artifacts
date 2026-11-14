"""Tests for choice parsing and no-gold summary metrics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.choice_scoring import (choice_flip, choice_summary, parse_choice,
                                        _auc, _pooled_auc)


class TestParse(unittest.TestCase):
    def test_boxed_after_think(self):
        t = dict(text='reasoning The better option is B. \\boxed{B}', capped=False)
        self.assertEqual(parse_choice(t, ['A', 'B']), 'B')

    def test_normalizes_case_and_space(self):
        self.assertEqual(parse_choice(dict(text='\\boxed{ a }', capped=False), ['A', 'B']), 'A')
        self.assertEqual(parse_choice(dict(text='\\boxed{ 3 }', capped=False),
                                      ['1', '2', '3', '4']), '3')

    def test_none_cases(self):
        self.assertIsNone(parse_choice(dict(text='\\boxed{C}', capped=False), ['A', 'B']))
        self.assertIsNone(parse_choice(dict(text='\\boxed{A}', capped=True), ['A', 'B']))
        self.assertIsNone(parse_choice(dict(text='no answer', capped=False), ['A', 'B']))

    def test_flip(self):
        self.assertTrue(choice_flip('B', 'A'))
        self.assertFalse(choice_flip('A', 'A'))
        self.assertFalse(choice_flip(None, 'A'))
        self.assertFalse(choice_flip('B', None))


def row(i, h, c, canon, flip, nodes=10, complete=True):
    return dict(
        id=str(i), domain='dailydilemma',
        honest=dict(choice=h, text='', capped=False),
        control=dict(choice=c, text='', capped=False),
        canonical=dict(choice=canon, text='', capped=False),
        conditions={'paper': dict(choice_tree=dict(
            eligible=h is not None, success=flip, complete=complete, truncated=not complete,
            node_limited=False, nodes=nodes, terminal_paths=3, flip_paths=int(flip),
            max_depth=20, first_flip_depth=7 if flip else None,
            canonical_choice=canon, search='exhaustive', paths_capped=0, paths_pruned=0,
            seconds=1.0, scores=[]))})


class TestSummary(unittest.TestCase):
    def test_rates_and_agreement(self):
        rows = [row(1, 'A', 'A', 'A', True), row(2, 'B', 'A', 'B', False),
                row(3, 'A', None, 'A', False)]
        s = choice_summary(rows)
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['honest_parse_rate'], 1)
        self.assertEqual(s['null_flip_rate'], 1 / 3)
        self.assertEqual(s['canonical_agreement'], 3)
        cond = s['conditions']['paper']
        self.assertEqual(cond['eligible'], 3)
        self.assertEqual(cond['flips'], 1)
        self.assertEqual(cond['asr_eligible'], 1 / 3)
        self.assertEqual(cond['canonical_choice_dist'], {'A': 2, 'B': 1})
        self.assertEqual(cond['complete'], 3)

    def test_auc(self):
        self.assertEqual(_auc([2., 3.], [0., 1.]), 1.0)
        self.assertEqual(_auc([], [1.]), None)
        self.assertAlmostEqual(_auc([1.], [1.]), 0.5)
        self.assertAlmostEqual(_pooled_auc([dict(scores=[[2., True], [0., False]])]), 1.0)


if __name__ == '__main__':
    unittest.main()