"""Torch-free tests for the no-gold choice data builders."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.choice_data import (load_dailydilemma, load_webshop,
                                     webshop_question, write_jsonl, read_jsonl)


class TestDilemma(unittest.TestCase):
    def _csv(self, path):
        rows = [
            dict(dilemma_idx='1', action_type='to_do', dilemma_situation='You are tired.',
                 action='Go to sleep.', values_aggregated='health',
                 topic_group='Health', topic='sleep'),
            dict(dilemma_idx='1', action_type='not_to_do', dilemma_situation='You are tired.',
                 action='Keep working.', values_aggregated='achievement',
                 topic_group='Health', topic='sleep'),
            dict(dilemma_idx='2', action_type='to_do', dilemma_situation='A friend calls.',
                 action='Answer the phone.', values_aggregated='friendship',
                 topic_group='Relationships', topic='friends'),
            dict(dilemma_idx='2', action_type='not_to_do', dilemma_situation='A friend calls.',
                 action='Ignore it.', values_aggregated='focus',
                 topic_group='Relationships', topic='friends'),
        ]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    def test_pairing_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'dd.csv'; self._csv(p)
            a = load_dailydilemma(p, seed=7)
            b = load_dailydilemma(p, seed=7)
        self.assertEqual(len(a), 2)
        self.assertEqual([x['id'] for x in a], [x['id'] for x in b])
        for it in a:
            self.assertEqual(it['labels'], ['A', 'B'])
            self.assertIsNone(it['gold_choice'])
            self.assertEqual(len(it['options']), 2)
            self.assertIn('\\boxed{A}', it['question'])
        texts = {o['text'] for it in a for o in it['options']}
        self.assertIn('Go to sleep.', texts)
        self.assertIn('Ignore it.', texts)


class TestWebshop(unittest.TestCase):
    def test_static_k_way_choice(self):
        items = [
            dict(asin='A1', name='Red Coffee Mug', brand='X', category='Mugs',
                 small_description=['ceramic coffee mug red'], full_description='A red mug.',
                 pricing='$10'),
            dict(asin='A2', name='Blue Coffee Mug', brand='Y', category='Mugs',
                 small_description=['ceramic coffee mug blue'], full_description='A blue mug.',
                 pricing='$8'),
            dict(asin='A3', name='Green Tea Cup', brand='Z', category='Mugs',
                 small_description=['ceramic cup green'], full_description='A green cup.',
                 pricing='$9'),
            dict(asin='B1', name='Lamp', brand='L', category='Lighting',
                 small_description=['desk lamp'], full_description='A lamp.', pricing='$20'),
        ]
        ins = {
            'A1': dict(instruction='red ceramic coffee mug', attributes=['red', 'ceramic']),
            'A2': dict(instruction='blue ceramic coffee mug', attributes=['blue']),
            'A3': dict(instruction='green ceramic cup', attributes=['green']),
        }
        with tempfile.TemporaryDirectory() as d:
            ip = Path(d) / 'items.json'; sp = Path(d) / 'ins.json'
            ip.write_text(json.dumps(items)); sp.write_text(json.dumps(ins))
            out = load_webshop(ip, sp, candidates=3, seed=3)
            self.assertEqual(len(out), 3)
            for it in out:
                self.assertEqual(len(it['options']), 3)
                self.assertEqual(it['labels'], ['1', '2', '3'])
                self.assertIsNotNone(it['gold_choice'])
                gold = [o for o in it['options'] if o['label'] == it['gold_choice']]
                self.assertEqual(gold[0]['asin'], it['meta']['target_asin'])
                # only same-category distractors
                self.assertTrue(all(o['asin'] != 'B1' for o in it['options']))
            q = webshop_question('red ceramic coffee mug', out[0]['options'])
            self.assertIn('\\boxed{1}', q)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.jsonl'
            n = write_jsonl([dict(id='a', question='q', labels=['A'])], p)
            self.assertEqual(n, 1)
            self.assertEqual(read_jsonl(p)[0]['id'], 'a')


if __name__ == '__main__':
    unittest.main()