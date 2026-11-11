"""Build no-gold choice datasets (DailyDilemma, static WebShop choice) as JSONL."""
import argparse
from pathlib import Path
from experiments.choice_data import (load_dailydilemma, load_webshop, write_jsonl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['dailydilemma', 'webshop'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--candidates', type=int, default=4)
    p.add_argument('--csv', type=Path, help='DailyDilemma aggregated CSV')
    p.add_argument('--items', type=Path, help='WebShop items_shuffle_1000.json')
    p.add_argument('--instructions', type=Path, help='WebShop items_ins_v2_1000.json')
    a = p.parse_args()
    if a.dataset == 'dailydilemma':
        if not a.csv: p.error('--csv required')
        items = load_dailydilemma(a.csv, a.limit, a.seed)
    else:
        if not a.items or not a.instructions: p.error('--items/--instructions required')
        items = load_webshop(a.items, a.instructions, a.limit, a.candidates, a.seed)
    n = write_jsonl(items, a.out)
    print(f'wrote {n} items to {a.out}')
    if items:
        print(items[0]['question'][:600])


if __name__ == '__main__':
    main()