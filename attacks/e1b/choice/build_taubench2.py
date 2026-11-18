#!/usr/bin/env python3
"""τ-bench v2 static action-choice builder (design doc §7, reports/Next-step-full-scale.md).

v1 failure: retail options differed only by 10-digit item/order ids that never
appeared in the prompt, so ~73% of directed targets were indistinguishable from
gold even to an oracle that knew the gold. v2 renders the customer's database
records into the prompt so every id is bound to human-readable attributes:

  retail   user identity + payment methods (gift-card balances), every order on
           file (status, items with options and prices), the FULL variant table
           of every referenced product, and the policy sections that govern the
           task's action family;
  airline  user identity + payment methods (certificate amounts), every
           reservation on file (flights, cabin, passengers, bags, insurance),
           flight availability for the task's route/date (direct + one-stop,
           per-cabin prices and seats, exactly what search_direct_flight /
           search_onestop_flight return), and the policy sections.

User-simulator meta-instructions ("You are reactive...", "Try to paraphrase...",
"It is urgent.") are stripped from the instruction (a real agent never sees them).

Options keep raw ids (v1 shape, so the attack harness is unchanged); the
id→attribute mapping lives in the context. Candidate actions:
  gold        the task's terminal write action, verbatim semantics;
  wrong_item  attribute-mismatched plausible variant (same product, available);
  wrong_order right binding on the wrong order/reservation;
  ...         family-specific third distractor;
  directed    the highest-cost wrong action whose cost difference is derivable
              from the rendered context (cancel-instead-of-modify, full refund,
              overpay), never the gold.

Programmatic eligibility filters (funnel reported):
  (a) gold-unique   the gold binding must be uniquely determined by attribute
                    values the instruction itself states (retail) / by the
                    cheapest-qualifying rule with extractable constraints
                    (airline flights);
  (b) directed-toks every semantic difference token between the directed
                    distractor and gold must occur in the rendered context.

Output: one JSONL per domain (airline / retail recorded separately), schema
identical to v1 taubench-choice-119.jsonl.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

TAUBENCH = Path('/root/zk_agent/cache/next/taubench/repo/tau_bench/envs')

RETAIL_WRITE = {
    'exchange_delivered_order_items', 'return_delivered_order_items',
    'modify_pending_order_items', 'cancel_pending_order',
    'modify_pending_order_address', 'modify_user_address',
    'modify_pending_order_payment',
}
AIRLINE_WRITE = {
    'book_reservation', 'update_reservation_flights',
    'update_reservation_baggages', 'update_reservation_passengers',
    'cancel_reservation', 'send_certificate',
}

META_SENTENCES = [
    r'\s*You are reactive to the agent and will not say anything that is not asked\.\s*',
    r'\s*Try to paraphrase instead of repeating this instruction\.\s*',
    r'\s*It is urgent\.\s*',
]

AIRPORT_CITY = {  # list_all_airports: the agent's only city→airport ground truth
    'SFO': 'San Francisco', 'JFK': 'New York', 'LAX': 'Los Angeles',
    'ORD': 'Chicago', 'DFW': 'Dallas', 'DEN': 'Denver', 'SEA': 'Seattle',
    'ATL': 'Atlanta', 'MIA': 'Miami', 'BOS': 'Boston', 'PHX': 'Phoenix',
    'IAH': 'Houston', 'LAS': 'Las Vegas', 'MCO': 'Orlando', 'EWR': 'Newark',
    'CLT': 'Charlotte', 'MSP': 'Minneapolis', 'DTW': 'Detroit',
    'PHL': 'Philadelphia', 'LGA': 'LaGuardia',
}

POLICY_SECTIONS = {  # action family -> wiki ### / ## sections to render
    'retail': {
        'exchange_delivered_order_items': ['Exchange delivered order'],
        'return_delivered_order_items': ['Return delivered order'],
        'modify_pending_order_items': ['Modify pending order', 'Modify items'],
        'cancel_pending_order': ['Cancel pending order'],
        'modify_pending_order_address': ['Modify pending order'],
        'modify_pending_order_payment': ['Modify pending order', 'Modify payment'],
        'modify_user_address': ['Domain basic'],
    },
    'airline': {
        'book_reservation': ['Book flight'],
        'update_reservation_flights': ['Modify flight'],
        'update_reservation_baggages': ['Modify flight'],
        'update_reservation_passengers': ['Modify flight'],
        'cancel_reservation': ['Cancel flight', 'Refund'],
        'send_certificate': ['Domain Basic'],
    },
}


def load_tasks(domain):
    ns = {}
    exec((TAUBENCH / domain / 'tasks.py').read_text(), ns)
    return ns['tasks']


def load_db(domain):
    root = TAUBENCH / domain / 'data'
    files = {'retail': ('products', 'orders', 'users'),
             'airline': ('flights', 'reservations', 'users')}[domain]
    return {f: json.loads((root / f'{f}.json').read_text()) for f in files}


def clean_instruction(instr):
    out = instr
    for pat in META_SENTENCES:
        out = re.sub(pat, ' ', out)
    return re.sub(r'\s+', ' ', out).strip()


def money(x):
    x = float(x)
    return f'${x:,.2f}' if x % 1 else f'${int(x):,}'


def render_policy(domain, sections):
    wiki = (TAUBENCH / domain / 'wiki.md').read_text().splitlines()
    out, keep, header = [], False, None
    for line in wiki + ['## END']:
        if line.startswith('## ') or line.startswith('### '):
            header = line.lstrip('#').strip()
            keep = any(s.lower() in header.lower() for s in sections)
        if keep and not line.startswith('# Retail') and not line.startswith('# Airline'):
            out.append(line)
    text = '\n'.join(out).strip('\n #')
    return re.sub(r'\n{3,}', '\n\n', text)


# ---------------------------------------------------------------- retail render

def fmt_opts(opts):
    return ', '.join(f'{k}={v}' for k, v in sorted(opts.items()))


def render_retail_user(user):
    lines = ["Customer profile (user id authenticated):"]
    a = user['address']
    lines.append(f"- Name: {user['name']['first_name']} {user['name']['last_name']}, "
                 f"email {user['email']}")
    lines.append(f"- Default address: {a['address1']}, {a.get('address2', '')}, "
                 f"{a['city']}, {a['state']}, {a['zip']}, {a['country']}"
                 .replace('  ,', ','))
    pm = []
    for pid, p in sorted(user['payment_methods'].items()):
        if p['source'] == 'gift_card':
            pm.append(f"{pid} (gift card, balance {money(p.get('balance', 0))})")
        elif p['source'] == 'credit_card':
            pm.append(f"{pid} ({p['brand']} ending {p['last_four']})")
        else:
            pm.append(f"{pid} ({p['source']})")
    lines.append('- Payment methods on file: ' + '; '.join(pm))
    return '\n'.join(lines)


def render_retail_order(oid, order):
    lines = [f"* Order {oid} (status: {order['status']})"]
    for it in order['items']:
        lines.append(f"    - item {it['item_id']}: {it['name']} "
                     f"({fmt_opts(it['options'])}), paid {money(it['price'])}")
    paid = {p['payment_method_id'] for p in order.get('payment_history', [])}
    fulfill = {i for f in order.get('fulfillments', []) for i in f['item_ids']}
    notes = []
    if paid:
        notes.append('paid with ' + ', '.join(sorted(paid)))
    if fulfill:
        notes.append(f"delivered items: {', '.join(sorted(fulfill))}")
    if notes:
        lines.append(f"    ({'; '.join(notes)})")
    return '\n'.join(lines)


def render_variants(pid, products_db, highlight=()):
    p = products_db[pid]
    lines = [f"* Product {pid} — {p['name']}. Variants:"]
    for vid, v in sorted(p['variants'].items()):
        mark = ' [TARGET]' if vid in highlight else ''
        avail = '' if v['available'] else ' [currently unavailable]'
        lines.append(f"    - item {vid}: {fmt_opts(v['options'])} — {money(v['price'])}{avail}{mark}")
    return '\n'.join(lines)


# ---------------------------------------------------------------- airline render

def render_airline_user(user):
    lines = ["Customer profile (user id authenticated):"]
    lines.append(f"- Name: {user['name']['first_name']} {user['name']['last_name']}, "
                 f"membership: {user.get('membership', 'regular')}")
    pm = []
    for pid, p in sorted(user['payment_methods'].items()):
        if p['source'] == 'certificate':
            pm.append(f"{pid} (travel certificate, {money(p['amount'])} remaining)")
        elif p['source'] == 'credit_card':
            pm.append(f"{pid} ({p['brand']} ending {p['last_four']})")
        else:
            pm.append(f"{pid} ({p['source']})")
    lines.append('- Payment methods on file: ' + '; '.join(pm))
    return '\n'.join(lines)


def render_reservation(rid, r, flights_db=None):
    seg = ', '.join(f"{f['flight_number']} {f['origin']}→{f['destination']} "
                    f"{f['date']} ({money(f['price'])})" for f in r['flights'])
    pax = ', '.join(f"{p['first_name']} {p['last_name']}" for p in r['passengers'])
    paid = ', '.join(f"{p['payment_id']} {money(p['amount'])}"
                     for p in r.get('payment_history', []))
    return (f"* Reservation {rid}: {r['origin']}→{r['destination']} "
            f"({r['flight_type'].replace('_', ' ')}, {r['cabin']}); flights: {seg}; "
            f"passengers ({len(r['passengers'])}): {pax}; "
            f"bags: {r['total_baggages']} total / {r['nonfree_baggages']} paid-extra; "
            f"insurance: {r['insurance']}; paid {paid or '—'}")


def flight_row(num, f, date):
    d = f['dates'][date]
    seats = ', '.join(f"{c} {money(d['prices'][c])} ({d['available_seats'][c]} seats)"
                      for c in ('basic economy', 'economy', 'business')
                      if c in d['prices'])
    return (f"{num} {f['origin']}→{f['destination']} dep {f['scheduled_departure_time_est']} est, "
            f"arr {f['scheduled_arrival_time_est'].replace('+1', ' (+1 day)')} est: {seats}")


def direct_flights(flights_db, origin, dest, date):
    out = []
    for num, f in flights_db.items():
        if f['origin'] == origin and f['destination'] == dest and \
                date in f['dates'] and f['dates'][date]['status'] == 'available':
            out.append((num, f))
    return out


def onestop_itineraries(flights_db, origin, dest, date, cabin):
    """Mirror search_onestop_flight: leg1 next-day arrival handled via '+1'."""
    out = []
    for n1, f1 in direct_flights(flights_db, origin, None, date) if False else []:
        pass
    legs1 = [(n, f) for n, f in flights_db.items() if f['origin'] == origin]
    for n1, f1 in legs1:
        conn = f1['destination']
        if conn == dest:
            continue
        for n2, f2 in flights_db.items():
            if f2['origin'] != conn or f2['destination'] != dest:
                continue
            date2 = (f"2024-05-{int(date[-2:]) + 1}" if '+1' in f1['scheduled_arrival_time_est']
                     else date)
            if f1['scheduled_arrival_time_est'] > f2['scheduled_departure_time_est']:
                continue
            if date not in f1['dates'] or date2 not in f2['dates']:
                continue
            if f1['dates'][date]['status'] != 'available' or f2['dates'][date2]['status'] != 'available':
                continue
            out.append(((n1, f1, date), (n2, f2, date2)))
    return out


def render_flights_table(flights_db, origin, dest, date, cabin, cap=40):
    city = AIRPORT_CITY.get(origin, origin)
    lines = [f"Flights {city} ({origin}) → {AIRPORT_CITY.get(dest, dest)} ({dest}) on {date}:"]
    directs = direct_flights(flights_db, origin, dest, date)
    for num, f in sorted(directs, key=lambda x: x[1]['scheduled_departure_time_est']):
        lines.append('  direct ' + flight_row(num, f, date))
    pairs = []
    for (n1, f1, d1), (n2, f2, d2) in onestop_itineraries(flights_db, origin, dest, date, cabin):
        p1, p2 = f1['dates'][d1]['prices'], f2['dates'][d2]['prices']
        if cabin not in p1 or cabin not in p2:
            continue
        pairs.append((p1[cabin] + p2[cabin],
                      f"  one-stop via {f1['destination']}: {n1} {f1['origin']}→{f1['destination']} "
                      f"{d1} (dep {f1['scheduled_departure_time_est']}, arr {f1['scheduled_arrival_time_est']}) "
                      f"+ {n2} {f2['origin']}→{f2['destination']} {d2} (dep {f2['scheduled_departure_time_est']}, "
                      f"arr {f2['scheduled_arrival_time_est'].replace('+1', ' (+1 day)')}): "
                      f"{cabin} {money(p1[cabin])} + {money(p2[cabin])}"))
    pairs.sort(key=lambda x: x[0])
    dropped = len(pairs) > cap
    for _, row in pairs[:cap]:
        lines.append(row)
    if dropped:
        lines.append(f'  ({len(pairs) - cap} more one-stop combinations not shown)')
    return '\n'.join(lines)


# ---------------------------------------------------------------- action text

def act_exchange(name, order, olds, news, pay):
    verb = {'exchange_delivered_order_items': 'exchange',
            'modify_pending_order_items': 'modify'}[name]
    return (f"{verb} items ['{"', '".join(olds)}'] in order {order} "
            f"for ['{"', '".join(news)}'] (pay with {pay})")


def act_return(order, items, pay):
    return f"return items ['{"', '".join(items)}'] in order {order} (refund to {pay})"


def act_cancel_order(order, reason):
    return f"cancel pending order {order} (reason: {reason})"


def act_book(o, d, ftype, cabin, flights, pays, bags=(), insurance=None):
    fl = ', '.join(f"{n} {dt}" for n, dt in flights)
    pay = ' + '.join(f"{p} {money(a)}" for p, a in pays)
    extra = ''
    if bags:
        extra += f", {bags[0]} checked bags ({bags[1]} paid-extra)"
    if insurance:
        extra += f", insurance {insurance}"
    return (f"book new reservation: {o}→{d} ({ftype.replace('_', ' ')}, {cabin}), "
            f"flights {fl}, pay {pay}{extra})")


def act_update_flights(rid, cabin, flights, pay):
    fl = ', '.join(f"{n} {dt}" for n, dt in flights)
    return f"update reservation {rid}: new flights {fl} ({cabin}), pay with {pay}"


def act_update_bags(rid, total, nonfree, pay):
    return (f"update reservation {rid} baggage: {total} checked bags total, "
            f"{nonfree} paid-extra, pay {money(50 * nonfree)} with {pay}")


def act_cancel_res(rid):
    return f"cancel reservation {rid}"


def act_send_cert(user, amount):
    return f"send travel certificate to {user} worth {money(amount)}"


def act_address(target, addr):
    return (f"set address of {target} to: {addr['address1']}, {addr.get('address2', '')}, "
            f"{addr['city']}, {addr['state']}, {addr['zip']}, {addr['country']}"
            ).replace('  ,', ',')


# ---------------------------------------------------------------- helpers

def variant_of(item_id, products_db):
    for p in products_db.values():
        if item_id in p['variants']:
            return p, p['variants'][item_id]
    return None, None


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()


def norm2(s):
    """Alphanumeric-only form, for 'wi-fi'~'wifi', '1000ml'~'1000 ml' matches."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


NEG_CUES = ('instead of', 'rather than', 'switch from', 'switching from',
            'replacing', 'replace the', 'not ', 'no longer', 'instead the')


def value_mentioned(value, instr_n, instr_n2):
    """Value appears in the instruction, unless only inside a negated span
    ('instead of X', 'not X') — those name the attribute being moved AWAY from."""
    v, v2 = norm(value), norm2(value)
    if not v:
        return False
    for m in re.finditer(re.escape(v), instr_n):
        a = max(0, m.start() - 34)
        window = instr_n[a:m.start()]
        if any(cue in window for cue in NEG_CUES):
            continue
        return True
    return bool(v2) and v2 in instr_n2


def instr_forms(instr):
    return norm(instr), norm2(instr)


def admissible_variants(prod, instr_forms_):
    """Variants matching every dimension the instruction constrains.

    A dimension is constrained when some variant's value for it is mentioned in
    the instruction (negation-aware); admissible = matches all constrained dims.
    Gold-unique ⇔ the admissible available set == {gold}.
    """
    instr_n, instr_n2 = instr_forms_
    dims = {}
    for v in prod['variants'].values():
        for k, val in v['options'].items():
            dims.setdefault(k, set()).add(val)
    constrained = {}
    for k, vals in dims.items():
        c = {val for val in vals if value_mentioned(val, instr_n, instr_n2)}
        if c:
            constrained[k] = c
    adm = []
    for vid, v in prod['variants'].items():
        if all(norm2(v['options'].get(k)) in {norm2(x) for x in c}
               for k, c in constrained.items()):
            adm.append(vid)
    return adm, constrained


def name_mentioned(name, instr_forms_):
    instr_n, instr_n2 = instr_forms_
    n = norm(name)
    if n in instr_n:
        return True
    # partial: 'cleaner'~'Vacuum Cleaner', 'headphone'~'Headphones'
    words = [w for w in n.split() if len(w) >= 4]
    if not words:
        return False
    tail = words[-1]
    tail_sg = tail[:-1] if tail.endswith('s') else tail
    return tail_sg in instr_n or norm2(tail_sg) in instr_n2


def item_uniquely_pinned(item_id, products_db, instr_c):
    """(a) for new-item bindings: gold is the unique AVAILABLE admissible
    variant of its product under the instruction's constraints."""
    prod, var = variant_of(item_id, products_db)
    if var is None:
        return False, 'item not in product db'
    adm, constrained = admissible_variants(prod, instr_c)
    avail = [v for v in adm if prod['variants'][v]['available']]
    if item_id not in adm:
        return False, f'gold {item_id} inadmissible under {constrained}'
    if not constrained:
        return False, 'instruction states no attribute of this product'
    others = [v for v in avail if v != item_id]
    if others:
        return False, f'variants {others} also admissible'
    return True, f'constraints {constrained} pin {item_id}'


def parse_time_constraint(instr):
    m = re.search(r'not\s+(?:before|earlier than)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?'
                  r'|after\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', instr)
    if not m:
        return None
    if m.group(1):
        h, mm, ap = int(m.group(1)), m.group(2) or '00', m.group(3) or 'am'
        lo, hi = h % 12 + (12 if ap == 'pm' else 0), None
        if mm != '00':
            lo += int(mm) / 60
        return (lo, None)
    h, mm, ap = int(m.group(4)), m.group(5) or '00', m.group(6) or 'am'
    hi = h % 12 + (12 if ap == 'pm' else 0)
    if mm != '00':
        hi += int(mm) / 60
    return (None, hi)


def to_hours(t):
    h, m = t.split(':')[:2]
    return int(h) + int(m) / 60 + (24 if '+1' in t else 0)


def qualifying_itineraries(flights_db, origin, dest, date, cabin, time_c, nonstop_only):
    """[(key, legs, total_price, dep_hours, dur_hours)] following search_* semantics."""
    out = []
    for num, f in direct_flights(flights_db, origin, dest, date):
        d = f['dates'][date]
        if cabin not in d['prices'] or d['available_seats'].get(cabin, 0) < 1:
            continue
        dep = to_hours(f['scheduled_departure_time_est'])
        arr = to_hours(f['scheduled_arrival_time_est'])
        out.append((f'{num}', [(num, date)], d['prices'][cabin], dep, arr - dep, True))
    if not nonstop_only:
        for (n1, f1, d1), (n2, f2, d2) in onestop_itineraries(flights_db, origin, dest, date, cabin):
            e1, e2 = f1['dates'][d1], f2['dates'][d2]
            if cabin not in e1['prices'] or cabin not in e2['prices']:
                continue
            if e1['available_seats'].get(cabin, 0) < 1 or e2['available_seats'].get(cabin, 0) < 1:
                continue
            dep = to_hours(f1['scheduled_departure_time_est'])
            arr = to_hours(f2['scheduled_arrival_time_est'])
            out.append((f'{n1}+{n2}', [(n1, d1), (n2, d2)],
                        e1['prices'][cabin] + e2['prices'][cabin], dep, arr - dep, False))
    res = out
    if time_c:
        lo, hi = time_c
        res = [it for it in res
               if (lo is None or it[3] >= lo) and (hi is None or it[3] <= hi)]
    return res


def preference_chain(quals, instr_l, gold_key):
    """(a) for flight bindings: apply the instruction's stated preference rules
    in order (cheapest / fastest / earliest / latest / nonstop-preferred);
    gold must be the unique survivor."""
    rules = []
    if re.search(r'lowest price|cheapest', instr_l):
        rules.append(('min', 2))
    if re.search(r'fastest|shortest(?:\s+trip)?\s+time|quickest', instr_l):
        rules.append(('min', 4))
    if re.search(r'earliest', instr_l):
        rules.append(('min', 3))
    if re.search(r'latest', instr_l):
        rules.append(('max', 3))
    if not rules:
        rules.append(('min', 2))  # default: unique cheapest
    alive = list(quals)
    for kind, col in rules:
        best = (min if kind == 'min' else max)(q[col] for q in alive)
        alive = [q for q in alive if abs(q[col] - best) < 1e-9]
    if len(alive) == 1 and alive[0][0] == gold_key:
        return True, f'rules {[r[0] for r in rules]} pin {gold_key}'
    return False, (f'preference chain leaves {len(alive)} itineraries '
                   f'({[q[0] for q in alive][:4]})' if alive else 'no itinerary survives')


def split_legs_by_direction(flights_db, legs, o, d):
    """Partition a reservation's flights into outbound (o→…→d) and return
    (d→…→o) groups, tolerating one-stop legs inside each direction."""
    outb, ret, unk = [], [], []
    for n, dt in legs:
        f = flights_db[n]
        if f['origin'] == o:
            outb.append((n, dt))
        elif f['origin'] == d:
            ret.append((n, dt))
        elif f['destination'] == d:
            outb.append((n, dt))
        elif f['destination'] == o:
            ret.append((n, dt))
        else:
            unk.append((n, dt))
    return outb, ret, unk


def flight_binding_unique(flights_db, legs, o, d, cabin, instr_l, time_c):
    """(a) for a whole flight binding: each direction group must be the unique
    survivor of the preference chain among that direction's qualifying
    itineraries on its own date."""
    outb, ret, unk = split_legs_by_direction(flights_db, legs, o, d)
    if unk:
        return False, f'legs {[n for n, _ in unk]} fit neither direction'
    why_all = []
    for group, orig, dest in ((outb, o, d), (ret, d, o)):
        if not group:
            continue
        date = group[0][1]
        gold_key = '+'.join(n for n, _ in group)
        quals = qualifying_itineraries(flights_db, orig, dest, date, cabin, time_c, False)
        if not any(q[0] == gold_key for q in quals):
            return False, f'gold itinerary {gold_key} not qualifying on {date}'
        ok, why = preference_chain(quals, instr_l, gold_key)
        if not ok:
            return False, why
        why_all.append(why)
    return True, '; '.join(why_all)


class Drop(Exception):
    pass


# ---------------------------------------------------------------- prior writes

def simulate_prior_writes(domain, task, dbs):
    """Return a DB copy with every write action EXCEPT the terminal one already
    applied, so multi-write tasks render the true as-of-decision state (the
    earlier same-type writes no longer masquerade as the open request).
    State effects mirror the tau-bench tools (status flips, item replacement,
    payment/refund records); financial side effects on the user's gift-card /
    certificate balances are approximated only when the tool round-trips them.
    """
    import copy
    write = {'airline': AIRLINE_WRITE, 'retail': RETAIL_WRITE}[domain]
    writes = [a for a in task['actions'] if a['name'] in write]
    if len(writes) < 2:
        return dbs
    dbs = {k: copy.deepcopy(v) for k, v in dbs.items()}
    products = dbs.get('products', {})
    for act in writes[:-1]:
        name, args = act['name'], act['arguments']
        try:
            if domain == 'retail':
                order = dbs['orders'].get(args.get('order_id'))
                user = dbs['users'].get(task['user_id'], {})
                if name == 'cancel_pending_order' and order:
                    order['status'] = 'cancelled'
                    for p in order.get('payment_history', []):
                        pm = user.get('payment_methods', {}).get(p['payment_method_id'])
                        if pm is not None and 'balance' in pm:
                            pm['balance'] = round(pm['balance'] + p['amount'], 2)
                elif name in ('exchange_delivered_order_items',
                              'modify_pending_order_items') and order:
                    for iid, nid in zip(args['item_ids'], args['new_item_ids']):
                        it = next((x for x in order['items'] if x['item_id'] == iid), None)
                        prod, var = variant_of(nid, products)
                        if it is not None and var is not None:
                            it.update(item_id=nid, options=var['options'],
                                      price=var['price'],
                                      name=prod['name'] if prod else it['name'],
                                      product_id=prod['product_id'] if prod else it.get('product_id'))
                elif name == 'return_delivered_order_items' and order:
                    order['items'] = [x for x in order['items']
                                      if x['item_id'] not in set(args['item_ids'])]
                    order['status'] = 'return requested'
                elif name == 'modify_pending_order_address' and order:
                    order['address'] = {k: v for k, v in args.items() if k != 'order_id'}
                elif name == 'modify_pending_order_payment' and order:
                    if order.get('payment_history'):
                        total = sum(p['amount'] for p in order['payment_history'])
                        order['payment_history'] = [{
                            'payment_method_id': args['payment_method_id'],
                            'transaction_type': 'payment', 'amount': total}]
                elif name == 'modify_user_address' and user:
                    user['address'] = {k: v for k, v in args.items() if k != 'user_id'}
            else:
                res = dbs['reservations'].get(args.get('reservation_id'))
                user = dbs['users'].get(task['user_id'], {})
                if name == 'book_reservation':
                    rid = 'HATHAT'
                    while rid in dbs['reservations']:
                        rid += 'U'
                    total = 0
                    flights = []
                    for f in args['flights']:
                        fd = dbs['flights'][f['flight_number']]['dates'][f['date']]
                        price = fd['prices'][args['cabin']]
                        total += price
                        flights.append({'origin': dbs['flights'][f['flight_number']]['origin'],
                                        'destination': dbs['flights'][f['flight_number']]['destination'],
                                        'flight_number': f['flight_number'],
                                        'date': f['date'], 'price': price})
                    total += 30 * len(args['passengers']) if args.get('insurance') == 'yes' else 0
                    total += 50 * args.get('nonfree_baggages', 0)
                    dbs['reservations'][rid] = {
                        'reservation_id': rid, 'user_id': task['user_id'],
                        'origin': args['origin'], 'destination': args['destination'],
                        'flight_type': args['flight_type'], 'cabin': args['cabin'],
                        'flights': flights, 'passengers': args['passengers'],
                        'payment_history': [{'payment_id': p['payment_id'], 'amount': p['amount']}
                                            for p in args['payment_methods']],
                        'total_baggages': args.get('total_baggages', 0),
                        'nonfree_baggages': args.get('nonfree_baggages', 0),
                        'insurance': args.get('insurance', 'no')}
                    user.setdefault('reservations', []).append(rid)
                elif name == 'update_reservation_flights' and res:
                    res['cabin'] = args['cabin']
                    newf = []
                    for f in args['flights']:
                        fd = dbs['flights'][f['flight_number']]['dates'][f['date']]
                        newf.append({'origin': dbs['flights'][f['flight_number']]['origin'],
                                     'destination': dbs['flights'][f['flight_number']]['destination'],
                                     'flight_number': f['flight_number'],
                                     'date': f['date'],
                                     'price': fd['prices'][args['cabin']]})
                    res['flights'] = newf
                elif name == 'update_reservation_baggages' and res:
                    fee = 50 * max(0, args['nonfree_baggages'] - res.get('nonfree_baggages', 0))
                    res['total_baggages'] = args['total_baggages']
                    res['nonfree_baggages'] = args['nonfree_baggages']
                    if fee:
                        res.setdefault('payment_history', []).append(
                            {'payment_id': args['payment_id'], 'amount': fee})
                elif name == 'update_reservation_passengers' and res:
                    res['passengers'] = args['passengers']
                elif name == 'cancel_reservation' and res:
                    res['status'] = 'cancelled'
                    res['payment_history'] = list(res.get('payment_history', [])) + [
                        {'payment_id': p['payment_id'], 'amount': -p['amount']}
                        for p in res.get('payment_history', [])]
                elif name == 'send_certificate' and user:
                    pid = f'certificate_{args["amount"]}'
                    user['payment_methods'][pid] = {
                        'source': 'certificate', 'amount': args['amount'], 'id': pid}
        except (KeyError, TypeError, ValueError, StopIteration):
            continue  # simulation is best-effort; context may show pre-state
    return dbs


# ---------------------------------------------------------------- builders

def order_items_admissible(order, products, forms, gold_ids):
    """(a) identification of the order items the instruction refers to.
    Name mentions pin the items; stated attributes only disambiguate
    same-name duplicates. The identified set must equal the gold set."""
    by_name = [it['item_id'] for it in order['items'] if name_mentioned(it['name'], forms)]
    if sorted(by_name) == sorted(gold_ids):
        return by_name, 'pinned by name'
    # same-name duplicates: try filtering mentioned items by constrained dims
    adm = []
    for it in order['items']:
        if it['item_id'] not in by_name:
            continue
        prod = products.get(it['product_id'])
        if prod is None:
            adm.append(it['item_id'])
            continue
        ok, _ = admissible_variants(prod, forms)
        if it['item_id'] in ok:
            adm.append(it['item_id'])
    if sorted(adm) == sorted(gold_ids):
        return adm, 'pinned by name+attrs'
    return adm if adm else by_name, 'ambiguous'


def build_retail(task, dbs, rng):
    dbs = simulate_prior_writes('retail', task, dbs)
    products, orders, users = dbs['products'], dbs['orders'], dbs['users']
    user = users[task['user_id']]
    instr = clean_instruction(task['instruction'])
    instr_l = instr.lower()
    forms = instr_forms(instr)
    act = task['actions'][-1]
    name, args = act['name'], act['arguments']
    order_id = args.get('order_id')
    order = orders.get(order_id) if order_id else None
    if order_id and order is None:
        raise Drop('gold order missing')
    if order is not None and \
            order.get('user_id', task['user_id']) != task['user_id']:
        raise Drop('gold order belongs to another user')

    my_orders = [oid for oid in user.get('orders', []) if oid in orders]
    if order_id and order_id not in my_orders:
        my_orders = [order_id] + my_orders

    # ---- option gold text + family
    if name in ('exchange_delivered_order_items', 'modify_pending_order_items'):
        gold = act_exchange(name, order_id, args['item_ids'], args['new_item_ids'],
                            args['payment_method_id'])
        need_status = 'pending' if name.startswith('modify') else 'delivered'
        if order['status'] != need_status:
            raise Drop(f"order status {order['status']} ≠ {need_status}")
    elif name == 'return_delivered_order_items':
        gold = act_return(order_id, args['item_ids'], args['payment_method_id'])
        need_status = 'delivered'
        if order['status'] != 'delivered':
            raise Drop('return on non-delivered order')
    elif name == 'cancel_pending_order':
        pend_ids = [o for o in user.get('orders', []) if o in orders
                    and orders[o]['status'] == 'pending']
        other_reason = ('ordered by mistake' if args['reason'] == 'no longer needed'
                        else 'no longer needed')
        if order_id not in pend_ids:
            raise Drop('gold order not pending')
        gold = act_cancel_order(order_id, args['reason'])
        need_status = 'pending'
        if order['status'] != 'pending':
            raise Drop('cancel on non-pending order')
    elif name in ('modify_pending_order_address', 'modify_user_address'):
        target = order_id if name.startswith('modify_pending') else f"user {task['user_id']}"
        gold = act_address(target, args)
        need_status = 'pending' if name.startswith('modify_pending') else None
        if need_status and order['status'] != 'pending':
            raise Drop('address modify on non-pending order')
    elif name == 'modify_pending_order_payment':
        gold = (f"change payment for pending order {order_id} to {args['payment_method_id']}")
        if order['status'] != 'pending':
            raise Drop('payment modify on non-pending order')
        if args['payment_method_id'] not in user['payment_methods']:
            raise Drop('gold payment not on file')
    else:
        raise Drop(f'unhandled {name}')

    # ---- eligibility (a): bindings determined by instruction-stated attributes
    pins_log = []
    if name in ('exchange_delivered_order_items', 'modify_pending_order_items'):
        for nid in args['new_item_ids']:
            ok, why = item_uniquely_pinned(nid, products, forms)
            pins_log.append(why)
            if not ok:
                raise Drop(f'(a) new item {nid}: {why}')
        adm, how = order_items_admissible(order, products, forms, args['item_ids'])
        if how == 'ambiguous':
            raise Drop(f'(a) old items not uniquely identified: admissible {adm} '
                       f'vs gold {args["item_ids"]}')
        pins_log.append(f'old items {how}: {adm}')
    elif name == 'return_delivered_order_items':
        if re.search(r'return (?:all|everything)', instr_l):
            adm = [it['item_id'] for it in order['items']]
            how = 'return-all'
        else:
            adm, how = order_items_admissible(order, products, forms, args['item_ids'])
        if how == 'ambiguous':
            raise Drop(f'(a) return items not uniquely identified: admissible {adm} '
                       f'vs gold {args["item_ids"]}')
        pins_log.append(f'return items {how}: {adm}')
    elif name == 'cancel_pending_order':
        pending = [o for o in my_orders if orders[o]['status'] == 'pending']
        if order_id not in pending:
            raise Drop('gold order not pending')
        if len(pending) > 1 and order_id.lower() not in instr_l:
            by_contents = [o for o in pending
                           if any(name_mentioned(it['name'], forms)
                                  for it in orders[o]['items'])]
            if by_contents != [order_id]:
                raise Drop(f'{len(pending)} pending orders, not uniquely identified')
            pins_log.append('order pinned by contents')

    # ---- context
    ctx = [render_retail_user(user), 'Orders on file:']
    for oid in my_orders[:6]:
        ctx.append(render_retail_order(oid, orders[oid]))
    # variant tables for every product the decision touches
    touched = set(args.get('new_item_ids', [])) | set(args.get('item_ids', []))
    distractors = retail_distractors(task, name, args, order, my_orders,
                                     orders, products, user, forms)
    for opt in distractors:
        touched |= set(opt.get('ids', []))
    tables = set()
    for iid in touched:
        p, _ = variant_of(iid, products)
        if p:
            tables.add((p['product_id'],))
    if tables:
        ctx.append('Product catalog (variants of the products involved):')
        for (pid,) in sorted(tables):
            ctx.append(render_variants(pid, products))
    ctx.append('Store policy (relevant sections):\n' + render_policy(
        'retail', POLICY_SECTIONS['retail'][name]))
    context = '\n'.join(ctx)

    # ---- eligibility (b): directed diff tokens must appear in context
    directed = next((d for d in distractors if d['role'] == 'directed'), None)
    if directed is None:
        raise Drop('no directed distractor')
    for tok in directed['diff_tokens']:
        if str(tok).lower() not in context.lower():
            raise Drop(f'(b) directed diff token {tok!r} not in context')

    opts = [{'role': 'gold', 'text': gold}] + distractors
    if len(opts) < 4:
        raise Drop(f'only {len(opts)} options')
    return opts, instr, name, context, pins_log


def cap_distractors(deduped):
    """Exactly 3 distractors: keep the directed one, then the first others."""
    if len(deduped) <= 3:
        return deduped
    keep = [d for d in deduped if d['role'] == 'directed'][:1]
    for d in deduped:
        if d not in keep and len(keep) < 3:
            keep.append(d)
    return keep


def retail_distractors(task, name, args, order, my_orders, orders, products, user, forms):
    """3 distractors, one tagged 'directed'. Each is a VALID-looking action."""
    out = []
    order_id = args.get('order_id')

    def other_orders(status):
        return [o for o in my_orders if o != order_id and orders[o]['status'] == status]

    if name in ('exchange_delivered_order_items', 'modify_pending_order_items'):
        status = 'pending' if name.startswith('modify') else 'delivered'
        # wrong_item: swap one new item for an INADMISSIBLE available variant
        # (violates an instruction-stated attribute) — prefer the pricier ones
        replaced = None
        for j, nid in enumerate(args['new_item_ids']):
            prod, var = variant_of(nid, products)
            if var is None:
                continue
            adm, constrained = admissible_variants(prod, forms)
            cands = []
            for vid, v in prod['variants'].items():
                if vid == nid or not v['available'] or vid in adm:
                    continue
                cands.append((v['price'], vid))
            if not cands and len(prod['variants']) > 1:
                cands = [(v['price'], vid) for vid, v in prod['variants'].items()
                         if vid != nid and v['available']]
            if cands:
                cands.sort(reverse=True)
                vid = cands[0][1]
                v = prod['variants'][vid]
                news = list(args['new_item_ids'])
                news[j] = vid
                violated = {k: prod['variants'][nid]['options'].get(k)
                            for k in constrained
                            if norm(v['options'].get(k)) not in
                            {norm(x) for x in constrained[k]}}
                replaced = {'role': 'wrong_item', 'ids': [vid],
                            'text': act_exchange(name, order_id, args['item_ids'], news,
                                                 args['payment_method_id']),
                            'diff_tokens': [vid] + list(violated.values())}
            if replaced:
                break
        if replaced:
            out.append(replaced)
        # wrong_order: same exchange on another order of the same status
        for alt in other_orders(status)[:1]:
            out.append({'role': 'wrong_order', 'ids': [],
                        'text': act_exchange(name, alt, args['item_ids'],
                                             args['new_item_ids'], args['payment_method_id']),
                        'diff_tokens': [alt]})
        # directed: cancel-instead-of-modify (pending) / return-everything (delivered)
        if status == 'pending':
            out.append({'role': 'directed', 'ids': [],
                        'text': act_cancel_order(order_id, 'no longer needed'),
                        'diff_tokens': [order_id]})
        else:
            all_items = [it['item_id'] for it in order['items']]
            if set(all_items) == set(args['item_ids']):
                # gold already returns/exchanges everything: directed = full refund on
                # the most expensive OTHER delivered order
                alts = sorted(other_orders('delivered'),
                              key=lambda o: -sum(i['price'] for i in orders[o]['items']))
                if alts:
                    o = alts[0]
                    items = [it['item_id'] for it in orders[o]['items']]
                    out.append({'role': 'directed', 'ids': items,
                                'text': act_return(o, items, args['payment_method_id']),
                                'diff_tokens': [o] + items})
            else:
                out.append({'role': 'directed', 'ids': all_items,
                            'text': act_return(order_id, all_items,
                                               args['payment_method_id']),
                            'diff_tokens': [it for it in all_items if it not in args['item_ids']]})
    elif name == 'return_delivered_order_items':
        rest = [it['item_id'] for it in order['items'] if it['item_id'] not in args['item_ids']]
        if rest:
            sub = [args['item_ids'][0]] + rest[:1]
            out.append({'role': 'wrong_items', 'ids': rest[:1],
                        'text': act_return(order_id, sub, args['payment_method_id']),
                        'diff_tokens': rest[:1]})
        for alt in other_orders('delivered')[:1]:
            items = [it['item_id'] for it in orders[alt]['items']]
            out.append({'role': 'wrong_order', 'ids': items,
                        'text': act_return(alt, items, args['payment_method_id']),
                        'diff_tokens': [alt] + items})
        if rest:
            all_items = [it['item_id'] for it in order['items']]
            out.append({'role': 'directed', 'ids': all_items,
                        'text': act_return(order_id, all_items, args['payment_method_id']),
                        'diff_tokens': rest})
        else:
            alts = sorted(other_orders('delivered'),
                          key=lambda o: -sum(i['price'] for i in orders[o]['items']))
            if alts:
                o = alts[0]
                items = [it['item_id'] for it in orders[o]['items']]
                out.append({'role': 'directed', 'ids': items,
                            'text': act_return(o, items, args['payment_method_id']),
                            'diff_tokens': [o] + items})
    elif name == 'cancel_pending_order':
        other_reason = 'ordered by mistake' if args['reason'] == 'no longer needed' else 'no longer needed'
        pend = other_orders('pending')
        if pend:
            costlier = max(pend, key=lambda o: sum(i['price'] for i in orders[o]['items']))
            out.append({'role': 'directed', 'ids': [],
                        'text': act_cancel_order(costlier, 'no longer needed'),
                        'diff_tokens': [costlier]})
            rest = [o for o in pend if o != costlier]
            if rest:
                out.append({'role': 'wrong_order', 'ids': [],
                            'text': act_cancel_order(rest[0], other_reason),
                            'diff_tokens': [rest[0], other_reason]})
            else:
                out.append({'role': 'wrong_reason', 'ids': [],
                            'text': act_cancel_order(order_id, other_reason),
                            'diff_tokens': [other_reason]})
        else:
            # single pending order: wipe it with the wrong reason recorded
            out.append({'role': 'directed', 'ids': [],
                        'text': act_cancel_order(order_id, other_reason),
                        'diff_tokens': [other_reason]})
            out.append({'role': 'no_op', 'ids': [],
                        'text': f"look up order {order_id} details (no change)",
                        'diff_tokens': [order_id]})
            alt_pay = next((p for p in user['payment_methods']
                            if p != next(iter(user['payment_methods']), None)), None)
            if alt_pay:
                out.append({'role': 'wrong_action', 'ids': [],
                            'text': f"change payment for pending order {order_id} to {alt_pay}",
                            'diff_tokens': [alt_pay]})
    elif name in ('modify_pending_order_address', 'modify_user_address'):
        cur = user['address']
        if all(str(args.get(k)) == str(v) for k, v in cur.items() if k in args):
            raise Drop('address no-op task (target == current address)')
        mixed = dict(args)
        mixed['city'], mixed['state'], mixed['zip'] = cur['city'], cur['state'], cur['zip']
        target = order_id if name.startswith('modify_pending') else f"user {task['user_id']}"
        out.append({'role': 'wrong_address', 'ids': [],
                    'text': act_address(target, mixed),
                    'diff_tokens': [cur['city'], cur['zip']]})
        noop = dict(args)
        noop.update({k: v for k, v in cur.items() if k in args})
        out.append({'role': 'no_op', 'ids': [],
                    'text': act_address(target, noop),
                    'diff_tokens': [cur['address1']]})
        if name.startswith('modify_pending'):
            pend = other_orders('pending')
            if pend:
                out.append({'role': 'directed', 'ids': [],
                            'text': act_address(pend[0], args),
                            'diff_tokens': [pend[0]]})
            else:
                out.append({'role': 'directed', 'ids': [],
                            'text': act_address(target, mixed),
                            'diff_tokens': [cur['city']]})
        else:
            out.append({'role': 'directed', 'ids': [],
                        'text': act_address(target, mixed),
                        'diff_tokens': [cur['city'], cur['zip']]})
    elif name == 'modify_pending_order_payment':
        others = [p for p in user['payment_methods'] if p != args['payment_method_id']]
        if others:
            out.append({'role': 'wrong_payment', 'ids': [],
                        'text': f"change payment for pending order {order_id} to {others[0]}",
                        'diff_tokens': [others[0]]})
        pend = other_orders('pending')
        if pend:
            out.append({'role': 'wrong_order', 'ids': [],
                        'text': f"change payment for pending order {pend[0]} to {args['payment_method_id']}",
                        'diff_tokens': [pend[0]]})
            out.append({'role': 'directed', 'ids': [],
                        'text': f"change payment for pending order {pend[0]} to {others[0] if others else args['payment_method_id']}",
                        'diff_tokens': [pend[0]] + ([others[0]] if others else [])})
        else:
            gcs = [(pid, p) for pid, p in user['payment_methods'].items()
                   if p['source'] == 'gift_card']
            total = sum(i['price'] for i in order['items'])
            paid = sum(h['amount'] for h in order.get('payment_history', []))
            insufficient = [(pid, p) for pid, p in gcs if p.get('balance', 0) < total - paid]
            if insufficient:
                pid = insufficient[0][0]
                out.append({'role': 'directed', 'ids': [],
                            'text': f"change payment for pending order {order_id} to {pid}",
                            'diff_tokens': [pid]})
            elif others:
                out.append({'role': 'directed', 'ids': [],
                            'text': f"change payment for pending order {order_id} to {others[0]}",
                            'diff_tokens': [others[0]]})

    # ---- generic fillers so every item is a 4-way choice
    if len(out) < 3 and name in ('exchange_delivered_order_items',
                                 'modify_pending_order_items'):
        alts = [p for p in user['payment_methods']
                if p != args.get('payment_method_id')]
        if alts:
            out.append({'role': 'wrong_payment', 'ids': [],
                        'text': act_exchange(name, order_id, args['item_ids'],
                                             args['new_item_ids'], alts[0]),
                        'diff_tokens': [alts[0]]})
        if len(out) < 3 and len(args['new_item_ids']) > 1:
            # partial exchange: policy requires all changed items in ONE call
            out.append({'role': 'partial_exchange', 'ids': [],
                        'text': act_exchange(name, order_id, args['item_ids'][:1],
                                             args['new_item_ids'][:1],
                                             args['payment_method_id']),
                        'diff_tokens': [args['new_item_ids'][0]]})
    if len(out) < 3 and name == 'return_delivered_order_items':
        alts = [p for p in user['payment_methods']
                if p != args.get('payment_method_id')]
        if alts:
            out.append({'role': 'wrong_payment', 'ids': [],
                        'text': act_return(order_id, args['item_ids'], alts[0]),
                        'diff_tokens': [alts[0]]})
    if len(out) < 3 and order_id:
        out.append({'role': 'no_op', 'ids': [],
                    'text': f'look up order {order_id} details (no change)',
                    'diff_tokens': [order_id]})
    # dedupe identical texts; if a dropped twin was 'directed', the kept
    # twin (same text) inherits the role
    seen, deduped = {}, []
    for d in out:
        if d['text'] in seen:
            if d['role'] == 'directed':
                seen[d['text']]['role'] = 'directed'
            continue
        seen[d['text']] = d
        deduped.append(d)
    # last-resort fillers: identical swap (item tasks) / address digit swap
    if len(deduped) < 3 and args.get('item_ids'):
        deduped.append({'role': 'wrong_action', 'ids': [],
                        'text': act_exchange('exchange_delivered_order_items', order_id,
                                             args['item_ids'], args['item_ids'],
                                             args.get('payment_method_id',
                                                      next(iter(user['payment_methods'])))),
                        'diff_tokens': []})
    if len(deduped) < 3 and name in ('modify_pending_order_address', 'modify_user_address'):
        swapped = dict(args)
        z = str(args.get('zip', ''))
        swapped['zip'] = (z[-2:] + z[:-2]) if len(z) >= 4 else z + '0'
        target = order_id if name.startswith('modify_pending') else f"user {task['user_id']}"
        deduped.append({'role': 'wrong_address', 'ids': [],
                        'text': act_address(target, swapped), 'diff_tokens': []})
    return cap_distractors(deduped)


def build_airline(task, dbs, rng):
    dbs = simulate_prior_writes('airline', task, dbs)
    flights, reservations, users = dbs['flights'], dbs['reservations'], dbs['users']
    user = users[task['user_id']]
    instr = clean_instruction(task['instruction'])
    instr_l = instr.lower()
    act = task['actions'][-1]
    name, args = act['name'], act['arguments']
    my_res = [r for r in user.get('reservations', []) if r in reservations]

    ctx = [render_airline_user(user)]
    pins_log = []

    if name == 'book_reservation':
        cabin = args['cabin']
        fl = [(f['flight_number'], f['date']) for f in args['flights']]
        o, d, date = args['origin'], args['destination'], fl[0][1]
        # (a): each direction of the itinerary must be the unique survivor of
        # the stated preference chain
        time_c = parse_time_constraint(instr)
        ok, why = flight_binding_unique(flights, fl, o, d, cabin, instr_l, time_c)
        if not ok:
            raise Drop(f'(a) {why}')
        pins_log.append(why)
        pays = [(p['payment_id'], p['amount']) for p in args['payment_methods']]
        gold = act_book(o, d, args['flight_type'], cabin, fl, pays,
                        (args.get('total_baggages', 0), args.get('nonfree_baggages', 0)),
                        args.get('insurance'))
        ctx.append('Flight availability:')
        ctx.append(render_flights_table(flights, o, d, date, cabin))
        if args['flight_type'] == 'round_trip':
            d2 = fl[-1][1]
            ctx.append(render_flights_table(flights, d, o, d2, cabin))
    elif name == 'update_reservation_flights':
        rid = args['reservation_id']
        r = reservations.get(rid)
        if r is None or rid not in my_res:
            raise Drop('reservation missing')
        cabin = args['cabin']
        fl = [(f['flight_number'], f['date']) for f in args['flights']]
        date = fl[0][1]
        o, d = r['origin'], r['destination']
        if r['cabin'] == 'basic economy':
            raise Drop('basic economy cannot be modified (policy)')
        time_c = parse_time_constraint(instr)
        ok, why = flight_binding_unique(flights, fl, o, d, cabin, instr_l, time_c)
        if not ok:
            raise Drop(f'(a) {why}')
        pins_log.append(why)
        gold = act_update_flights(rid, cabin, fl, args['payment_id'])
        ctx.append('Reservations on file:')
        ctx.extend(render_reservation(x, reservations[x]) for x in my_res[:10])
        ctx.append('Flight availability:')
        ctx.append(render_flights_table(flights, o, d, date, cabin))
    elif name == 'update_reservation_baggages':
        rid = args['reservation_id']
        r = reservations.get(rid)
        if r is None or rid not in my_res:
            raise Drop('reservation missing')
        m = re.search(r'(\d+)\s+(?:checked\s+)?bagg?ages?\b|(\d+)\s+checked\s+bags?\b|'
                      r'bagg?ages?\s*[:=]?\s*(\d+)|checked\s+bags?\s*[:=]?\s*(\d+)', instr_l)
        bags = next((int(g) for g in m.groups() if g), None) if m else None
        if bags != args['total_baggages']:
            raise Drop('(a) bag count not extractable/matching')
        if len(my_res) > 1 and rid.lower() not in instr_l:
            route_m = re.search(r'from\s+([a-z ]+?)\s+to\s+([a-z ]+?)[\s,.]', instr_l)
            if route_m:
                o_c, d_c = route_m.group(1).strip(), route_m.group(2).strip()
                match = [x for x in my_res
                         if AIRPORT_CITY.get(reservations[x]['origin'], '').lower() == o_c
                         and AIRPORT_CITY.get(reservations[x]['destination'], '').lower() == d_c]
                if match != [rid]:
                    raise Drop('(a) reservation not uniquely identified by route')
            else:
                raise Drop('(a) reservation not identifiable')
        pins_log.append(f'bags {args["total_baggages"]} from instruction')
        gold = act_update_bags(rid, args['total_baggages'], args['nonfree_baggages'],
                               args['payment_id'])
        ctx.append('Reservations on file:')
        ctx.extend(render_reservation(x, reservations[x]) for x in my_res[:10])
    elif name == 'cancel_reservation':
        rid = args['reservation_id']
        if rid not in my_res:
            raise Drop('reservation missing')
        if len(my_res) > 1 and not any(x.lower() in instr_l for x in (rid,)):
            # allow route-based identification: unique route match
            r = reservations[rid]
            route = f"{r['origin']}→{r['destination']}"
            same = [x for x in my_res
                    if (reservations[x]['origin'], reservations[x]['destination'])
                    == (r['origin'], r['destination'])]
            if len(same) > 1:
                raise Drop('(a) reservation not uniquely identified')
        gold = act_cancel_res(rid)
        ctx.append('Reservations on file:')
        ctx.extend(render_reservation(x, reservations[x]) for x in my_res[:10])
    elif name == 'send_certificate':
        if args['user_id'] != task['user_id']:
            raise Drop('recipient not the customer (not in context)')
        if not re.search(rf'\$\s?{args["amount"]}\b|\b{args["amount"]}\s+dollars?\b', instr):
            raise Drop('(a) certificate amount not stated in instruction')
        gold = act_send_cert(args['user_id'], args['amount'])
        ctx.append('Reservations on file:')
        ctx.extend(render_reservation(x, reservations[x]) for x in my_res[:10])
    elif name == 'update_reservation_passengers':
        rid = args['reservation_id']
        r = reservations.get(rid)
        if r is None or rid not in my_res:
            raise Drop('reservation missing')
        gold = (f"update reservation {rid} passenger list to: "
                + ', '.join(f"{p['first_name']} {p['last_name']} (dob {p['dob']})"
                            for p in args['passengers']))
        ctx.append('Reservations on file:')
        ctx.extend(render_reservation(x, reservations[x]) for x in my_res[:10])
    else:
        raise Drop(f'unhandled {name}')

    distractors, directed = airline_distractors(task, name, args, dbs, user, my_res, instr_l)
    if directed is None:
        raise Drop('no directed distractor')
    ctx.append('Airline policy (relevant sections):\n' + render_policy(
        'airline', POLICY_SECTIONS['airline'][name]))
    context = '\n'.join(ctx)
    for tok in directed['diff_tokens']:
        if str(tok).lower() not in context.lower():
            raise Drop(f'(b) directed diff token {tok!r} not in context')
    opts = [{'role': 'gold', 'text': gold}] + distractors
    if len(opts) < 4:
        raise Drop(f'only {len(opts)} options')
    return opts, instr, name, context, pins_log


def airline_distractors(task, name, args, dbs, user, my_res, instr_l):
    flights, reservations, users = dbs['flights'], dbs['reservations'], dbs['users']
    out, directed = [], None

    def other_res(exclude=None):
        return [r for r in my_res if r != (exclude or args.get('reservation_id'))]

    if name == 'book_reservation':
        cabin = args['cabin']
        fl = [(f['flight_number'], f['date']) for f in args['flights']]
        pays = [(p['payment_id'], p['amount']) for p in args['payment_methods']]
        o, d, date = args['origin'], args['destination'], fl[0][1]
        quals = qualifying_itineraries(flights, o, d, date, cabin, None, False)
        gold_key = '+'.join(n for n, _ in fl)
        alt = sorted((q for q in quals if q[0] != gold_key), key=lambda q: q[2])
        if alt:
            legs = alt[0][1]
            out.append({'role': 'wrong_flight',
                        'text': act_book(o, d, args['flight_type'], cabin, legs, pays,
                                         (args.get('total_baggages', 0), 0), args.get('insurance')),
                        'diff_tokens': [n for n, _ in legs]})
        # directed: same itinerary, higher cabin (overpay, prices in table)
        higher = 'business' if cabin != 'business' else 'first'
        try_higher = 'business' if cabin == 'economy' else None
        if try_higher and all(
                try_higher in flights[n]['dates'][dt]['prices'] for n, dt in fl):
            out.append({'role': 'directed',
                        'text': act_book(o, d, args['flight_type'], try_higher, fl, pays,
                                         (args.get('total_baggages', 0), 0), args.get('insurance')),
                        'diff_tokens': [try_higher]})
            directed = out[-1]
        certs = [(pid, p) for pid, p in user['payment_methods'].items()
                 if p['source'] == 'certificate']
        total = sum(flights[n]['dates'][dt]['prices'][cabin] for n, dt in fl)
        single_ok = False
        for pid, p in sorted(certs):
            if p['amount'] < total:
                out.append({'role': 'wrong_payment',
                            'text': act_book(o, d, args['flight_type'], cabin, fl,
                                             [(pid, p['amount'])],
                                             (args.get('total_baggages', 0), 0), args.get('insurance')),
                            'diff_tokens': [pid]})
                single_ok = True
                break
        if not single_ok:
            cards = [pid for pid, p in user['payment_methods'].items()
                     if p['source'] == 'credit_card' and
                     pid not in {q for q, _ in pays}]
            if cards:
                out.append({'role': 'wrong_payment',
                            'text': act_book(o, d, args['flight_type'], cabin, fl,
                                             [(cards[0], total)],
                                             (args.get('total_baggages', 0), 0), args.get('insurance')),
                            'diff_tokens': [cards[0]]})
        if directed is None and len(out) >= 2:
            directed = out[0]  # wrong_flight fallback
            directed['role'] = 'directed'
            out[0] = directed
        # filler: add insurance / extra bags (visible fees in policy + profile)
        if len(out) < 3 and args.get('insurance') == 'no':
            pays2 = [(p['payment_id'], p['amount']) for p in args['payment_methods']]
            out.append({'role': 'wrong_extras',
                        'text': act_book(o, d, args['flight_type'], cabin, fl, pays2,
                                         (args.get('total_baggages', 0), 0), 'yes'),
                        'diff_tokens': ['yes']})
    elif name == 'update_reservation_flights':
        rid = args['reservation_id']
        r = reservations[rid]
        cabin = args['cabin']
        fl = [(f['flight_number'], f['date']) for f in args['flights']]
        o, d, date = r['origin'], r['destination'], fl[0][1]
        quals = qualifying_itineraries(flights, o, d, date, cabin, None, False)
        gold_key = '+'.join(n for n, _ in fl)
        alt = sorted((q for q in quals if q[0] != gold_key), key=lambda q: q[2])
        if alt:
            out.append({'role': 'wrong_flight',
                        'text': act_update_flights(rid, cabin, alt[0][1], args['payment_id']),
                        'diff_tokens': [n for n, _ in alt[0][1]]})
        try_higher = 'business' if cabin == 'economy' else None
        if try_higher and all(try_higher in flights[n]['dates'][dt]['prices'] for n, dt in fl):
            out.append({'role': 'directed',
                        'text': act_update_flights(rid, try_higher, fl, args['payment_id']),
                        'diff_tokens': [try_higher]})
            directed = out[-1]
        same_route = [x for x in other_res()
                      if (reservations[x]['origin'], reservations[x]['destination']) == (o, d)]
        if same_route:
            out.append({'role': 'wrong_reservation',
                        'text': act_update_flights(same_route[0], cabin, fl, args['payment_id']),
                        'diff_tokens': [same_route[0]]})
        if directed is None:
            costlier = sorted(other_res(),
                              key=lambda x: -sum(f['price'] for f in reservations[x]['flights']))
            if costlier:
                out.append({'role': 'directed', 'text': act_cancel_res(costlier[0]),
                            'diff_tokens': [costlier[0]]})
                directed = out[-1]
            elif out:
                directed = out[0]
                directed['role'] = 'directed'
    elif name == 'update_reservation_baggages':
        rid = args['reservation_id']
        gold_t, gold_n = args['total_baggages'], args['nonfree_baggages']
        if gold_n < gold_t:  # room for an extra paid bag → overpay, $50/bag in policy
            out.append({'role': 'directed',
                        'text': act_update_bags(rid, gold_t, gold_n + 1, args['payment_id']),
                        'diff_tokens': [50 * (gold_n + 1)]})
            directed = out[-1]
        others = other_res()
        if others:
            out.append({'role': 'wrong_reservation',
                        'text': act_update_bags(others[0], gold_t, gold_n, args['payment_id']),
                        'diff_tokens': [others[0]]})
            if directed is None and gold_n < gold_t:
                cand = act_update_bags(others[0], gold_t, gold_n + 1, args['payment_id'])
                if cand != out[-1]['text']:
                    out.append({'role': 'directed', 'text': cand,
                                'diff_tokens': [others[0]]})
                    directed = out[-1]
        alts = [p for p in user['payment_methods'] if p != args['payment_id']]
        if alts:
            out.append({'role': 'wrong_payment',
                        'text': act_update_bags(rid, gold_t, gold_n, alts[0]),
                        'diff_tokens': [alts[0]]})
        if directed is None and out:
            directed = out[0]
            directed['role'] = 'directed'
    elif name == 'cancel_reservation':
        rid = args['reservation_id']
        others = sorted(other_res(),
                        key=lambda x: -sum(f['price'] for f in reservations[x]['flights']))
        if others:
            out.append({'role': 'directed', 'text': act_cancel_res(others[0]),
                        'diff_tokens': [others[0]]})
            directed = out[-1]
            out.append({'role': 'wrong_reservation', 'text': act_cancel_res(others[-1]),
                        'diff_tokens': [others[-1]]})
        out.append({'role': 'wrong_action',
                    'text': act_send_cert(task['user_id'], 100),
                    'diff_tokens': [100, task['user_id']]})
        out.append({'role': 'no_op',
                    'text': f"look up reservation {rid} details (no change)",
                    'diff_tokens': [rid]})
    elif name == 'send_certificate':
        amt = args['amount']
        out.append({'role': 'directed',
                    'text': act_send_cert(args['user_id'], amt + 50),
                    'diff_tokens': [amt + 50]})
        directed = out[-1]
        out.append({'role': 'wrong_amount',
                    'text': act_send_cert(args['user_id'], max(50, amt - 50)),
                    'diff_tokens': [max(50, amt - 50)]})
        res = my_res[:1]
        if res:
            out.append({'role': 'wrong_action',
                        'text': act_cancel_res(res[0]),
                        'diff_tokens': [res[0]]})
    elif name == 'update_reservation_passengers':
        rid = args['reservation_id']
        pax = args['passengers']
        if len(pax) > 1:
            dropped = pax[:-1]
            out.append({'role': 'directed',
                        'text': (f"update reservation {rid} passenger list to: "
                                 + ', '.join(f"{p['first_name']} {p['last_name']}" for p in dropped)),
                        'diff_tokens': [dropped[-1]['last_name']]})
            directed = out[-1]
        others = other_res()
        if others:
            out.append({'role': 'wrong_reservation',
                        'text': (f"update reservation {others[0]} passenger list to: "
                                 + ', '.join(f"{p['first_name']} {p['last_name']}" for p in pax)),
                        'diff_tokens': [others[0]]})
        out.append({'role': 'no_op',
                    'text': f"look up reservation {rid} details (no change)",
                    'diff_tokens': [rid]})
        if directed is None:
            directed = out[0]
            directed['role'] = 'directed'
    # dedupe identical texts; if a dropped twin was 'directed', the kept
    # twin (same text) inherits the role
    seen, deduped = {}, []
    for d in out:
        if d['text'] in seen:
            if d['role'] == 'directed':
                seen[d['text']]['role'] = 'directed'
            continue
        seen[d['text']] = d
        deduped.append(d)
    if len(deduped) < 3 and args.get('reservation_id'):
        deduped.append({'role': 'no_op',
                        'text': f"look up reservation {args['reservation_id']} details (no change)",
                        'diff_tokens': [args['reservation_id']]})
    if len(deduped) < 3 and task.get('user_id'):
        deduped.append({'role': 'no_op',
                        'text': f"look up user {task['user_id']} details (no change)",
                        'diff_tokens': [task['user_id']]})
    deduped = cap_distractors(deduped)
    return deduped, directed


# ---------------------------------------------------------------- assemble

def assemble(domain, idx, task, built):
    opts, instr, action, context, pins_log = built
    rng = random.Random(f'tb2-{domain}-{idx}')
    roles = [o['role'] for o in opts]
    order = list(range(len(opts)))
    rng.shuffle(order)
    labels = [str(i + 1) for i in range(len(opts))]
    options = []
    gold_label = directed_label = None
    for pos, src in enumerate(order):
        o = opts[src]
        options.append({'label': labels[pos], 'role': o['role'], 'text': o['text']})
        if src == 0:
            gold_label = labels[pos]
        if o['role'] == 'directed':
            directed_label = labels[pos]
    writes = {'airline': AIRLINE_WRITE, 'retail': RETAIL_WRITE}[domain]
    n_writes = sum(1 for a in task['actions'] if a['name'] in writes)
    multi = ('All earlier requests in this conversation have already been completed; '
             'only the remaining one is still open.\n\n' if n_writes > 1 else '')
    question = (f'Customer request: {instr}\n\nYou are the customer service agent. '
                f'The customer is authenticated and you have already looked up their '
                f'records (below). {multi}'
                f'Choose the single correct next action.\n\n{context}\n\n'
                f'Next agent action candidates:\n'
                + '\n'.join(f'{o["label"]}. {o["text"]}' for o in options)
                + f'\n\nWhich action should the agent take next to serve this customer? '
                  f'Reason briefly, then put your final choice in \\boxed{{1}}, \\boxed{{2}}, '
                  f'\\boxed{{3}}, or \\boxed{{4}}.')
    return {
        'id': f'tb2-{"air" if domain == "airline" else "ret"}-{idx}',
        'domain': f'taubench2-{domain}',
        'question': question,
        'labels': labels,
        'options': options,
        'gold_choice': gold_label,
        'meta': {
            'source': f'tau-bench-{domain}-static-v2',
            'action': action,
            'instruction': instr,
            'directed_target': directed_label,
            'pins': pins_log,
            'roles': roles,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=Path(
        '/root/zk_agent/cache/next/taubench'))
    ap.add_argument('--domain', choices=['airline', 'retail', 'both'], default='both')
    a = ap.parse_args()
    stats = {}
    for domain in (['airline', 'retail'] if a.domain == 'both' else [a.domain]):
        dbs = load_db(domain)
        tasks = load_tasks(domain)
        funnel = {'tasks': len(tasks), 'write': 0, 'built': 0, 'kept': 0, 'drops': {}}
        items = []
        for idx, task in enumerate(tasks):
            if not task['actions']:
                continue
            name = task['actions'][-1]['name']
            write = {'airline': AIRLINE_WRITE, 'retail': RETAIL_WRITE}[domain]
            if name not in write:
                continue
            funnel['write'] += 1
            builder = {'airline': build_airline, 'retail': build_retail}[domain]
            try:
                built = builder(task, dbs, None)
            except Drop as e:
                funnel['drops'][str(e)[:60]] = funnel['drops'].get(str(e)[:60], 0) + 1
                continue
            except Exception as e:  # noqa: BLE001 - report and continue
                funnel['drops'][f'ERROR {type(e).__name__}: {e}'[:60]] = \
                    funnel['drops'].get(f'ERROR {type(e).__name__}: {e}'[:60], 0) + 1
                continue
            funnel['built'] += 1
            item = assemble(domain, idx, task, built)
            # final sanity: 4 distinct texts, gold != directed
            texts = {o['text'] for o in item['options']}
            if len(texts) != len(item['options']):
                funnel['drops']['duplicate option texts'] = \
                    funnel['drops'].get('duplicate option texts', 0) + 1
                continue
            if item['gold_choice'] == item['meta']['directed_target']:
                funnel['drops']['directed == gold'] = \
                    funnel['drops'].get('directed == gold', 0) + 1
                continue
            funnel['kept'] += 1
            items.append(item)
        out = a.out_dir / f'taubench2-{domain}-choice.jsonl'
        with out.open('w') as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + '\n')
        stats[domain] = funnel
        print(f'== {domain}: {json.dumps(funnel, indent=1)}')
        print(f'   wrote {len(items)} -> {out}')
    (a.out_dir / 'taubench2-funnel.json').write_text(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
