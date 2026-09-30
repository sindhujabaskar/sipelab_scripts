#!/usr/bin/env python3
"""
=============================================================================
 INJECTION / STIMULUS RANDOMIZATION  -  8 animals x 4 rounds x 5 days
=============================================================================
 Design: each animal completes four 5-day rounds. Each round pairs ONE visual
 stimulus with a 5-day injection sequence:
       Day 1 = baseline, no injection   (fixed)
       Day 2-4 = a permutation of {Saline, Low, High}   (randomized)
       Day 5 = saline control           (fixed)

 The script draws a design by REJECTION SAMPLING from the space of designs
 satisfying five constraints, then asserts all five before writing output.
 If any assertion fails the script dies rather than emitting a schedule.
=============================================================================
"""
import itertools, collections, csv, os, sys, json, time, hashlib, pathlib
import numpy as np

# =============================================================================
# STEP 0 - SEED.  Drawn from the OS entropy pool, not chosen by hand, then
#          printed and written to the output so the draw is reproducible.
# =============================================================================
SEED = int.from_bytes(os.urandom(4), "big")     # override here to reproduce a past draw
rng  = np.random.default_rng(SEED)

# =============================================================================
# STEP 1 - FIXED SCAFFOLDING (nothing random here)
# =============================================================================
T      = ["Saline", "Low", "High"]
ORDERS = [tuple(p) for p in itertools.permutations(T)]   # all 3! = 6 injection orders
STIM   = ["Stim_A", "Stim_B", "Stim_C", "Stim_D"]

# Williams square for 4 treatments: each symbol once per row AND once per column,
# and all 12 ordered pairs occur exactly once (first-order carryover balance).
WILLIAMS = [[0, 1, 3, 2],
            [1, 2, 0, 3],
            [2, 3, 1, 0],
            [3, 0, 2, 1]]

# Which pairs of "extra" orders keep a round balanced?  A round holds 8 animals
# but only 6 distinct orders exist, so 2 orders must repeat.  The 6 distinct
# orders give 2/2/2 on every day; the 2 extras push two cells to 3.  For the day
# to land on 3/3/2 rather than 4/2/2, the two extras must differ on THAT day.
# Requiring it on all three days means the two extras differ in every position.
VALID_EXTRA_PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)
                     if all(ORDERS[i][d] != ORDERS[j][d] for d in range(3))]

# =============================================================================
# CONSTRAINTS  (checked during sampling, re-asserted at the end)
#   C1  each animal gets 4 DISTINCT injection orders across its 4 rounds
#   C2  each animal sees each of the 4 stimuli exactly once
#   C3  each stimulus occurs in each round exactly twice
#   C4  every ROUND is 3/3/2 across {S,L,H} on each of D2, D3, D4
#   C5  every STIMULUS group is 3/3/2 across {S,L,H} on each of D2, D3, D4
# 3/3/2 is the floor: 8 animals cannot split evenly into 3 doses.
# =============================================================================
def is_332(orders):
    return all(sorted(collections.Counter(o[d] for o in orders).values()) == [2, 3, 3]
               for d in range(3))

# =============================================================================
# STEP 2 - one candidate draw
# =============================================================================
def random_matching(block, used):
    """Deal this round's 8 orders to the 8 animals so no animal repeats an order
       it already had. Randomized backtracking; returns None if it paints itself
       into a corner (caller just redraws)."""
    order_of_animals = list(rng.permutation(8))
    assign = [None] * 8
    def rec(k, free_slots):
        if k == 8:
            return True
        a = order_of_animals[k]
        cands = [s for s in free_slots if block[s] not in used[a]]
        rng.shuffle(cands)
        for s in cands:
            assign[a] = block[s]
            if rec(k + 1, [f for f in free_slots if f != s]):
                return True
            assign[a] = None
        return False
    return assign if rec(0, list(range(8))) else None

def draw():
    # 2a. animals -> Williams rows (two animals per row). Satisfies C2 and C3 by
    #     construction, whatever the draw.
    rows = list(rng.permutation([0, 0, 1, 1, 2, 2, 3, 3]))
    stim = {(a, r): WILLIAMS[rows[a]][r] for a in range(8) for r in range(4)}

    # 2b. per round: pick the repeated-order pair from VALID_EXTRA_PAIRS only.
    #     Satisfies C4 by construction.
    used, rounds = [set() for _ in range(8)], []
    for r in range(4):
        i, j = VALID_EXTRA_PAIRS[rng.integers(len(VALID_EXTRA_PAIRS))]
        block = list(ORDERS) + [ORDERS[i], ORDERS[j]]
        # 2c. deal to animals under the no-repeat rule. Satisfies C1 or fails.
        m = random_matching(block, used)
        if m is None:
            return None
        for a in range(8):
            used[a].add(m[a])
        rounds.append(m)

    # 2d. C5 is the only constraint left to chance -> REJECT if it misses.
    by_stim = collections.defaultdict(list)
    for a in range(8):
        for r in range(4):
            by_stim[stim[(a, r)]].append(rounds[r][a])
    if not all(is_332(by_stim[s]) for s in range(4)):
        return None
    return rounds, stim, rows, by_stim

# =============================================================================
# STEP 3 - draw until accepted
# =============================================================================
t0, n_draws, result = time.time(), 0, None
while result is None:
    n_draws += 1
    result = draw()
    if n_draws > 2_000_000:
        sys.exit("ERROR: no design accepted; constraints may be infeasible.")
rounds, stim, rows, by_stim = result
elapsed = time.time() - t0

# =============================================================================
# STEP 4 - VERIFY. Independent re-check of all five constraints.
# =============================================================================
assert all(len({rounds[r][a] for r in range(4)}) == 4 for a in range(8)), "C1"
assert all(len({stim[(a, r)] for r in range(4)}) == 4 for a in range(8)), "C2"
_sc = collections.Counter((stim[(a, r)], r) for a in range(8) for r in range(4))
assert all(_sc[(s, r)] == 2 for s in range(4) for r in range(4)), "C3"
for r in range(4):
    assert is_332([rounds[r][a] for a in range(8)]), f"C4 round {r+1}"
for s in range(4):
    assert is_332(by_stim[s]), f"C5 {STIM[s]}"

# =============================================================================
# STEP 5 - REPORT
# =============================================================================
print("=" * 78)
print(f"SEED = {SEED}    (from os.urandom; set SEED to this value to reproduce)")
print(f"accepted after {n_draws:,} draws in {elapsed:.1f}s")
print("=" * 78)

print("\nBalanced extra-pairs available at step 2b (6 of the 15 possible pairs):")
for i, j in VALID_EXTRA_PAIRS:
    print("   ", "/".join(x[0] for x in ORDERS[i]), "+", "/".join(x[0] for x in ORDERS[j]))

print("\nAnimal -> Williams row (step 2a):", {f"M{a+1:02d}": rows[a] for a in range(8)})

print("\nSCHEDULE   (stimulus : D2/D3/D4;  D1=baseline, D5=saline throughout)")
print(f"{'Animal':7s}" + "".join(f"{'Round '+str(r+1):24s}" for r in range(4)))
for a in range(8):
    line = f"M{a+1:02d}    "
    for r in range(4):
        o = rounds[r][a]
        line += f"{STIM[stim[(a,r)]]}: {'/'.join(x[0] for x in o):<6s}".ljust(24)
    print(line)

def fmt(orders):
    return " | ".join("D%d " % (d + 2) + "/".join(
        f"{t[0]}{collections.Counter(o[d] for o in orders)[t]}" for t in T) for d in range(3))

print("\nC4 - balance within each ROUND (n=8):")
for r in range(4):
    print(f"   Round {r+1}: {fmt([rounds[r][a] for a in range(8)])}")
print("\nC5 - balance within each STIMULUS (n=8):")
for s in range(4):
    print(f"   {STIM[s]}: {fmt(by_stim[s])}")
print("\nC3 - stimulus x round occupancy (every cell must be 2):")
for s in range(4):
    print(f"   {STIM[s]}: " + "  ".join(f"R{r+1}={_sc[(s,r)]}" for r in range(4)))
print("\nC1/C2 - per animal: 4 distinct orders and 4 distinct stimuli:",
      all(len({rounds[r][a] for r in range(4)}) == 4 and
          len({stim[(a, r)] for r in range(4)}) == 4 for a in range(8)))

# =============================================================================
# STEP 6 - WRITE
# =============================================================================
out = pathlib.Path("/home/claude")
csv_path = out / f"injection_schedule_seed{SEED}.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Seed", "Round", "Animal", "Stimulus",
                "Day1", "Day2", "Day3", "Day4", "Day5"])
    for r in range(4):
        for a in range(8):
            o = rounds[r][a]
            w.writerow([SEED, r + 1, f"M{a+1:02d}", STIM[stim[(a, r)]],
                        "Baseline (no inj)", o[0], o[1], o[2], "Saline"])
rows_txt = "".join(f"{r+1},M{a+1:02d},{STIM[stim[(a,r)]]},{'/'.join(rounds[r][a])}\n"
                   for r in range(4) for a in range(8))
print(f"\nwrote {csv_path}")
print(f"SHA256 of schedule body: {hashlib.sha256(rows_txt.encode()).hexdigest()[:16]}")