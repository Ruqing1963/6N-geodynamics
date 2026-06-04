#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute the prime-triplet (6N-1, 6N+1, 6N+5) count at the deep shells from
scratch (segmented value sieve, memory-light), cross-check against the
published Paper XV omega-band sums, and report the depth-flow slope.

  shell S_k :  6N in [10^(k-1), 10^k) ,  centre N in [10^(k-1)/6, 10^k/6)
  rho_triplet = (#triplets) / (#centres)
  ell         = ln ln(10^k)
  target      : d ln rho_triplet / d ell -> -3   (with a +3/lnX analytic tail)

Usage (PowerShell):
    python triplet_deep.py                 # S9 and S10
    $env:SHELLS="9,10,11"; python triplet_deep.py
    $env:SEG=20000000; python triplet_deep.py    # larger blocks, more memory

S10 takes a few minutes; S11 (if you have the headroom) appreciably longer.
Requires: numpy.
"""
import os, math, time
import numpy as np

# published Paper XV totals (sum over omega bands) for cross-check
PUBLISHED = {9: 323907, 10: 2333839}
SHELLS = [int(x) for x in os.environ.get("SHELLS", "9,10").split(",")]
SEG    = int(os.environ.get("SEG", 10_000_000))
TRIP   = ((0, -1), (0, +1), (1, -1))     # (6N-1, 6N+1, 6N+5) wing geometry

def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0].astype(np.int64)

def triplet_shell(k):
    """Count N in shell S_k with 6N-1, 6N+1, 6N+5 all prime. Returns (count, centres)."""
    base = primes_upto(int(math.isqrt(10**k)) + 2)
    Nlo = 10**(k-1)//6 + 1
    Nhi = (10**k - 1)//6
    cnt = 0
    ns  = Nlo
    while ns <= Nhi:
        ne  = min(ns + SEG, Nhi + 1)            # N in [ns, ne)
        vlo = 6*ns - 1
        vhi = 6*(ne - 1) + 5
        comp = np.zeros(vhi - vlo + 1, bool)
        for p in base:
            if p*p > vhi: break
            st = max(p*p, ((vlo + p - 1)//p)*p)
            comp[st - vlo::p] = True
        isp = ~comp
        N = np.arange(ns, ne, dtype=np.int64)
        ok = np.ones(ne - ns, bool)
        for off, _s in TRIP:
            ok &= isp[6*N + (6*off + _s) - vlo]   # 6N-1 -> off0 s-1 ; etc.
        cnt += int(np.count_nonzero(ok))
        ns = ne
    return cnt, (Nhi - Nlo + 1)

def main():
    print("== triplet deep recompute ==  shells", SHELLS, " SEG", SEG, "\n")
    rows = []
    for k in SHELLS:
        t = time.time()
        c, ncen = triplet_shell(k)
        rho = c / ncen
        ell = math.log(math.log(10.0**k))
        rows.append((k, c, ncen, rho, ell))
        line = (f"S{k}: triplet={c}  centres={ncen}  "
                f"rho={rho:.8f}  ln_rho={math.log(rho):.5f}  ell={ell:.5f}  "
                f"[{time.time()-t:.0f}s]")
        if k in PUBLISHED:
            d = abs(c - PUBLISHED[k]) / PUBLISHED[k]
            line += f"  | published {PUBLISHED[k]} (diff {d:.4%})"
        print(line)
    print()
    for (k0, *_), (k1, *_2) in zip(rows, rows[1:]):
        r0 = next(r for r in rows if r[0] == k0)
        r1 = next(r for r in rows if r[0] == k1)
        slope = (math.log(r1[3]) - math.log(r0[3])) / (r1[4] - r0[4])
        print(f"d ln rho/d ell  S{k0}->S{k1} = {slope:.3f}   (target -3, "
              f"tail = {slope+3:+.3f})")

if __name__ == "__main__":
    main()
