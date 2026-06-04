# 6N Arithmetic Geodynamics: Tuple Size Determines Stratum Decay Rate (Part XVII)

Making the decadic shells continuous turns the static cores of the 6N programme
into a **flow**. In the proper-depth coordinate `ell = ln ln X`, the twin-centre
density and the mean factor count linearise, and the decay rate of a prime
`m`-tuple is exactly its size:

```
    d ln rho_m / d ell  ->  -m .
```

The twin (m=2) and the triplet (6N-1, 6N+1, 6N+5) (m=3) confirm it. The clean,
scale-free statement is their **ratio**, in which the shared clock distortion and
the shared logarithmic tail cancel.

**Why ln ln X.** Hardy-Littlewood gives `rho_m(X) ~ A_m / (ln X)^m`, so
`ln rho_m = const - m·ln ln X + O(1/ln X)`. The depth in which the flow is linear
is therefore `ell = ln ln X`. In X or in `tau = ln X` the CRT "shield" (the
dead(q) sieve, q-2 admissible residues) appears to weaken with depth — pure
coordinate distortion; in `ell` its strangulation is **constant**, of magnitude m.

**The flow equations.** With Hardy-Ramanujan/Mertens `omega_bar ~ ln ln N + B`,

```
    d rho_m / d ell = -m · rho_m ,      d omega_bar_{>3} / d ell = 1 ,
```

so on the (omega_bar, ln rho) phase plane the slope is `-m / (d omega_bar/d ell)`.
The clock factor `d omega_bar/d ell = 1 + eps(X)` (a convergent Mertens-B approach
plus the Part XII enrichment shift -> 0.2604) is **shared** across patterns, so it
cancels in the ratio.

**Results (from-scratch).** The S10 triplet recompute reproduces the Part XV total
(2,333,839) to **0.0000%**, so both flow lines rest on sieve primitives.

| | twin (m=2) | triplet (m=3) | ratio |
|---|---:|---:|---:|
| ln rho at S9 | -3.9176 | -6.1379 | |
| ln rho at S10 | -4.1364 | -6.4657 | |
| deep slope d ln rho/d ell (S9->S10) | -2.072 | -3.111 | **1.5011** |
| extrapolated intercept (1/ln X -> 0) | -2.085 | -3.065 | |
| asymptote (target) | -2 | -3 | 1.5000 |

> **The invariant.** Neither absolute slope is its integer at this depth — each
> carries a shrinking tail (+0.072, +0.111) and is read against an omega_bar-clock
> running ~15% fast. Both effects are shared, so the **ratio** reads off the
> structural integers directly: `3.111/2.072 = 1.5011 ~ 3/2`, to 0.1%.
>
> **Honest asymptotes.** The intercepts reach -2.085 and -3.065, *consistent with*
> -2 and -3; the residual (~0.07-0.09) is the part of the tail not linear in
> 1/ln X (higher-order log terms a two-point-per-shell fit cannot capture). We do
> not claim the extrapolation lands exactly on the integer.
>
> **First moments only.** The flow governs the means. A second-order transport
> equation for the *variance* of omega_{>3} over twin centres is the open
> Erdos-Kac problem of Part XII (the naive Bernoulli variance diverges; the true
> variance is finite by anti-correlation and the sqrt(6N) cutoff). The drift (+1)
> is established here; the diffusion term is not.
>
> **Attribution.** The `(ln X)^{-m}` law is the classical Hardy-Littlewood
> k-tuple heuristic; `ln ln X + B` is Hardy-Ramanujan/Mertens; rho is the Part I
> scale and 0.2604 the Part XII shift. The contribution is the proper-depth
> coordinate, the flow system, and the scale-free ratio invariant. No statement is
> made about the infinitude of any constellation.

**A falsifiable prediction (m=4).** The prime quadruplet
(6N-1, 6N+1, 6N+5, 6N+7) = two complete twin centres N and N+1 must give slope
**-4** and ratio **2:1** against the twin (and 4:3 against the triplet), each to
the precision at which the invariant holds. A measured intercept outside
[-4.2, -3.8], or a twin ratio outside 2 +/- 0.05, would falsify the tuple-size
law. We deliberately do not run it: a one-parameter law should predict before it
measures.

Part I: doi:10.5281/zenodo.20470367 · XII: doi:10.5281/zenodo.20528446 ·
XV: https://github.com/Ruqing1963/6N-prime-triplet

---

## Layout

```
.
├── README.md
├── LICENSE                 (MIT)
├── CITATION.cff
├── data/
│   ├── geodynamics_flow.csv   shell, ell, rho_twin, ln_rho_twin, omega_twin,
│   │                          rho_twin_HLavg, triplet_count, rho_triplet, ln_rho_triplet
│   └── triplet_counts.csv     shell, triplet_count, centres, rho_triplet, source
├── code/
│   ├── 6n_geodynamics.py    proper-depth reduction: tail peeling, omega_bar
│   │                        decomposition, twin/triplet phase diagram. Triplets
│   │                        S2..MAXK recomputed (segmented sieve); S9/S10
│   │                        cross-checked against Part XV. Emits the figure + CSV.
│   └── triplet_deep.py      from-scratch triplet recompute at the deep shells
│                            (default S9, S10), with cross-check and slope.
├── figures/                fig_paper17_geodynamics.{pdf,png}
└── paper/                  Chen_6N_Paper17.{tex,pdf} + figure
```

## Reproducing

Requirements: Python 3.8+, `numpy`, `matplotlib`.

```bash
pip install numpy matplotlib

# 1. Full analysis + 4-panel phase diagram. Default MAXK=8 (fast, runs anywhere).
#    Set MAXK=10 to recompute the triplet line to S10 from scratch (a few minutes).
python code/6n_geodynamics.py
MAXK=10 SEG=20000000 python code/6n_geodynamics.py      # full

# 2. From-scratch deep recompute (the bedrock cross-check). Default S9, S10.
python code/triplet_deep.py
SHELLS=10 python code/triplet_deep.py                    # S10 only (~a few min)
```

On Windows PowerShell, set environment variables separately, e.g.
`$env:MAXK=10; $env:SEG=20000000; python code/6n_geodynamics.py`.

### Conventions (same as Parts I-XVI)

- Skeleton 6N±1; N the centre; a twin centre has both wings 6N±1 prime.
- omega_{>3}(N) = number of distinct prime factors >3 of the centre N.
- rho = #{m-tuples} / #{centres N} per shell; ell = ln ln X (proper depth).
- Shells S_k : 6N in [10^(k-1), 10^k); S10 twin count 23,988,173 matches Part I.
- Triplet (6N-1, 6N+1, 6N+5) at (offset, wing) = (0,-1),(0,+1),(1,-1).
- Engine: segmented value sieve; every count recomputed from primitives.

## License

MIT — see `LICENSE`.
