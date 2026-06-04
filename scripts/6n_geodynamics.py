#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6N geodynamics: from static cores to a stratum-evolution flow.
================================================================
Turns the per-shell observables of the 6N programme (twin-centre density rho,
mean factor count omega-bar) into continuous flow lines in the proper-depth
coordinate  ell = ln ln X , and tests the tuple-size law

        d ln rho_m / d ell  ->  -m          (m = pattern size)

for the twin (m=2) and the triplet (6N-1, 6N+1, 6N+5) (m=3).

Four outputs, matching the design brief:
  (1) Tail peeling: local slope d ln rho / d ell vs 1/ln X, extrapolated to the
      asymptote -m, with the shell-averaged Hardy-Littlewood prediction
      rho_avg(k) = 2 C2 * [Li2(10^k) - Li2(10^(k-1))] / (10^k - 10^(k-1)).
  (2) omega-bar drift decomposition: ln ln Y main term + Mertens offset
      (B1 - 1/2 - 1/3) + enrichment shift -> 0.2604 (Paper XII).
  (3) Depth phase diagram (ln rho vs ell): clean -2 and -3 asymptotes.
  (4) omega-bar phase diagram (ln rho vs omega-bar, the requested plot):
      measured slopes ~ -m / (d omega-bar/d ell); the twin:triplet ratio is
      exactly 3:2 on this axis.

PROVENANCE. Twin counts and mean-omega are the published Paper I / XII tables
(table1_shell_counts, table4_mean_omega, omega_summary). C2, the singular
series, and the (ln X)^{-2}/(ln X)^{-3} laws are classical Hardy-Littlewood;
ln ln X + B is Hardy-Ramanujan/Mertens; the enrichment shift 0.2604 is Paper
XII. The contribution here is the depth-coordinate reduction and the flow fit.
No claim is made about the infinitude of any pattern. First moments only: the
omega variance (Erdos-Kac on twin centres) is left as the open problem it is.

Triplet counts S2..MAXK are recomputed from scratch (segmented value sieve,
memory-light, scales to S10); S9/S10 are cross-checked against the published
Paper XV omega-band sums.

Usage:  MAXK=8 python3 6n_geodynamics.py        # fast, runs anywhere
        MAXK=10 SEG=20000000 python3 6n_geodynamics.py   # full, more memory
Requires: numpy, matplotlib.
"""
import os, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
C2       = 0.6601618158468696   # Hardy-Littlewood twin-prime constant
TWO_C2   = 2.0 * C2
B1       = 0.2614972128476428   # Mertens constant  (mean omega(n) ~ ln ln n + B1)
B_OMEGA  = B1 - 0.5 - 1.0/3.0   # offset for omega_{>3} (drop primes 2 and 3)
SHIFT_INF = 0.2604              # Paper XII enrichment shift  sum (1/(q-2) - 1/q)
MAXK     = int(os.environ.get("MAXK", 8))
SEG      = int(os.environ.get("SEG", 10_000_000))
OUT      = os.environ.get("OUT", ".")

# ----------------------------------------------------------------------------
# Published per-shell observables (authoritative; from the Paper I/XII tables)
#   k : (twin_pairs, total_nodes, mean_omega_twin)
# ----------------------------------------------------------------------------
TWIN = {
    1: (1,           1,            0.000),
    2: (6,           15,           0.500),
    3: (27,          150,          1.074),
    4: (170,         1500,         1.565),
    5: (1019,        15000,        1.876),
    6: (6945,        150000,       2.082),
    7: (50811,       1500000,      2.282),
    8: (381332,      15000000,     2.447),
    9: (2984194,     150000000,    2.582),
    10:(23988173,    1500000000,   2.703),
}
ORD_MEAN_MEASURED = {9: 2.3599, 10: 2.4757}                 # Paper XII, cross-check
TRIPLET_PUBLISHED = {9: 323907, 10: 2333839}               # Paper XV omega-band sums

def total_nodes(k):   return TWIN[k][1]
def rho_twin(k):      return TWIN[k][0] / TWIN[k][1]
def omega_twin(k):    return TWIN[k][2]
def scale_wing(k):    return 10.0**k          # wing value ~ 6N up to 10^k
def scale_centre(k):  return 10.0**k / 6.0    # centre N up to 10^k / 6
def ell(k):           return math.log(math.log(scale_wing(k)))   # proper depth

# ----------------------------------------------------------------------------
# Li2(x) = \int_2^x dt / (ln t)^2  (smooth integrand; log-grid Simpson)
# ----------------------------------------------------------------------------
def Li2(x, n=200000):
    if x <= 2.0:
        return 0.0
    u0, u1 = math.log(2.0), math.log(x)        # substitute t = e^u, dt = e^u du
    if n % 2: n += 1
    u = np.linspace(u0, u1, n + 1)
    f = np.exp(u) / (u * u)                     # integrand in u
    h = (u1 - u0) / n
    return float(h / 3.0 * (f[0] + f[-1] + 4.0*f[1:-1:2].sum() + 2.0*f[2:-1:2].sum()))

def rho_twin_HL_avg(k):
    """Shell-averaged HL twin-centre density (the analytic prediction).
    Factor 6 = the skeleton: one centre per 6 integers, so the per-centre twin
    rate is 6x the raw integer twin-prime density 2 C2 (ln t)^{-2}."""
    lo, hi = 10.0**(k-1), 10.0**k
    return 6.0 * TWO_C2 * (Li2(hi) - Li2(lo)) / (hi - lo)

def rho_twin_HL_single(k):
    """Single-point (upper-edge) leading term; its slope in ell is exactly -2."""
    return 6.0 * TWO_C2 / (math.log(scale_wing(k)))**2

# ----------------------------------------------------------------------------
# Triplet (6N-1, 6N+1, 6N+5) counts via a segmented value sieve
# ----------------------------------------------------------------------------
def _primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0].astype(np.int64)

def triplet_counts(maxk):
    base = _primes_upto(int(math.isqrt(10**maxk)) + 2)
    out = {}
    for k in range(2, maxk + 1):
        Nlo = 10**(k-1)//6 + 1
        Nhi = (10**k - 1)//6
        cnt = 0
        nstart = Nlo
        while nstart <= Nhi:
            nend = min(nstart + SEG, Nhi + 1)          # N in [nstart, nend)
            vlo  = 6*nstart - 1
            vhi  = 6*(nend - 1) + 5
            comp = np.zeros(vhi - vlo + 1, bool)
            for p in base:
                if p*p > vhi: break
                st = max(p*p, ((vlo + p - 1)//p)*p)
                comp[st - vlo::p] = True
            isp = ~comp
            N = np.arange(nstart, nend, dtype=np.int64)
            cnt += int(np.count_nonzero(
                isp[6*N - 1 - vlo] & isp[6*N + 1 - vlo] & isp[6*N + 5 - vlo]))
            nstart = nend
        out[k] = cnt
    return out

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def local_slope(xs, ys):
    """Backward local slopes dy/dx between consecutive points."""
    return [(ys[i]-ys[i-1])/(xs[i]-xs[i-1]) for i in range(1, len(xs))]

def main():
    t0 = time.time()
    print(f"== 6N geodynamics ==  MAXK={MAXK}\n")

    ks = list(range(2, MAXK + 1))

    # --- triplet line (compute + merge published S9/S10) ---
    print("computing triplet counts ...")
    tc = triplet_counts(MAXK)
    for k in (9, 10):
        if k <= MAXK and k in TRIPLET_PUBLISHED:
            d = abs(tc[k] - TRIPLET_PUBLISHED[k]) / TRIPLET_PUBLISHED[k]
            print(f"  S{k} triplet recomputed {tc[k]} vs published "
                  f"{TRIPLET_PUBLISHED[k]}  (diff {d:.2%})")
    tri = dict(tc)
    for k in (9, 10):
        if k > MAXK:                       # fall back to published if not reached
            tri[k] = TRIPLET_PUBLISHED[k]
    tri_ks = sorted(tri)

    # ---------- arrays ----------
    kk        = [k for k in range(2, 11) if k in TWIN]
    ell_t     = [ell(k) for k in kk]
    lnrho_t   = [math.log(rho_twin(k)) for k in kk]
    om_t      = [omega_twin(k) for k in kk]

    ell_tri   = [ell(k) for k in tri_ks]
    lnrho_tri = [math.log(tri[k]/total_nodes(k)) for k in tri_ks]
    # share the omega clock: use the twin centre's mean-omega at the same shell
    om_tri    = [omega_twin(k) for k in tri_ks]

    # ---------- (1) tail peeling ----------
    print("\n(1) TAIL PEELING  (twin: slope d ln rho/d ell -> -2)")
    print("  shell  1/lnX     slope_meas   rho_meas     rho_HLavg    rel.err")
    inv_lnX, sl_meas = [], []
    sl_twin = local_slope(ell_t, lnrho_t)
    for i, k in enumerate(kk):
        rm  = rho_twin(k); ra = rho_twin_HL_avg(k)
        sl  = sl_twin[i-1] if i >= 1 else float('nan')
        if i >= 1:
            inv_lnX.append(1.0/math.log(scale_wing(k))); sl_meas.append(sl)
        print(f"  S{k:<4} {1/math.log(scale_wing(k)):.5f}  {sl:9.3f}   "
              f"{rm:.7f}   {ra:.7f}   {(rm-ra)/ra:+.3%}")
    # extrapolate slope -> intercept at 1/lnX = 0
    A = np.polyfit(inv_lnX[-5:], sl_meas[-5:], 1)
    print(f"  twin slope extrapolated to 1/lnX=0 :  {A[1]:.3f}   (target -2)")

    sl_tri = local_slope(ell_tri, lnrho_tri)
    inv_tri = [1.0/math.log(scale_wing(tri_ks[i])) for i in range(1, len(tri_ks))]
    At = np.polyfit(inv_tri[-5:], sl_tri[-5:], 1)
    print(f"  triplet slope extrapolated to 1/lnX=0 : {At[1]:.3f}   (target -3)")

    # ---------- (2) omega-bar decomposition ----------
    print("\n(2) OMEGA-BAR DRIFT DECOMPOSITION")
    print("  shell  main=lnlnY   +B_off    =ord_pred   ord_meas   shift=tw-ordpred")
    for k in kk:
        main = math.log(math.log(scale_centre(k)))
        ordp = main + B_OMEGA
        ordm = ORD_MEAN_MEASURED.get(k, float('nan'))
        shift = omega_twin(k) - ordp
        print(f"  S{k:<4} {main:8.4f}   {B_OMEGA:+.4f}  {ordp:8.4f}   "
              f"{ordm:8.4f}   {shift:.4f}")
    print(f"  enrichment shift  ->  {SHIFT_INF}  (Paper XII asymptote)")
    dom = local_slope(ell_t, om_t)
    print(f"  d omega-bar / d ell  (deep)  = {dom[-1]:.3f}   (target 1)")

    # ---------- (3)+(4) slopes on the two axes ----------
    sa_t  = (lnrho_t[-1]-lnrho_t[-3])/(ell_t[-1]-ell_t[-3])
    sa_tr = (lnrho_tri[-1]-lnrho_tri[-3])/(ell_tri[-1]-ell_tri[-3])
    so_t  = (lnrho_t[-1]-lnrho_t[-3])/(om_t[-1]-om_t[-3])
    so_tr = (lnrho_tri[-1]-lnrho_tri[-3])/(om_tri[-1]-om_tri[-3])
    print("\n(3)/(4) DEEP SLOPES")
    print(f"  vs ell  : twin {sa_t:.3f} (->-2)   triplet {sa_tr:.3f} (->-3)")
    print(f"  vs omega: twin {so_t:.3f}          triplet {so_tr:.3f}")
    print(f"  omega-axis slope ratio triplet/twin = {so_tr/so_t:.3f}  (exact 3/2)")

    # ---------- figure ----------
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 11))
    fig.suptitle("6N arithmetic geodynamics: tuple size = stratum decay rate",
                 fontsize=15, fontweight="bold")
    cT, cR = "#1f4e79", "#b4341f"
    sh = np.array([k for k in kk])

    # (A) depth phase diagram
    a = ax[0,0]
    a.scatter(ell_t, lnrho_t, c=cT, s=46, zorder=3, label="twin (m=2)")
    a.scatter(ell_tri, lnrho_tri, c=cR, marker="s", s=42, zorder=3,
              label="triplet (m=3)")
    for x,y,k in zip(ell_t, lnrho_t, kk): a.annotate(f"S{k}",(x,y),
        textcoords="offset points", xytext=(5,3), fontsize=7, color=cT)
    xa = np.array([min(ell_t)-0.05, max(ell_t)+0.05])
    a.plot(xa, lnrho_t[-1]-2*(xa-ell_t[-1]), "--", c=cT, lw=1.2,
           label="asymptote slope -2")
    a.plot(xa, lnrho_tri[-1]-3*(xa-ell_tri[-1]), "--", c=cR, lw=1.2,
           label="asymptote slope -3")
    a.set_xlabel(r"proper depth  $\ell=\ln\ln X$"); a.set_ylabel(r"$\ln\rho$")
    a.set_title("(A) depth flow: slope $\\to -m$"); a.legend(fontsize=8); a.grid(alpha=.3)

    # (B) requested omega-bar phase diagram
    b = ax[0,1]
    b.scatter(om_t, lnrho_t, c=cT, s=46, zorder=3, label="twin")
    b.scatter(om_tri, lnrho_tri, c=cR, marker="s", s=42, zorder=3, label="triplet")
    xb = np.array([min(om_t)-0.05, max(om_t)+0.05])
    b.plot(xb, lnrho_t[-1]+so_t*(xb-om_t[-1]), "--", c=cT, lw=1.2,
           label=f"slope {so_t:.2f}")
    b.plot(xb, lnrho_tri[-1]+so_tr*(xb-om_tri[-1]), "--", c=cR, lw=1.2,
           label=f"slope {so_tr:.2f}")
    b.set_xlabel(r"$\bar\omega$  (geological clock)"); b.set_ylabel(r"$\ln\rho$")
    b.set_title(f"(B) $\\bar\\omega$ phase plane: ratio = {so_tr/so_t:.2f} (= 3/2)")
    b.legend(fontsize=8); b.grid(alpha=.3)

    # (C) tail peeling
    c = ax[1,0]
    c.scatter(inv_lnX, sl_meas, c=cT, s=40, label="twin local slope")
    c.scatter(inv_tri, sl_tri, c=cR, marker="s", s=38, label="triplet local slope")
    xc = np.array([0, max(inv_lnX)*1.05])
    c.plot(xc, A[1]+A[0]*xc, "--", c=cT, lw=1)
    c.plot(xc, At[1]+At[0]*xc, "--", c=cR, lw=1)
    c.axhline(-2, color=cT, ls=":", lw=1); c.axhline(-3, color=cR, ls=":", lw=1)
    c.scatter([0,0],[A[1],At[1]], facecolors="none",
              edgecolors=["k","k"], s=80, zorder=4)
    c.set_xlabel(r"$1/\ln X$  (analytic tail)")
    c.set_ylabel(r"$d\ln\rho/d\ell$")
    c.set_title("(C) tail dissolves: slope $\\to -2,-3$ as $1/\\ln X\\to0$")
    c.legend(fontsize=8); c.grid(alpha=.3)

    # (D) omega decomposition
    d = ax[1,1]
    mains = [math.log(math.log(scale_centre(k))) for k in kk]
    ordp  = [m + B_OMEGA for m in mains]
    shifts= [omega_twin(k) - (m + B_OMEGA) for k,m in zip(kk, mains)]
    d.plot(sh, om_t, "o-", c=cT, label=r"$\bar\omega_{\rm twin}$ (measured)")
    d.plot(sh, ordp, "s--", c="#888", label=r"$\ln\ln Y + B$ (ordinary, theory)")
    d.plot(sh, [ORD_MEAN_MEASURED.get(k,np.nan) for k in kk], "x",
           c="k", ms=8, label="ordinary (Paper XII)")
    d2 = d.twinx()
    d2.plot(sh, shifts, "^-", c=cR, label="enrichment shift")
    d2.axhline(SHIFT_INF, color=cR, ls=":", lw=1)
    d2.set_ylabel("shift (twin - ordinary)", color=cR)
    d2.set_ylim(0, 0.5); d2.tick_params(axis="y", colors=cR)
    d.set_xlabel("shell k"); d.set_ylabel(r"$\bar\omega_{>3}$")
    d.set_title("(D) $\\bar\\omega$ drift: main + Mertens $B$ + shift $\\to 0.2604$")
    d.legend(fontsize=7, loc="upper left"); d.grid(alpha=.3)

    fig.tight_layout(rect=[0,0,1,0.97])
    png = os.path.join(OUT, "fig_6n_geodynamics.png")
    pdf = os.path.join(OUT, "fig_6n_geodynamics.pdf")
    fig.savefig(png, dpi=200); fig.savefig(pdf)

    # ---------- CSV ----------
    csv = os.path.join(OUT, "geodynamics_flow.csv")
    with open(csv, "w") as f:
        f.write("shell,ell,rho_twin,ln_rho_twin,omega_twin,"
                "rho_twin_HLavg,triplet_count,rho_triplet,ln_rho_triplet\n")
        for k in kk:
            tcnt = tri.get(k, "")
            rtri = (tri[k]/total_nodes(k)) if k in tri else ""
            lrt  = math.log(tri[k]/total_nodes(k)) if k in tri else ""
            f.write(f"S{k},{ell(k):.6f},{rho_twin(k):.8f},{math.log(rho_twin(k)):.6f},"
                    f"{omega_twin(k):.4f},{rho_twin_HL_avg(k):.8f},"
                    f"{tcnt},{rtri},{lrt}\n")

    print(f"\nwrote {png}\n      {pdf}\n      {csv}")
    print(f"done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
