"""
fig02_resolution_ladder.py

Figure 2. What coarse temperature fields cost, and what they do not.

    (a) Baseline exposure error and the population whose local cooling demand is
        misstated by more than 20 percent, against effective resolution.
        End of century shown dashed on the same axes.
    (b) End of century GROWTH exposure error, decomposed into the covariance
        term and the threshold nonlinearity term.
    (c) Map of baseline local misstatement at 1 degree.
    (d) Population reassigned to a different building climate tier, baseline and
        end of century.

The figure exists to make one contrast visible without the reader consulting the
axis numbers: panels (a) and (b) share a y axis, so the level error visibly
dwarfs the growth error. That contrast is the paper's argument. A guide line at
the practical recommendation in Section 5.4 is drawn across (a) and (d).

The decomposition in (b) is exact, not assumed. Writing E for population
weighted exposure, T for the daily temperature field and C for the CDD operator:

    total        = E[C(coarsen(T))]  -  E[C(T)]
    nonlinearity = E[C(coarsen(T))]  -  E[coarsen(C(T))]
    covariance   = E[coarsen(C(T))]  -  E[C(T)]

so the two components sum to the total by construction. Stage 6 computes both
routes; nothing here is inferred.

Requires: preprocess_build_cache.py stages 1 and 6.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


MISSTATE_THRESHOLD = 0.20     # local relative error counted as a misstatement
GUIDE_RESOLUTION = 0.25       # the practical recommendation in Section 5.4


def zone_from_cache(lad, factor, period):
    """
    ASHRAE 169 zone index at a given effective resolution.

    Uses CDD base 10 C and HDD base 18.3 C, the bases the standard itself
    defines. These differ deliberately from the 24 C demand base used elsewhere
    in this paper: 24 C is India's mandated setpoint and the right anchor for
    demand, while the zone thresholds are only meaningful against the standard's
    own bases. Methods 3.7 states the distinction.
    """
    if factor is None:
        c = cc.climatology(lad.cdd10_fine, period)
        h = cc.climatology(lad.hdd18_fine, period)
    else:
        c = cc.climatology(lad.cdd10_coarse.sel(factor=factor), period)
        h = cc.climatology(lad.hdd18_coarse.sel(factor=factor), period)
    return cc.ashrae_zone(c, h)


def main() -> None:
    cc.set_style()
    gdf = cc.load_boundary()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    print("Loading population and the resolution ladder")
    pop = cc.load_population(cc.PATHS["POP_2010"], gdf,
                             expect_total=cc.CENSUS_2011_POP, label="WorldPop 2010")

    lad_h = xr.open_dataset(cc.cache_path("ladder_historical.nc"))
    lad_f = xr.open_dataset(cc.cache_path("ladder_ssp585.nc"))
    zon_h = xr.open_dataset(cc.cache_path("zone_indices_historical.nc"))
    zon_f = xr.open_dataset(cc.cache_path("zone_indices_ssp585.nc"))

    factors = lad_h.factor.values
    res = np.array([cc.factor_to_degrees(int(f)) for f in factors])

    # Climatologies on the fine grid and at every rung of the ladder.
    base_fine = cc.climatology(lad_h.cdd_fine, cc.BASELINE)
    far_fine = cc.climatology(lad_f.cdd_fine, cc.FUTURE_PERIODS["far"])
    growth_fine = far_fine - base_fine

    pop = pop.reindex_like(base_fine, method="nearest", tolerance=0.051)

    print(f"  fine baseline exposure "
          f"{cc.exposure(base_fine, pop)/1e12:.2f} x 10^12 person degC days")

    # ---- panel (a) and (d) quantities -----------------------------------
    rows = []
    zone_fine_base = zone_from_cache(zon_h, None, cc.BASELINE)
    zone_fine_far = zone_from_cache(zon_f, None, cc.FUTURE_PERIODS["far"])

    # Validation against the shares reported in Section 4.6. If these do not
    # reproduce, the zone thresholds are wrong and panel (d) is untrustworthy.
    tot = float(pop.sum())
    print("\n  Baseline zone shares (compare with Section 4.6):")
    for i, nm in enumerate(cc.ZONE_ORDER):
        sh = 100 * float(pop.where(zone_fine_base == i).sum()) / tot
        if sh > 0.05:
            print(f"    {nm:<16} {sh:5.1f}%")
    print("  End of century:")
    for i, nm in enumerate(cc.ZONE_ORDER):
        sh = 100 * float(pop.where(zone_fine_far == i).sum()) / tot
        if sh > 0.05:
            print(f"    {nm:<16} {sh:5.1f}%")

    for k, f in enumerate(factors):
        cb = cc.climatology(lad_h.cdd_from_coarse_T.sel(factor=f), cc.BASELINE)
        cf = cc.climatology(lad_f.cdd_from_coarse_T.sel(factor=f),
                            cc.FUTURE_PERIODS["far"])

        # National population weighted exposure error, baseline and far future.
        e_base = 100 * (cc.pop_weighted(cb, pop) / cc.pop_weighted(base_fine, pop) - 1)
        e_far = 100 * (cc.pop_weighted(cf, pop) / cc.pop_weighted(far_fine, pop) - 1)

        # Growth exposure error, and its two components.
        g_coarse = cf - cb
        g_flat = (cc.climatology(lad_f.cdd_coarsened.sel(factor=f),
                                 cc.FUTURE_PERIODS["far"])
                  - cc.climatology(lad_h.cdd_coarsened.sel(factor=f), cc.BASELINE))
        gw_fine = cc.pop_weighted(growth_fine, pop)
        e_growth = 100 * (cc.pop_weighted(g_coarse, pop) / gw_fine - 1)
        e_nonlin = 100 * (cc.pop_weighted(g_coarse, pop)
                          - cc.pop_weighted(g_flat, pop)) / gw_fine
        e_cov = 100 * (cc.pop_weighted(g_flat, pop) - gw_fine) / gw_fine

        # Population whose local level is misstated by more than the threshold.
        rel = np.abs(cb - base_fine) / base_fine.where(base_fine > 1.0)
        miss = 100 * float((pop.where(rel > MISSTATE_THRESHOLD).sum()) / float(pop.sum()))

        # Population reassigned to a different climate tier.
        zb = zone_from_cache(zon_h, f, cc.BASELINE)
        zf = zone_from_cache(zon_f, f, cc.FUTURE_PERIODS["far"])
        re_base = 100 * float(pop.where(zb != zone_fine_base).sum()) / float(pop.sum())
        re_far = 100 * float(pop.where(zf != zone_fine_far).sum()) / float(pop.sum())

        rows.append(dict(factor=int(f), resolution_deg=res[k],
                         exposure_err_base=e_base, exposure_err_far=e_far,
                         growth_err=e_growth, growth_nonlinearity=e_nonlin,
                         growth_covariance=e_cov,
                         pop_misstated=miss,
                         tier_reassigned_base=re_base, tier_reassigned_far=re_far))
        print(f"  {res[k]:.1f} deg: level {e_base:+.2f}%, growth {e_growth:+.2f}% "
              f"(cov {e_cov:+.2f} + nonlin {e_nonlin:+.2f}), "
              f"misstated {miss:.1f}%, tier {re_far:.1f}%")

    df = pd.DataFrame(rows)
    csv = f"{cc.FIGDIR}/Jha_etal_Fig2_ladder.csv"
    df.to_csv(csv, index=False, float_format="%.4f")
    print(f"\nLadder table -> {csv}")

    resid = np.abs(df.growth_err - df.growth_covariance - df.growth_nonlinearity).max()
    print(f"Decomposition closure, maximum residual: {resid:.2e} percentage points")
    assert resid < 1e-6, "the growth decomposition does not close"

    at1 = df[df.factor == 10].iloc[0]
    print(f"\nAt 1 degree: level error {at1.exposure_err_base:+.2f}%, "
          f"growth error {at1.growth_err:+.2f}%, "
          f"{at1.pop_misstated:.1f}% of population misstated by >"
          f"{100*MISSTATE_THRESHOLD:.0f}%")

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(7.5, 6.6))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.38,
                          left=0.085, right=0.965, top=0.95, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) level error, shared y axis with (b)
    ax_a.axhline(0, color="0.6", linewidth=0.6)
    ax_a.plot(df.resolution_deg, df.exposure_err_base, "o-", color="#CC3311",
              markersize=4, linewidth=1.4, label="Baseline 1985-2014")
    ax_a.plot(df.resolution_deg, df.exposure_err_far, "s--", color="#CC3311",
              markersize=3.5, linewidth=1.2, alpha=0.65, label="2081-2100")
    ax_a.axvline(GUIDE_RESOLUTION, color="0.35", linewidth=0.8, linestyle=":")
    ax_a.set_xlabel("Effective resolution ($\\degree$)")
    ax_a.set_ylabel("Exposure error (%)")
    cc.panel_label(ax_a, "(a) Demand level")
    ax_a.legend(frameon=False, loc="lower left")

    ax_a2 = ax_a.twinx()
    ax_a2.plot(df.resolution_deg, df.pop_misstated, "^-", color="#4477AA",
               markersize=3.5, linewidth=1.1)
    ax_a2.set_ylabel(f"Population misstated by >"
                     f"{100*MISSTATE_THRESHOLD:.0f}% (%)", color="#4477AA")
    ax_a2.tick_params(axis="y", colors="#4477AA")
    ax_a2.spines["right"].set_color("#4477AA")

    # (b) growth error and its decomposition, same y limits as (a)
    ax_b.axhline(0, color="0.6", linewidth=0.6)
    ax_b.plot(df.resolution_deg, df.growth_err, "o-", color="#333333",
              markersize=4, linewidth=1.4, label="Total")
    ax_b.plot(df.resolution_deg, df.growth_covariance, "v--", color="#EE7733",
              markersize=3.5, linewidth=1.1, label="Covariance")
    ax_b.plot(df.resolution_deg, df.growth_nonlinearity, "^--", color="#009988",
              markersize=3.5, linewidth=1.1, label="Threshold nonlinearity")
    ax_b.set_xlabel("Effective resolution ($\\degree$)")
    ax_b.set_ylabel("Growth exposure error (%)")
    cc.panel_label(ax_b, "(b) Demand growth")
    ax_b.legend(frameon=False, loc="lower left")

    lim = ax_a.get_ylim()
    span = max(abs(lim[0]), abs(lim[1]))
    for ax in (ax_a, ax_b):
        ax.set_ylim(-span, span)
        ax.set_xlim(0.15, 1.05)
    ax_b.text(0.97, 0.05,
              "same $y$ scale as (a)", transform=ax_b.transAxes,
              ha="right", va="bottom", fontsize=7, style="italic", color="0.35")

    # (c) map of local misstatement at 1 degree
    cb1 = cc.climatology(lad_h.cdd_from_coarse_T.sel(factor=10), cc.BASELINE)
    rel = 100 * (cb1 - base_fine) / base_fine.where(base_fine > 1.0)
    vmax = cc.robust_max(abs(rel), 98.0)
    im = ax_c.pcolormesh(rel.lon, rel.lat, rel, cmap=cc.CMAP_BIAS,
                         vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    cc.style_map(ax_c, gdf)
    cc.panel_label(ax_c, "(c) Local misstatement at 1$\\degree$")
    ax_c.set_xlabel("Longitude ($\\degree$E)")
    ax_c.set_ylabel("Latitude ($\\degree$N)")
    cb = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03, extend="both")
    cb.set_label("Relative error (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.5)

    # (d) tier reassignment
    ax_d.plot(df.resolution_deg, df.tier_reassigned_base, "o-", color="#882255",
              markersize=4, linewidth=1.4, label="Baseline")
    ax_d.plot(df.resolution_deg, df.tier_reassigned_far, "s--", color="#882255",
              markersize=3.5, linewidth=1.2, alpha=0.7, label="2081-2100")
    ax_d.axvline(GUIDE_RESOLUTION, color="0.35", linewidth=0.8, linestyle=":")
    ax_d.set_xlabel("Effective resolution ($\\degree$)")
    ax_d.set_ylabel("Population in the wrong zone (%)")
    ax_d.set_xlim(0.15, 1.05)
    cc.panel_label(ax_d, "(d) ASHRAE 169 zone assignment")
    ax_d.legend(frameon=False, loc="lower right")
    ax_d.annotate("0.25$\\degree$", xy=(GUIDE_RESOLUTION, ax_d.get_ylim()[0]),
                  xytext=(4, 4), textcoords="offset points",
                  fontsize=7, color="0.35", va="bottom", ha="left")

    cc.save_figure(fig, "Jha_etal_Fig2")
    plt.close(fig)

    print("\nNOTE: zones follow ASHRAE 169 (CDD base 10 C, HDD base 18.3 C). "
          "ECBC and Eco-Niwas Samhita key on the NBC five zone scheme, which is "
          "defined on mean monthly temperature and relative humidity, not on "
          "degree days. Sections 4.6 and 5.3 must not claim these tiers ARE the "
          "ECBC zones; they indicate code relevant pressure.")


if __name__ == "__main__":
    main()
