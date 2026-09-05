"""
fig07_drivers.py

Figure 7. Climate change against population growth as drivers of cooling
exposure, at global warming levels, for two demographic pathways.

    (a) Exposure terms against warming level, SSP2 population
    (b) The same for SSP5 population
    (c) The crossover warming level under both attributions of the joint term

Decomposition. Writing P for population and C for population weighted CDD, with
subscript 0 for baseline and 1 for a warming level, total exposure change
decomposes exactly:

    total   = P1*C1 - P0*C0
    climate = P0*(C1 - C0)
    popul   = (P1 - P0)*C0
    joint   = (P1 - P0)*(C1 - C0)

and total = climate + popul + joint identically. The script asserts closure.

Why both attributions are shown. The joint term belongs to neither driver alone,
and it is not small: it reaches a substantial fraction of the population term at
high warming. Assigning it to population moves the crossover by more than a
degree. Draft v5 therefore quotes the crossover as a range spanning both
attributions rather than a single value, and this figure shows the band rather
than a line so the text and the figure agree.

Warming level epochs are defined for the CMIP6 ensemble mean under SSP5-8.5.
Per model warming levels are not computable here, because global mean surface
temperature requires global fields and every product in this study covers the
India domain only. Methods must cite the source of the epochs.

Requires: preprocess_build_cache.py stage 1, and the SSP2 and SSP5 population
grids aggregated to the analysis grid.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


PATHWAYS = ["ssp2", "ssp5"]
PATHWAY_LABEL = {"ssp2": "SSP2 population", "ssp5": "SSP5 population"}
GWLS = list(cc.WARMING_EPOCHS.keys())


def main() -> None:
    cc.set_style()
    gdf = cc.load_boundary()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    print("Loading CDD and population")
    hist = cc.read_cache("annual_cdd_ddpm_historical.nc", var="cdd")
    fut = cc.read_cache("annual_cdd_ddpm_ssp585.nc", var="cdd")

    cdd0 = cc.climatology(hist, cc.BASELINE)
    pop0 = cc.load_population(cc.PATHS["POP_2010"], gdf,
                              expect_total=cc.CENSUS_2011_POP,
                              label="baseline 2010")
    pop0 = pop0.reindex_like(cdd0, method="nearest", tolerance=0.051)
    e0 = float((cdd0 * pop0).sum())
    print(f"  baseline exposure {e0/1e12:.3f} x 10^12 person degC days")

    rows = []
    for pw in PATHWAYS:
        for gwl in GWLS:
            epoch = cc.WARMING_EPOCHS[gwl]
            decade = cc.EPOCH_TO_DECADE[gwl]
            cdd1 = cc.climatology(fut, epoch)

            path = cc.pop_path(pw, decade)
            if not os.path.exists(path):
                print(f"  MISSING {path}")
                continue
            pop1 = cc.load_population(path, gdf, label=f"{pw} {decade}")
            pop1 = pop1.reindex_like(cdd0, method="nearest", tolerance=0.051)

            valid = cdd0.notnull() & cdd1.notnull() & pop0.notnull() & pop1.notnull()
            c0, c1 = cdd0.where(valid), cdd1.where(valid)
            p0, p1 = pop0.where(valid), pop1.where(valid)

            # Accumulate in float64. Summing ~29,000 float32 cells of order
            # 1e12 otherwise leaves a relative residual near 1e-7, which is
            # rounding rather than a broken decomposition.
            c0d, c1d = c0.astype("float64"), c1.astype("float64")
            p0d, p1d = p0.astype("float64"), p1.astype("float64")

            climate = float((p0d * (c1d - c0d)).sum())
            popul = float(((p1d - p0d) * c0d).sum())
            joint = float(((p1d - p0d) * (c1d - c0d)).sum())
            total = float((p1d * c1d).sum() - (p0d * c0d).sum())

            resid = abs(total - climate - popul - joint) / abs(total)
            assert resid < 1e-6, (
                f"decomposition does not close: relative residual {resid:.2e}")

            rows.append(dict(pathway=pw, gwl=float(gwl), decade=decade,
                             epoch=f"{epoch[0]}-{epoch[1]}",
                             climate=climate, population=popul, joint=joint,
                             total=total,
                             pop_total=float(p1.sum()),
                             climate_share=100 * climate / total,
                             population_share=100 * popul / total,
                             joint_share=100 * joint / total))
            r = rows[-1]
            print(f"  {pw} {gwl} C ({epoch[0]}-{epoch[1]}, pop {decade}): "
                  f"climate {climate/1e12:6.2f}, population {popul/1e12:6.2f}, "
                  f"joint {joint/1e12:6.2f} (10^12 person degC days), "
                  f"joint = {100*joint/max(popul,1):.0f}% of population term")

    df = pd.DataFrame(rows)
    csv = f"{cc.FIGDIR}/Jha_etal_Fig7_drivers.csv"
    df.to_csv(csv, index=False, float_format="%.4f")
    print(f"\nDriver table -> {csv}")

    # ---- crossover under both attributions -------------------------------
    def crossover(sub, attribute_joint_to_population: bool):
        """
        Warming level at which the climate term overtakes the population term.

        Returns (value, censored). The analysis resolves the crossover only
        within the span of the warming levels evaluated. If the climate term is
        already the larger one at the lowest level, the crossover lies at or
        below that level and cannot be dated more precisely; the value is then
        the lowest level and censored is True. Reporting a left censored bound
        as though it were a measured crossing would overstate the precision of
        the analysis, in the same way that the time of emergence in Section 4.4
        is bounded rather than resolved.
        """
        x = sub.gwl.values
        clim = sub.climate.values
        popl = sub.population.values + (sub.joint.values
                                        if attribute_joint_to_population else 0.0)
        d = clim - popl
        idx = np.where(d > 0)[0]
        if idx.size == 0:
            return np.nan, False            # no crossing within the range
        i = idx[0]
        if i == 0:
            return float(x[0]), True        # left censored
        return float(x[i - 1] + (x[i] - x[i - 1])
                     * (-d[i - 1]) / (d[i] - d[i - 1])), False

    cross = []
    for pw in PATHWAYS:
        sub = df[df.pathway == pw].sort_values("gwl")
        if sub.empty:
            continue
        a, a_cens = crossover(sub, False)
        b, b_cens = crossover(sub, True)
        cross.append(dict(pathway=pw, additive=a, additive_censored=a_cens,
                          joint_with_population=b,
                          joint_with_population_censored=b_cens))
        pa = f"at or below {a:.1f}" if a_cens else f"{a:.1f}"
        pb = f"at or below {b:.1f}" if b_cens else f"{b:.1f}"
        print(f"  {PATHWAY_LABEL[pw]}: crossover {pa} C additive, "
              f"{pb} C if the joint term goes to population")
    cdf = pd.DataFrame(cross)
    cdf.to_csv(f"{cc.FIGDIR}/Jha_etal_Fig7_crossover.csv", index=False,
               float_format="%.2f")

    if not cdf.empty:
        lo_a, hi_a = cdf.additive.min(), cdf.additive.max()
        lo_j, hi_j = (cdf.joint_with_population.min(),
                      cdf.joint_with_population.max())
        pre_a = "at or below " if bool(cdf.additive_censored.any()) else ""
        pre_j = ("at or below "
                 if bool(cdf.joint_with_population_censored.any()) else "")
        print(f"\n  Report the crossover as {pre_a}{lo_a:.1f} to {hi_a:.1f} C "
              f"under additive attribution, or {pre_j}{lo_j:.1f} to {hi_j:.1f} C "
              f"if the joint term is assigned to population.")
        if bool(cdf.additive_censored.any()):
            print("  NOTE: at least one pathway is left censored. The climate "
                  "term already exceeds the population term at the lowest "
                  "warming level evaluated, so the lower bound is a bound and "
                  "not a measured crossing. Write it as 'at or below'.")

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(7.5, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.075, right=0.985,
                          top=0.90, bottom=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    C_CLIM, C_POP, C_JOINT = "#CC3311", "#4477AA", "#999933"

    for k, pw in enumerate(PATHWAYS):
        ax = axes[k]
        sub = df[df.pathway == pw].sort_values("gwl")
        if sub.empty:
            continue
        x = sub.gwl.values
        ax.plot(x, sub.climate / 1e12, "o-", color=C_CLIM, markersize=4,
                linewidth=1.6, label="Climate")
        ax.plot(x, sub.population / 1e12, "s-", color=C_POP, markersize=4,
                linewidth=1.6, label="Population")
        ax.fill_between(x, sub.population / 1e12,
                        (sub.population + sub.joint) / 1e12,
                        color=C_JOINT, alpha=0.30, linewidth=0,
                        label="Joint (attribution range)")
        ax.plot(x, sub.joint / 1e12, "^--", color=C_JOINT, markersize=3.5,
                linewidth=1.0)
        ax.axhline(0, color="0.7", linewidth=0.6)
        ax.set_xlabel("Global warming level ($\\degree$C)")
        if k == 0:
            ax.set_ylabel("Exposure change\n($10^{12}$ person $\\degree$C days)")
        cc.panel_label(ax, f"({'ab'[k]}) {PATHWAY_LABEL[pw]}")
        if k == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper left")

    lims = [ax.get_ylim() for ax in axes[:2]]
    lo = min(l[0] for l in lims); hi = max(l[1] for l in lims)
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
    axes[1].set_yticklabels([])

    # (c) crossover bands
    ax = axes[2]
    for k, r in cdf.iterrows():
        lo_, hi_ = sorted([r.additive, r.joint_with_population])
        ax.barh(k, hi_ - lo_, left=lo_, height=0.42, color="#882255",
                alpha=0.28, edgecolor="none")
        ax.plot([r.additive], [k], "o", color="#882255", markersize=6,
                zorder=4, label="Additive" if k == 0 else None)
        ax.plot([r.joint_with_population], [k], "D", color="#882255",
                markersize=5, markerfacecolor="white", zorder=4,
                label="Joint to population" if k == 0 else None)
        lbl = (f"$\\leq${lo_:.1f} to {hi_:.1f} $\\degree$C"
               if bool(r.additive_censored) else
               f"{lo_:.1f} to {hi_:.1f} $\\degree$C")
        ax.text(hi_ + 0.10, k, lbl, fontsize=7, va="center")
        if bool(r.additive_censored):
            ax.annotate("", xy=(lo_ - 0.30, k), xytext=(lo_, k),
                        arrowprops=dict(arrowstyle="->", color="#882255",
                                        linewidth=1.0))
    ax.set_yticks(range(len(cdf)))
    ax.set_yticklabels([])
    for k, r in cdf.iterrows():
        ax.text(1.06, k - 0.28, PATHWAY_LABEL[r.pathway], fontsize=7.5,
                va="bottom", ha="left", color="0.25")
    ax.invert_yaxis()
    ax.set_xlabel("Warming level at crossover ($\\degree$C)")
    cc.panel_label(ax, "(c) Climate overtakes population")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="x", color="0.92", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0.9, 5.0)
    ax.set_ylim(len(cdf) - 0.4, -0.9)

    cc.save_figure(fig, "Jha_etal_Fig7")
    plt.close(fig)


if __name__ == "__main__":
    main()
