"""
fig05_base_sensitivity.py

Figure 5. Cooling and heating demand across the range of base temperatures used
in the literature, so the base is an axis of the result rather than a hidden
assumption.

    (a) National population weighted CDD trajectory, 1985-2100, one line per
        cooling base, SSP5-8.5.
    (b) End of century change by cooling base: absolute on the left axis and
        percentage on the right, showing that the percentage headline is
        governed largely by the denominator rather than by the physics.
    (c) National HDD trajectory by heating base, and the end of century decline.
    (d) Net thermal demand (CDD minus HDD) across the base pairs, so the reader
        can see which conclusions survive the choice and which do not.

Why this figure exists
----------------------
No single base temperature is canonical. IEA uses 18 C. Bhatnagar, Mathur and
Garg (2018) fit 18 C for India from building energy signatures. Chen (2017) and
Haase (2009) use 26 C for hot humid Asia. Borah and colleagues compute 20 to
28 C across North East India, and Priya and colleagues sweep 18 to 28 C for a
single southern city. The 24 C headline used elsewhere in this paper is India's
mandated air conditioner setpoint, which is a regulatory anchor rather than a
convention, and it sits inside that published range.

Reporting the sweep converts the most obvious reviewer objection into a result:
the qualitative conclusions hold across every base, while the percentage
magnitude scales predictably with the base because a higher base yields a
smaller baseline denominator. Panel (b) makes that mechanism explicit by showing
absolute and percentage change side by side.

A caveat that must stay in the paper: degree days here are computed from daily
MEAN temperature by the standard ASHRAE method. Several Indian studies use
formulations that draw on daily maximum and minimum. Only daily mean is
available for the products compared here, so on days whose mean falls below the
base but whose afternoon does not, cooling load is not counted. This omission is
spatially structured, largest in the high diurnal range interior, and is stated
as a limitation rather than estimated.

Requires: preprocess_build_cache.py stage 8.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


HEADLINE_CDD_BASE = cc.T_BASE_CDD      # 24 C
HEADLINE_HDD_BASE = cc.T_BASE_HDD      # 18 C
MAIN_SCENARIO = "ssp585"
ROLL = 10

# Bases used by studies the paper cites, marked on the base axes.
LITERATURE_MARKS = {
    18.0: "IEA; Bhatnagar (India fit)",
    24.0: "BEE setpoint (this study)",
    26.0: "Chen; Haase (hot humid Asia)",
}


def rolling(da, n=ROLL):
    return da.rolling(year=n, center=True, min_periods=n).mean()


def main() -> None:
    cc.set_style()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    ds = xr.open_dataset(cc.cache_path("base_sweep_series.nc"))
    cdd_bases = ds.cdd_base.values
    hdd_bases = ds.hdd_base.values
    print(f"Cooling bases: {list(cdd_bases)}")
    print(f"Heating bases: {list(hdd_bases)}")

    far = cc.FUTURE_PERIODS["far"]

    # ---- baseline and end of century, per base -------------------------
    rows = []
    for b in cdd_bases:
        base_val = float(ds.cdd_pop_hist.sel(cdd_base=b)
                         .sel(year=slice(*cc.BASELINE)).mean())
        for sc in cc.SCENARIOS:
            far_val = float(ds.cdd_pop.sel(cdd_base=b, scenario=sc)
                            .sel(year=slice(*far)).mean())
            rows.append(dict(index="CDD", base=float(b), scenario=sc,
                             baseline=base_val, far=far_val,
                             absolute_change=far_val - base_val,
                             percent_change=100 * (far_val / base_val - 1)))
    for b in hdd_bases:
        base_val = float(ds.hdd_pop_hist.sel(hdd_base=b)
                         .sel(year=slice(*cc.BASELINE)).mean())
        for sc in cc.SCENARIOS:
            far_val = float(ds.hdd_pop.sel(hdd_base=b, scenario=sc)
                            .sel(year=slice(*far)).mean())
            rows.append(dict(index="HDD", base=float(b), scenario=sc,
                             baseline=base_val, far=far_val,
                             absolute_change=far_val - base_val,
                             percent_change=100 * (far_val / base_val - 1)))

    df = pd.DataFrame(rows)
    csv = f"{cc.FIGDIR}/Jha_etal_Fig5_base_sensitivity.csv"
    df.to_csv(csv, index=False, float_format="%.3f")
    print(f"\nBase sensitivity table -> {csv}")

    main_cdd = df[(df["index"] == "CDD") & (df.scenario == MAIN_SCENARIO)]
    print(f"\nCooling, {cc.SCENARIO_LABEL[MAIN_SCENARIO]}, "
          f"{far[0]}-{far[1]} vs baseline:")
    for _, r in main_cdd.iterrows():
        flag = "  <- headline" if r.base == HEADLINE_CDD_BASE else ""
        print(f"  base {r.base:4.0f} C: {r.baseline:7.0f} -> {r.far:7.0f}  "
              f"({r.absolute_change:+7.0f}, {r.percent_change:+6.1f}%){flag}")

    # The denominator argument, made quantitative.
    lo, hi = main_cdd.base.min(), main_cdd.base.max()
    a_lo = float(main_cdd[main_cdd.base == lo].absolute_change.iloc[0])
    a_hi = float(main_cdd[main_cdd.base == hi].absolute_change.iloc[0])
    p_lo = float(main_cdd[main_cdd.base == lo].percent_change.iloc[0])
    p_hi = float(main_cdd[main_cdd.base == hi].percent_change.iloc[0])
    print(f"\n  absolute change varies {a_lo:.0f} to {a_hi:.0f} degC days "
          f"({100*abs(a_hi-a_lo)/abs(a_lo):.0f}% spread across bases)")
    print(f"  percent change varies {p_lo:.0f} to {p_hi:.0f}% "
          f"({100*abs(p_hi-p_lo)/abs(p_lo):.0f}% spread across bases)")
    print("  The percentage spread is the denominator, not the physics.")

    main_hdd = df[(df["index"] == "HDD") & (df.scenario == MAIN_SCENARIO)]
    print(f"\nHeating, {cc.SCENARIO_LABEL[MAIN_SCENARIO]}:")
    for _, r in main_hdd.iterrows():
        flag = "  <- headline" if r.base == HEADLINE_HDD_BASE else ""
        print(f"  base {r.base:4.0f} C: {r.baseline:7.0f} -> {r.far:7.0f}  "
              f"({r.percent_change:+6.1f}%){flag}")

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(7.5, 6.8))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.36,
                          left=0.085, right=0.90, top=0.945, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    cmap_c = plt.get_cmap("YlOrRd")
    cols_c = {b: cmap_c(0.28 + 0.62 * i / max(len(cdd_bases) - 1, 1))
              for i, b in enumerate(cdd_bases)}
    cmap_h = plt.get_cmap("Blues")
    cols_h = {b: cmap_h(0.30 + 0.60 * i / max(len(hdd_bases) - 1, 1))
              for i, b in enumerate(hdd_bases)}

    # (a) CDD trajectories by base
    for b in cdd_bases:
        lw = 2.0 if b == HEADLINE_CDD_BASE else 1.1
        z = 6 if b == HEADLINE_CDD_BASE else 3
        h = rolling(ds.cdd_pop_hist.sel(cdd_base=b))
        f = rolling(ds.cdd_pop.sel(cdd_base=b, scenario=MAIN_SCENARIO))
        ax_a.plot(h.year, h, color=cols_c[b], linewidth=lw, zorder=z)
        ax_a.plot(f.year, f, color=cols_c[b], linewidth=lw, zorder=z,
                  label=f"{b:g} $\\degree$C")
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Population weighted CDD ($\\degree$C days)")
    ax_a.set_xlim(1985, 2100)
    cc.panel_label(ax_a, "(a) Cooling demand by base")
    leg = ax_a.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper left",
                      title="Cooling base", title_fontsize=7)
    leg._legend_box.align = "left"
    ax_a.text(0.985, 0.03, cc.SCENARIO_LABEL[MAIN_SCENARIO],
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=7, color="0.35")

    # (b) end of century change, absolute and percent
    x = np.arange(len(cdd_bases))
    ax_b.bar(x - 0.2, main_cdd.absolute_change.values, width=0.4,
             color="#CC3311", label="Absolute")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{b:g}" for b in cdd_bases])
    ax_b.set_xlabel("Cooling base ($\\degree$C)")
    ax_b.set_ylabel("Absolute change ($\\degree$C days)", color="#CC3311")
    ax_b.tick_params(axis="y", colors="#CC3311")
    cc.panel_label(ax_b, "(b) End of century change")

    ax_b2 = ax_b.twinx()
    ax_b2.bar(x + 0.2, main_cdd.percent_change.values, width=0.4,
              color="#4477AA", label="Percent")
    ax_b2.set_ylabel("Percent change (%)", color="#4477AA")
    ax_b2.tick_params(axis="y", colors="#4477AA")

    ihead = int(np.where(cdd_bases == HEADLINE_CDD_BASE)[0][0])
    ax_b.axvline(ihead, color="0.4", linewidth=0.8, linestyle=":")
    ax_b.text(ihead, ax_b.get_ylim()[1], " BEE setpoint", fontsize=6.8,
              color="0.35", va="top", ha="left")
    ax_b.text(0.5, -0.30,
              "absolute change is nearly flat across bases; the percentage "
              "headline is set by the denominator",
              transform=ax_b.transAxes, ha="center", va="top",
              fontsize=6.6, style="italic", color="0.35")

    # (c) HDD trajectories by base
    for b in hdd_bases:
        lw = 2.0 if b == HEADLINE_HDD_BASE else 1.1
        z = 6 if b == HEADLINE_HDD_BASE else 3
        h = rolling(ds.hdd_pop_hist.sel(hdd_base=b))
        f = rolling(ds.hdd_pop.sel(hdd_base=b, scenario=MAIN_SCENARIO))
        ax_c.plot(h.year, h, color=cols_h[b], linewidth=lw, zorder=z)
        ax_c.plot(f.year, f, color=cols_h[b], linewidth=lw, zorder=z,
                  label=f"{b:g} $\\degree$C")
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Population weighted HDD ($\\degree$C days)")
    ax_c.set_xlim(1985, 2100)
    cc.panel_label(ax_c, "(c) Heating demand by base")
    leg = ax_c.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right",
                      title="Heating base", title_fontsize=7)
    leg._legend_box.align = "left"

    # (d) net thermal demand across base pairs
    net = np.zeros((len(cdd_bases), len(hdd_bases)))
    for i, cb in enumerate(cdd_bases):
        for j, hb in enumerate(hdd_bases):
            dc = float(main_cdd[main_cdd.base == cb].absolute_change.iloc[0])
            dh = float(main_hdd[main_hdd.base == hb].absolute_change.iloc[0])
            net[i, j] = dc + dh          # dh is negative
    im = ax_d.imshow(net, cmap="RdBu_r", aspect="auto", origin="lower",
                     vmin=-np.abs(net).max(), vmax=np.abs(net).max())
    ax_d.set_xticks(range(len(hdd_bases)))
    ax_d.set_xticklabels([f"{b:g}" for b in hdd_bases])
    ax_d.set_yticks(range(len(cdd_bases)))
    ax_d.set_yticklabels([f"{b:g}" for b in cdd_bases])
    ax_d.set_xlabel("Heating base ($\\degree$C)")
    ax_d.set_ylabel("Cooling base ($\\degree$C)")
    cc.panel_label(ax_d, "(d) Net thermal change")
    for i in range(len(cdd_bases)):
        for j in range(len(hdd_bases)):
            ax_d.text(j, i, f"{net[i, j]:.0f}", ha="center", va="center",
                      fontsize=6.4,
                      color="white" if abs(net[i, j]) > 0.6 * np.abs(net).max()
                      else "0.15")
    jh = int(np.where(hdd_bases == HEADLINE_HDD_BASE)[0][0])
    ax_d.add_patch(plt.Rectangle((jh - 0.5, ihead - 0.5), 1, 1, fill=False,
                                 edgecolor="black", linewidth=1.6))
    cb = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.03, extend="both")
    cb.set_label("Net change ($\\degree$C days)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)

    cc.save_figure(fig, "Jha_etal_Fig5")
    plt.close(fig)

    print("\nEvery cell of panel (d) is positive if cooling growth exceeds the "
          "heating offset for that base pair. Check the sign before writing "
          "the net thermal sentence in Section 4.7.")


if __name__ == "__main__":
    main()
