"""
fig08_net_thermal.py

Figure 8. Net thermal demand: does falling heating offset rising cooling?

    (a) Map of net change in thermal degree days, 2081-2100 minus baseline,
        SSP5-8.5, where net = change in CDD plus change in HDD (the latter
        negative)
    (b) National change by scenario, cooling and heating shown separately with
        the net marked, and circular block bootstrap intervals
    (c) Population share for which the heating offset exceeds the cooling
        increase, by scenario

A caution the caption should carry. Adding CDD and HDD treats a cooling degree
day and a heating degree day as interchangeable units of thermal demand. They
are not: in India cooling is overwhelmingly electric while heating is largely
non-electric, and the two have different conversion efficiencies. The net is a
climatic quantity, not an energy quantity, and panel (b) therefore shows the two
components alongside the net rather than only the sum.

Requires: preprocess_build_cache.py stage 1.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


MAIN_SCENARIO = "ssp585"


def national(field, weights):
    return float((field * weights).sum() / weights.sum())


def main() -> None:
    cc.set_style()
    gdf = cc.load_boundary()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    print("Loading cached CDD and HDD")
    cdd_h = cc.read_cache("annual_cdd_ddpm_historical.nc", var="cdd")
    hdd_h = cc.read_cache("annual_hdd_ddpm_historical.nc", var="hdd")

    c0 = cc.climatology(cdd_h, cc.BASELINE)
    h0 = cc.climatology(hdd_h, cc.BASELINE)

    pop = cc.load_population(cc.PATHS["POP_2010"], gdf,
                             expect_total=cc.CENSUS_2011_POP, label="WorldPop 2010")
    pop = pop.reindex_like(c0, method="nearest", tolerance=0.051)
    tot_pop = float(pop.sum())

    w = np.cos(np.deg2rad(c0.lat)).broadcast_like(c0).where(c0.notnull())

    far = cc.FUTURE_PERIODS["far"]
    rows, fields = [], {}

    for sc in cc.SCENARIOS:
        cdd_f = cc.read_cache(f"annual_cdd_ddpm_{sc}.nc", var="cdd")
        hdd_f = cc.read_cache(f"annual_hdd_ddpm_{sc}.nc", var="hdd")
        c1 = cc.climatology(cdd_f, far)
        h1 = cc.climatology(hdd_f, far)

        dc, dh = c1 - c0, h1 - h0
        net = dc + dh
        fields[sc] = dict(dc=dc, dh=dh, net=net)

        # National series for bootstrap intervals on the change.
        cs = ((cdd_f * w).sum(dim=("lat", "lon")) / w.sum()).sel(
            year=slice(*far)).values
        c_base = ((cdd_h * w).sum(dim=("lat", "lon")) / w.sum()).sel(
            year=slice(*cc.BASELINE)).values
        lo_c, hi_c = cc.block_bootstrap_ci(cs - c_base.mean(), block=3)

        hs = ((hdd_f * w).sum(dim=("lat", "lon")) / w.sum()).sel(
            year=slice(*far)).values
        h_base = ((hdd_h * w).sum(dim=("lat", "lon")) / w.sum()).sel(
            year=slice(*cc.BASELINE)).values
        lo_h, hi_h = cc.block_bootstrap_ci(hs - h_base.mean(), block=3)

        lo_n, hi_n = cc.block_bootstrap_ci((cs - c_base.mean())
                                           + (hs - h_base.mean()), block=3)

        offset_pop = 100 * float(pop.where(net < 0).sum()) / tot_pop

        rows.append(dict(scenario=sc,
                         d_cdd=national(dc, w), d_cdd_lo=lo_c, d_cdd_hi=hi_c,
                         d_hdd=national(dh, w), d_hdd_lo=lo_h, d_hdd_hi=hi_h,
                         net=national(net, w), net_lo=lo_n, net_hi=hi_n,
                         d_cdd_popw=cc.pop_weighted(dc, pop),
                         d_hdd_popw=cc.pop_weighted(dh, pop),
                         net_popw=cc.pop_weighted(net, pop),
                         pop_pct_offset=offset_pop))
        r = rows[-1]
        print(f"  {cc.SCENARIO_LABEL[sc]}: CDD {r['d_cdd']:+.0f} "
              f"[{lo_c:+.0f}, {hi_c:+.0f}], HDD {r['d_hdd']:+.0f} "
              f"[{lo_h:+.0f}, {hi_h:+.0f}], net {r['net']:+.0f} "
              f"[{lo_n:+.0f}, {hi_n:+.0f}]; heating offset dominates for "
              f"{offset_pop:.2f}% of population")

    df = pd.DataFrame(rows)
    csv = f"{cc.FIGDIR}/Jha_etal_Fig8_net_thermal.csv"
    df.to_csv(csv, index=False, float_format="%.3f")
    print(f"\nNet thermal table -> {csv}")

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(7.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.85], wspace=0.34,
                          left=0.075, right=0.985, top=0.90, bottom=0.145)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # (a) net change map
    net = fields[MAIN_SCENARIO]["net"]
    vmax = cc.robust_max(abs(net), 99.0)
    im = ax_a.pcolormesh(net.lon, net.lat, net, cmap=cc.CMAP_DIVERGE,
                         vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    cc.style_map(ax_a, gdf)
    cc.panel_label(ax_a, f"(a) Net change, {cc.SCENARIO_LABEL[MAIN_SCENARIO]}")
    ax_a.set_xlabel("Longitude ($\\degree$E)")
    ax_a.set_ylabel("Latitude ($\\degree$N)")
    cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.03, extend="both")
    cb.set_label("Net thermal change ($\\degree$C days)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)

    # (b) components and net
    x = np.arange(len(cc.SCENARIOS))
    ax_b.axhline(0, color="0.6", linewidth=0.7)
    ax_b.bar(x - 0.24, df.d_cdd, width=0.30, color="#CC3311", label="Cooling")
    ax_b.bar(x + 0.06, df.d_hdd, width=0.30, color="#4477AA", label="Heating")
    ax_b.errorbar(x - 0.24, df.d_cdd,
                  yerr=[df.d_cdd - df.d_cdd_lo, df.d_cdd_hi - df.d_cdd],
                  fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2)
    ax_b.plot(x + 0.30, df.net, "D", color="0.15", markersize=6, zorder=5,
              label="Net")
    ax_b.errorbar(x + 0.30, df.net,
                  yerr=[df.net - df.net_lo, df.net_hi - df.net],
                  fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2)
    for k, r in df.iterrows():
        ax_b.text(k + 0.30, r.net, f"  {r.net:+.0f}", fontsize=6.8,
                  va="center", ha="left")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([cc.SCENARIO_LABEL[s] for s in cc.SCENARIOS],
                         fontsize=7.5)
    ax_b.set_ylabel("Change, 2081-2100 vs baseline ($\\degree$C days)")
    cc.panel_label(ax_b, "(b) National components")
    ax_b.legend(frameon=False, fontsize=7, loc="upper left")
    ax_b.text(0.98, 0.03, "bars: cos-lat weighted;\nintervals: block bootstrap",
              transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=6.4, color="0.4", linespacing=1.4)

    # (c) population for which heating offset dominates
    ax_c.bar(x, df.pop_pct_offset, width=0.55, color="#4477AA")
    for k, v in enumerate(df.pop_pct_offset):
        ax_c.text(k, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=7)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([cc.SCENARIO_LABEL[s] for s in cc.SCENARIOS],
                         fontsize=7.5, rotation=20, ha="right")
    ax_c.set_ylabel("Population where heating\noffset exceeds cooling (%)")
    cc.panel_label(ax_c, "(c) Where the offset wins")
    ax_c.set_ylim(0, max(df.pop_pct_offset.max() * 1.45, 1.0))
    ax_c.grid(axis="y", color="0.92", linewidth=0.5)
    ax_c.set_axisbelow(True)

    cc.save_figure(fig, "Jha_etal_Fig8")
    plt.close(fig)

    print("\nCaption note: CDD and HDD are summed as climatic degree days. In "
          "India cooling is overwhelmingly electric and heating largely is not, "
          "so the net is not an energy quantity. Panel (b) shows the components "
          "for that reason.")


if __name__ == "__main__":
    main()
