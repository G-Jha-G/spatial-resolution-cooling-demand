"""
fig06_zones.py

Figure 6. Building climate zone reclassification under warming.

    (a) Baseline zone map, 1985-2014
    (b) End of century zone map, 2081-2100, SSP5-8.5
    (c) Population transition matrix, baseline zone to future zone, in millions
    (d) Population share by zone, baseline and end of century

Panel (c) carries the result. The share plot in (d) shows only the marginals of
that table, so it cannot say where people came from. The transition matrix can,
and the largest single transition is the sentence the paper should quote.

Zones follow ASHRAE Standard 169, which is defined on CDD base 10 C and HDD base
18.3 C. Those bases differ deliberately from the 24 C demand base used elsewhere
in this paper: 24 C is India's mandated air conditioner setpoint and the right
anchor for demand, whereas the zone thresholds are only meaningful against the
standard's own bases.

A correction the manuscript must carry. ECBC and Eco-Niwas Samhita do NOT key on
ASHRAE zones. India's Energy Conservation Building Code uses the National
Building Code five zone scheme (Hot-Dry, Warm-Humid, Composite, Temperate,
Cold), which is defined on mean monthly temperature and mean monthly relative
humidity rather than on degree days. Computing those zones would require
humidity, which is not available for the products compared here. The ASHRAE
reclassification shown is therefore an indicator of code relevant thermal
pressure, not a restatement of ECBC zone membership, and Sections 4.6 and 5.3
must say so.

THE MISASSIGNMENT RECONCILIATION
--------------------------------
The manuscript reports that 1 degree fields place 18.1% of the population in the
wrong zone by end of century, while the resolution ladder in Figure 2d returns
about 5%. These measure different things, and this script computes both so the
discrepancy can be settled rather than argued about:

    resolution_error  coarse field vs fine field, SAME period. This is the
                      resolution effect, and it is what Figure 2d plots.

    warming_shift     fine field, baseline period vs future period. This is the
                      climate signal, not an error, and it is large.

    combined          coarse baseline zone vs fine future zone, which conflates
                      the two and is the most likely origin of 18.1%.

Whichever the original figure computed, the manuscript sentence must name it.
Reporting a warming shift as a resolution error would be a serious mistake.

Requires: preprocess_build_cache.py stage 7.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


COARSE_FACTOR = 10      # 1.0 degree


def zone_field(ds, factor, period):
    if factor is None:
        c = cc.climatology(ds.cdd10_fine, period)
        h = cc.climatology(ds.hdd18_fine, period)
    else:
        c = cc.climatology(ds.cdd10_coarse.sel(factor=factor), period)
        h = cc.climatology(ds.hdd18_coarse.sel(factor=factor), period)
    return cc.ashrae_zone(c, h)


def share_by_zone(zone, pop):
    tot = float(pop.sum())
    return {nm: 100 * float(pop.where(zone == i).sum()) / tot
            for i, nm in enumerate(cc.ZONE_ORDER)}


def main() -> None:
    cc.set_style()
    gdf = cc.load_boundary()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    zh = xr.open_dataset(cc.cache_path("zone_indices_historical.nc"))
    zf = xr.open_dataset(cc.cache_path("zone_indices_ssp585.nc"))

    pop = cc.load_population(cc.PATHS["POP_2010"], gdf,
                             expect_total=cc.CENSUS_2011_POP, label="WorldPop 2010")

    far = cc.FUTURE_PERIODS["far"]
    z_base_fine = zone_field(zh, None, cc.BASELINE)
    z_far_fine = zone_field(zf, None, far)
    z_base_coarse = zone_field(zh, COARSE_FACTOR, cc.BASELINE)
    z_far_coarse = zone_field(zf, COARSE_FACTOR, far)

    pop = pop.reindex_like(z_base_fine, method="nearest", tolerance=0.051)
    tot = float(pop.sum())

    sh_base = share_by_zone(z_base_fine, pop)
    sh_far = share_by_zone(z_far_fine, pop)

    print("\nPopulation share by zone (%)")
    print(f"  {'zone':<16} {'baseline':>9} {'2081-2100':>10}")
    for nm in cc.ZONE_ORDER:
        if sh_base[nm] > 0.05 or sh_far[nm] > 0.05:
            print(f"  {nm:<16} {sh_base[nm]:9.1f} {sh_far[nm]:10.1f}")

    # ---- the three misassignment measures --------------------------------
    def pct(mask):
        return 100 * float(pop.where(mask).sum()) / tot

    res_base = pct(z_base_coarse != z_base_fine)
    res_far = pct(z_far_coarse != z_far_fine)
    warming = pct(z_far_fine != z_base_fine)
    combined = pct(z_base_coarse != z_far_fine)
    hotter = pct(z_far_fine < z_base_fine)   # lower index is hotter

    print(f"\nMisassignment measures at {cc.factor_to_degrees(COARSE_FACTOR):.1f} degree")
    print(f"  resolution error, baseline period      {res_base:5.1f}%")
    print(f"  resolution error, 2081-2100            {res_far:5.1f}%")
    print(f"  warming shift, fine field              {warming:5.1f}%  "
          f"(climate signal, not an error)")
    print(f"    of which move to a HOTTER zone       {hotter:5.1f}%")
    print(f"  coarse baseline vs fine future         {combined:5.1f}%  "
          f"(conflates both)")
    print("\n  The manuscript reports 18.1%. Identify which of these it is and "
          "name it in the text. A warming shift reported as a resolution error "
          "would be a serious error.")

    pd.DataFrame([dict(
        resolution_error_baseline=res_base, resolution_error_far=res_far,
        warming_shift=warming, warming_shift_to_hotter=hotter,
        combined_coarse_baseline_vs_fine_future=combined,
        coarse_resolution_deg=cc.factor_to_degrees(COARSE_FACTOR),
    )]).to_csv(f"{cc.FIGDIR}/Jha_etal_Fig6_misassignment.csv",
               index=False, float_format="%.3f")

    pd.DataFrame({"zone": cc.ZONE_ORDER,
                  "baseline_pct": [sh_base[n] for n in cc.ZONE_ORDER],
                  "far_pct": [sh_far[n] for n in cc.ZONE_ORDER]}).to_csv(
        f"{cc.FIGDIR}/Jha_etal_Fig6_zone_shares.csv", index=False,
        float_format="%.3f")

    # ---- figure -----------------------------------------------------------
    present = [i for i, nm in enumerate(cc.ZONE_ORDER)
               if sh_base[nm] > 0.05 or sh_far[nm] > 0.05]
    names = [cc.ZONE_ORDER[i] for i in present]

    # Ordered hot to cold, so the palette itself carries the ordering.
    palette = plt.get_cmap("RdYlBu")(np.linspace(0.03, 0.92, len(present)))
    cmap = mcolors.ListedColormap(palette)
    norm = mcolors.BoundaryNorm(np.arange(len(present) + 1) - 0.5,
                                len(present))
    remap = {z: k for k, z in enumerate(present)}

    def to_plot(z):
        out = xr.full_like(z, np.nan)
        for orig, new in remap.items():
            out = xr.where(z == orig, float(new), out)
        return out.where(z.notnull())

    # ---- population transition matrix ------------------------------------
    zb = z_base_fine.values.ravel()
    zf = z_far_fine.values.ravel()
    pv = pop.values.ravel()
    ok = np.isfinite(zb) & np.isfinite(zf) & np.isfinite(pv)
    M, _, _ = np.histogram2d(zb[ok], zf[ok],
                             bins=np.arange(len(cc.ZONE_ORDER) + 1) - 0.5,
                             weights=pv[ok])
    M = M / 1e6                                  # millions
    Msub = M[np.ix_(present, present)]

    tm = pd.DataFrame(Msub, index=names, columns=names)
    tm.to_csv(f"{cc.FIGDIR}/Jha_etal_Fig6_transition_matrix.csv",
              float_format="%.2f")
    print("\nPopulation transition matrix (millions, rows baseline, cols future)")
    print(tm.round(1).to_string())

    off = Msub.copy()
    np.fill_diagonal(off, 0.0)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    print(f"\n  Largest single transition: {off[i, j]:.0f} million from "
          f"{names[i]} to {names[j]}. This is the sentence to quote.")

    fig = plt.figure(figsize=(7.5, 7.4))
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28,
                          left=0.075, right=0.975, top=0.945, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    for ax, z, title in ((ax_a, z_base_fine,
                          f"(a) Baseline {cc.BASELINE[0]}-{cc.BASELINE[1]}"),
                         (ax_b, z_far_fine,
                          f"(b) {far[0]}-{far[1]}, "
                          f"{cc.SCENARIO_LABEL['ssp585']}")):
        p = to_plot(z)
        ax.pcolormesh(p.lon, p.lat, p, cmap=cmap, norm=norm,
                      shading="auto", rasterized=True)
        cc.style_map(ax, gdf)
        cc.panel_label(ax, title)
        ax.set_xlabel("Longitude ($\\degree$E)")
    ax_a.set_ylabel("Latitude ($\\degree$N)")
    ax_b.set_yticklabels([])

    # (c) transition matrix
    shown = np.where(Msub > 0, Msub, np.nan)
    imt = ax_t.imshow(shown, cmap="rocket_r" if "rocket_r" in plt.colormaps()
                      else "YlGnBu", aspect="auto",
                      norm=mcolors.LogNorm(vmin=max(np.nanmin(shown), 0.05),
                                           vmax=np.nanmax(shown)))
    ax_t.set_xticks(range(len(names)))
    ax_t.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax_t.set_yticks(range(len(names)))
    ax_t.set_yticklabels(names, fontsize=7)
    ax_t.set_xlabel("Zone in 2081-2100")
    ax_t.set_ylabel("Zone in 1985-2014")
    cc.panel_label(ax_t, "(c) Population transition (millions)")
    for a in range(len(names)):
        for b_ in range(len(names)):
            v = Msub[a, b_]
            if v > 0.05:
                ax_t.text(b_, a, f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                          ha="center", va="center", fontsize=6.6,
                          color="white" if v > 0.25 * np.nanmax(shown) else "0.15",
                          fontweight="bold" if (a, b_) == (i, j) else "normal")
    ax_t.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                 edgecolor="#CC3311", linewidth=1.8))
    # cells above the diagonal are impossible under monotone warming
    ax_t.plot([-0.5, len(names) - 0.5], [-0.5, len(names) - 0.5],
              color="0.55", linewidth=0.7, linestyle=":")

    # (c) population share
    yy = np.arange(len(present))
    b = [sh_base[n] for n in names]
    f = [sh_far[n] for n in names]
    ax_c.barh(yy + 0.19, b, height=0.36, color=palette, edgecolor="0.3",
              linewidth=0.4, label="Baseline")
    ax_c.barh(yy - 0.19, f, height=0.36, color=palette, edgecolor="0.3",
              linewidth=0.4, hatch="///", label=f"{far[0]}-{far[1]}")
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels(names, fontsize=7.5)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Population share (%)")
    cc.panel_label(ax_c, "(d) Population by zone")
    ax_c.legend(frameon=False, fontsize=7, loc="lower right")
    for k, (bb, ff) in enumerate(zip(b, f)):
        if bb > 1:
            ax_c.text(bb + 1.2, k + 0.19, f"{bb:.1f}", fontsize=6.4, va="center")
        if ff > 1:
            ax_c.text(ff + 1.2, k - 0.19, f"{ff:.1f}", fontsize=6.4, va="center")
    ax_c.set_xlim(0, 108)
    ax_c.grid(axis="x", color="0.92", linewidth=0.5)
    ax_c.set_axisbelow(True)

    moved = sh_far["Extremely hot"] - sh_base["Extremely hot"]
    ax_c.text(0.98, 0.02,
              f"{moved:.0f} percentage points move\ninto the hottest zone",
              transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=6.8, color="0.35", linespacing=1.4)

    cc.save_figure(fig, "Jha_etal_Fig6")
    plt.close(fig)


if __name__ == "__main__":
    main()
