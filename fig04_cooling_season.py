"""
fig04_cooling_season.py

Figure 4. The cooling season lengthens and deepens.

    (a) Hovmoller of national daily exceedance above the demand base, by day of
        year and calendar year, historical concatenated with SSP5-8.5.
    (b) Season length by year for all three scenarios, with the onset and
        cessation dates that define it.
    (c) Onset and cessation dates, baseline against end of century.

Season definition, and a trap worth stating. A day counts as a cooling day when
the national area weighted daily MEAN TEMPERATURE exceeds the base. It would be
wrong to define it on the national mean of daily exceedance, because that
quantity is positive whenever any single cell in India exceeds the base, which
is true on every day of the year; defining a season that way returns 366 days
in every year, which is what a first version of this script did.

Onset is the first cooling day of the calendar year and cessation the last, so
length is cessation minus onset plus one. This is an envelope definition and
does not require continuity, which matters in India where the monsoon interrupts
the season across much of the country. That interruption is visible in panel (a)
and should be described in the caption rather than defined away. Panel (b)
therefore also reports the count of cooling days, which is insensitive to the
envelope convention, alongside the envelope length.

Requires: preprocess_build_cache.py stage 3.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


MAIN_SCENARIO = "ssp585"
SMOOTH = 5      # days, for onset and cessation only, to suppress single day flicker


def season_stats(tas_cube: xr.DataArray, exc_cube: xr.DataArray,
                 base: float = cc.T_BASE_CDD) -> pd.DataFrame:
    """
    Onset, cessation, envelope length and cooling day count per calendar year.

    The season is defined on national mean temperature above the base. The
    exceedance cube is used only for the annual total, not for the season.
    """
    rows = []
    for y in tas_cube.year.values:
        t = tas_cube.sel(year=y)
        v = t.values
        if not np.isfinite(v).any():
            continue
        sm = pd.Series(v).rolling(SMOOTH, center=True, min_periods=1).mean().values
        hot = np.where(sm > base)[0]
        n_days = int(np.nansum(v > base))
        total = float(np.nansum(exc_cube.sel(year=y).values))
        if hot.size == 0:
            rows.append(dict(year=int(y), onset=np.nan, cessation=np.nan,
                             length=0, n_days=n_days, total=total))
            continue
        onset = int(t.doy.values[hot[0]])
        cess = int(t.doy.values[hot[-1]])
        rows.append(dict(year=int(y), onset=onset, cessation=cess,
                         length=cess - onset + 1, n_days=n_days, total=total))
    return pd.DataFrame(rows)


def main() -> None:
    cc.set_style()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    print("Loading national daily cubes")
    dh = xr.open_dataset(cc.cache_path("national_doy_exceedance_historical.nc"))
    dfs = {s: xr.open_dataset(cc.cache_path(f"national_doy_exceedance_{s}.nc"))
           for s in cc.SCENARIOS}

    if "tas_national" not in dh:
        raise KeyError(
            "national_doy_exceedance_*.nc does not contain 'tas_national'. "
            "Rerun stage 3 with the patched preprocess_build_cache.py: the "
            "season cannot be defined from exceedance alone."
        )

    hist = dh.exceedance.sel(year=slice(*cc.BASELINE))
    hist_t = dh.tas_national.sel(year=slice(*cc.BASELINE))
    fut = {s: dfs[s].exceedance for s in cc.SCENARIOS}
    fut_t = {s: dfs[s].tas_national for s in cc.SCENARIOS}

    stats = {"historical": season_stats(hist_t, hist)}
    for s in cc.SCENARIOS:
        stats[s] = season_stats(fut_t[s], fut[s])

    base = stats["historical"]
    b_len = base.length.mean()
    b_on = base.onset.mean()
    b_off = base.cessation.mean()
    print(f"\nBaseline {cc.BASELINE[0]}-{cc.BASELINE[1]}: onset day {b_on:.0f}, "
          f"cessation day {b_off:.0f}, envelope {b_len:.0f} days, "
          f"{base.n_days.mean():.0f} cooling days")
    if b_len > 360:
        print("  WARNING: the envelope spans the whole year, so onset and "
              "cessation are not informative. Report the cooling day count "
              "instead, and check the season definition.")

    far = cc.FUTURE_PERIODS["far"]
    rows = []
    for s in cc.SCENARIOS:
        d = stats[s][(stats[s].year >= far[0]) & (stats[s].year <= far[1])]
        rows.append(dict(scenario=s, onset=d.onset.mean(),
                         cessation=d.cessation.mean(), length=d.length.mean(),
                         n_days=d.n_days.mean(),
                         d_length=d.length.mean() - b_len,
                         d_n_days=d.n_days.mean() - base.n_days.mean(),
                         d_onset=d.onset.mean() - b_on,
                         d_cessation=d.cessation.mean() - b_off,
                         total=d.total.mean()))
        r = rows[-1]
        print(f"  {cc.SCENARIO_LABEL[s]}: envelope {r['length']:.0f} days "
              f"({r['d_length']:+.0f}), cooling days {r['n_days']:.0f} "
              f"({r['d_n_days']:+.0f}), onset {r['d_onset']:+.0f}, "
              f"cessation {r['d_cessation']:+.0f}")

    df = pd.DataFrame(rows)
    csv = f"{cc.FIGDIR}/Jha_etal_Fig4_season.csv"
    df.to_csv(csv, index=False, float_format="%.2f")
    print(f"\nSeason table -> {csv}")

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(7.5, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0],
                          wspace=0.30, hspace=0.34,
                          left=0.085, right=0.90, top=0.945, bottom=0.075)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # (a) Hovmoller
    joined = xr.concat([hist, fut[MAIN_SCENARIO]], dim="year")
    arr = joined.transpose("doy", "year")
    im = ax_a.pcolormesh(arr.year, arr.doy, arr.values, cmap="YlOrRd",
                         vmin=0, vmax=cc.robust_max(arr, 99.0),
                         shading="auto", rasterized=True)
    ax_a.axvline(2014.5, color="0.25", linewidth=0.8, linestyle="--")
    ax_a.text(2015.5, 358, cc.SCENARIO_LABEL[MAIN_SCENARIO], fontsize=7,
              color="0.25", va="top")
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Day of year")
    ax_a.set_ylim(1, 365)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax_a.set_yticks(month_starts)
    ax_a.set_yticklabels(list("JFMAMJJASOND"))
    cc.panel_label(ax_a, "(a) National daily exceedance")
    cb = fig.colorbar(im, ax=ax_a, fraction=0.040, pad=0.02, extend="max")
    cb.set_label(f"Exceedance above {cc.T_BASE_CDD:g} $\\degree$C "
                 f"($\\degree$C)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)

    # (b) season length
    for s in cc.SCENARIOS:
        d = stats[s]
        r = pd.Series(d.n_days.values, index=d.year.values).rolling(
            10, center=True, min_periods=10).mean()
        ax_b.plot(r.index, r.values, color=cc.SCENARIO_COLOR[s], linewidth=1.6,
                  label=cc.SCENARIO_LABEL[s])
    rh = pd.Series(base.n_days.values, index=base.year.values).rolling(
        10, center=True, min_periods=10).mean()
    ax_b.plot(rh.index, rh.values, color="0.25", linewidth=1.6,
              label="Historical")
    ax_b.axhline(base.n_days.mean(), color="0.45", linewidth=0.8, linestyle="--")
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("Cooling days per year")
    ax_b.set_xlim(1985, 2100)
    cc.panel_label(ax_b, "(b) Cooling days per year")
    ax_b.legend(frameon=False, fontsize=7, loc="upper left")

    # (c) onset and cessation shift
    y = np.arange(len(cc.SCENARIOS))
    for k, s in enumerate(cc.SCENARIOS):
        r = df[df.scenario == s].iloc[0]
        ax_c.plot([r.onset, r.cessation], [k, k], color=cc.SCENARIO_COLOR[s],
                  linewidth=4, solid_capstyle="round", zorder=3)
    ax_c.plot([b_on, b_off], [len(cc.SCENARIOS), len(cc.SCENARIOS)],
              color="0.35", linewidth=4, solid_capstyle="round", zorder=3)
    ax_c.set_yticks(list(y) + [len(cc.SCENARIOS)])
    ax_c.set_yticklabels([cc.SCENARIO_LABEL[s] for s in cc.SCENARIOS]
                         + [f"Baseline\n{cc.BASELINE[0]}-{cc.BASELINE[1]}"],
                         fontsize=7.5)
    ax_c.set_xticks(month_starts)
    ax_c.set_xticklabels(list("JFMAMJJASOND"))
    ax_c.set_xlim(1, 365)
    ax_c.set_xlabel("Day of year")
    ax_c.invert_yaxis()
    cc.panel_label(ax_c, "(c) Onset and cessation, 2081-2100")
    ax_c.grid(axis="x", color="0.9", linewidth=0.5)
    ax_c.set_axisbelow(True)

    cc.save_figure(fig, "Jha_etal_Fig4")
    plt.close(fig)

    print("\nCaption note: the season is an envelope from first to last cooling "
          "day and does not require continuity. Monsoon interruption is visible "
          "in panel (a) and should be described rather than defined away.")


if __name__ == "__main__":
    main()
