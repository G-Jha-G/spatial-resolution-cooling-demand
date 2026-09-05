"""
fig03_trajectories.py

Figure 3. National annual cooling degree days, 1985-2100, for the downscaled
product under three scenarios, with the driving ensemble bounding range.

Two things the caption must say, because both are easy to misread:

  The shaded band is a BOUNDING RANGE across the ten bias corrected driving
  members, not a confidence interval. It brackets structural disagreement among
  the driving models. It carries no probability statement, and it is not a
  spread from the generative model, which is evaluated in the companion study.

  IITM-ESM ends in 2099 for SSP2-4.5 and SSP5-8.5 and in 2098 for SSP3-7.0, so
  the final one or two years of each band rest on nine members rather than ten.
  Those years are marked. Nothing is padded or extrapolated. Because the missing
  years are the warmest, a short window biases that member's end of century mean
  low by roughly 0.2 to 0.5 percent of the level, which is negligible against a
  band spanning tens of percent.

The envelope is re-centred on the downscaled trajectory. The members and the
downscaled product differ in level by construction, since the downscaled field
is generated from the ensemble mean and the ensemble mean of a convex index is
not the mean of the members' indices. Plotting the raw member range against the
downscaled line would therefore show an offset that is an artefact of operator
ordering rather than a disagreement. The offset is computed once over an early
reference window, where the scenarios have not yet diverged, and applied as a
constant. Both the raw and the re-centred band are written to the CSV so the
choice is auditable.

Requires: preprocess_build_cache.py stages 1 and 5.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


ROLL = 10                       # rolling mean window, years
RECENTRE_WINDOW = (2015, 2044)  # scenarios have barely diverged here
FAR = cc.FUTURE_PERIODS["far"]


def national_series(annual: xr.DataArray) -> xr.DataArray:
    """Area weighted national annual mean of a gridded annual index."""
    w = np.cos(np.deg2rad(annual.lat)).broadcast_like(annual.isel(year=0))
    w = w.where(annual.isel(year=0).notnull())
    return (annual * w).sum(dim=("lat", "lon")) / w.sum()


def main() -> None:
    cc.set_style()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    print("Building national trajectories from the downscaled product")
    hist = national_series(
        cc.read_cache("annual_cdd_ddpm_historical.nc", var="cdd")).compute()
    fut = {s: national_series(
        cc.read_cache(f"annual_cdd_ddpm_{s}.nc", var="cdd")).compute()
        for s in cc.SCENARIOS}

    base_mean = float(hist.sel(year=slice(*cc.BASELINE)).mean())
    base_sd = float(hist.sel(year=slice(*cc.BASELINE)).std(ddof=1))
    lo, hi = cc.block_bootstrap_ci(
        hist.sel(year=slice(*cc.BASELINE)).values, block=3)
    print(f"  baseline {cc.BASELINE[0]}-{cc.BASELINE[1]}: "
          f"{base_mean:.1f} degC days, interannual SD {base_sd:.1f}, "
          f"bootstrap 95% CI {lo:.1f} to {hi:.1f}")

    print("\nLoading the per member envelope")
    mem = xr.open_dataset(cc.cache_path("national_annual_cdd_by_member.nc"))

    rows, bands = [], {}
    for s in cc.SCENARIOS:
        d = mem.national_cdd.sel(scenario=s)
        n = mem.n_members.sel(scenario=s)

        # Re-centre on the downscaled trajectory using an early window.
        w0, w1 = RECENTRE_WINDOW
        offset = (float(fut[s].sel(year=slice(w0, w1)).mean())
                  - float(d.sel(year=slice(w0, w1)).mean()))

        raw_lo = d.min(dim="model")
        raw_hi = d.max(dim="model")
        bands[s] = dict(lo=raw_lo + offset, hi=raw_hi + offset,
                        n=n, offset=offset,
                        short_years=n.where(n < len(cc.MODELS), drop=True).year.values)

        far = fut[s].sel(year=slice(*FAR))
        far_mem = d.sel(year=slice(*FAR)).mean(dim="year")
        pct = 100 * (float(far.mean()) / base_mean - 1)
        pct_lo = 100 * ((float(far_mem.min()) + offset) / base_mean - 1)
        pct_hi = 100 * ((float(far_mem.max()) + offset) / base_mean - 1)
        slope = cc.theil_sen(fut[s].year.values.astype(float), fut[s].values)

        rows.append(dict(scenario=s, offset=offset,
                         far_mean=float(far.mean()),
                         pct_change=pct, bound_lo_pct=pct_lo, bound_hi_pct=pct_hi,
                         theil_sen_per_decade=10 * slope,
                         short_years=",".join(str(int(y))
                                              for y in bands[s]["short_years"])))
        print(f"  {cc.SCENARIO_LABEL[s]}: offset {offset:+.1f}, "
              f"{FAR[0]}-{FAR[1]} mean {float(far.mean()):.0f} "
              f"({pct:+.1f}%), bounding {pct_lo:+.0f} to {pct_hi:+.0f}%, "
              f"trend {10*slope:+.1f} per decade")

    df = pd.DataFrame(rows)
    csv = f"{cc.FIGDIR}/Jha_etal_Fig3_trajectories.csv"
    df.to_csv(csv, index=False, float_format="%.3f")
    print(f"\nTrajectory table -> {csv}")

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    # Concatenate history with each scenario BEFORE smoothing, otherwise the
    # rolling window leaves a five year gap either side of the 2014/2015 join.
    joined = {s: xr.concat([hist, fut[s]], dim="year") for s in cc.SCENARIOS}
    h_roll = joined[cc.SCENARIOS[0]].rolling(
        year=ROLL, center=True, min_periods=ROLL).mean().sel(
        year=slice(cc.BASELINE[0], 2014))
    ax.plot(h_roll.year, h_roll, color="0.25", linewidth=1.8, zorder=6,
            label="Historical")
    ax.plot(hist.year, hist, color="0.25", linewidth=0.5, alpha=0.30, zorder=2)

    for s in cc.SCENARIOS:
        col = cc.SCENARIO_COLOR[s]
        b = bands[s]
        ax.fill_between(b["lo"].year, b["lo"], b["hi"], color=col, alpha=0.16,
                        linewidth=0, zorder=1)
        roll = joined[s].rolling(year=ROLL, center=True,
                                 min_periods=ROLL).mean().sel(
            year=slice(2010, 2100))
        ax.plot(roll.year, roll, color=col, linewidth=1.8, zorder=6,
                label=cc.SCENARIO_LABEL[s])
        ax.plot(fut[s].year, fut[s], color=col, linewidth=0.5, alpha=0.28, zorder=2)

        if len(b["short_years"]):
            y0 = int(np.min(b["short_years"]))
            ax.axvspan(y0 - 0.5, 2100.5, color="0.5", alpha=0.06,
                       linewidth=0, zorder=0)

    ax.axhspan(base_mean - 2 * base_sd, base_mean + 2 * base_sd,
               color="0.6", alpha=0.13, linewidth=0, zorder=0)
    ax.axhline(base_mean, color="0.35", linewidth=0.8, linestyle="--", zorder=3)
    ax.text(1986, base_mean + 0.12 * base_sd,
            f"baseline {base_mean:.0f} $\\pm$ 2$\\sigma$",
            fontsize=7.5, color="0.35", va="bottom")

    all_short = sorted({int(y) for s in cc.SCENARIOS
                        for y in bands[s]["short_years"]})
    if all_short:
        ax.annotate(f"9 of 10 members\nfrom {min(all_short)}",
                    xy=(min(all_short), ax.get_ylim()[1]),
                    xytext=(-6, -6), textcoords="offset points",
                    fontsize=6.2, color="0.45", va="top", ha="right",
                    linespacing=1.3)

    ax.set_xlabel("Year")
    ax.set_ylabel("National annual CDD ($\\degree$C days, base 24 $\\degree$C)")
    ax.set_xlim(1985, 2100)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.text(0.985, 0.03,
            "Shaded bands: bounding range across ten driving members,\n"
            "re-centred on the downscaled trajectory. Not a confidence interval.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.8, color="0.35", linespacing=1.4)

    fig.tight_layout()
    cc.save_figure(fig, "Jha_etal_Fig3")
    plt.close(fig)


if __name__ == "__main__":
    main()
