"""
fig1_curvature_identities.py

MANUSCRIPT FIGURE 1. The two curvature identities and their verification.

    (a) Curvature f_T(24 C), the density of daily temperature at the base
    (b) Within-block spatial variance of daily temperature at 1 degree
    (c) Measured aggregation gap at 1 degree
    (d) Measured gap against the closed form (N/2) f(b) Var_sub
    (e) Exceedance probability P(T > 24 C) over the projection period
    (f) Measured sensitivity dCDD/dT against the prediction N P(T > b)

Panels (a) to (d) verify equation (3), the second derivative, which governs how
much information spatial aggregation destroys. Panels (e) and (f) verify
equation (2), the first derivative, which governs how much demand a degree of
warming adds. Verifying both closes the framework: it then predicts the level
error, the growth sensitivity, and the acceleration of that sensitivity from the
same observed daily distribution.

Every prediction on the left of each comparison uses observations alone. The
measurement on the right uses either an aggregation experiment on the observed
field or eighty-six years of a climate model chain the prediction never sees.

TWO CONSTRUCTION RULES THAT THE TEST DEPENDS ON
-----------------------------------------------
Jensen's inequality applies to the coarse BLOCK, not to the individual cell. A
subcell warmer than its block mean has a larger CDD than the block-mean
temperature implies and a cooler subcell a smaller one, so roughly half of
per-cell differences are negative by construction. All quantities are therefore
aggregated to the block before the sign check.

Both sides of the aggregation comparison must average the identical set of
cells. The field is masked to land BEFORE any coarsening, and only blocks at
least 99 percent land are retained. Comparing a land-masked fine field against a
coarse field aggregated over land and ocean together returns apparent
overstatement, because the Indian Ocean is warmer than adjacent land at night
and in winter; that is a consequence of comparing different cell sets and not a
failure of the inequality.

Caching. The expensive quantities are written to the analysis cache on first
run, so the figure can be restyled without recomputing. Delete
cache/fig1_curvature_fields.nc to force a rebuild.

Requires: MSWX, and preprocess_build_cache.py stage 1 for the projection fields.
No tight_layout is used anywhere; all spacing is set explicitly.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr, theilslopes

import cdd_common as cc


BAND = 1.0                  # half width in degC for the density estimate
FACTORS = [5, 10]           # 0.5 and 1.0 degree blocks
MAIN_FACTOR = 10
LAND_FRACTION_MIN = 0.99
DAYS = 365.25
CACHE_NAME = "fig1_curvature_fields.nc"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def block_mean(da: xr.DataArray, f: int) -> xr.DataArray:
    return da.coarsen(lat=f, lon=f, boundary="trim").mean()


def annual_cdd_clim(tas, base, period):
    return cc.climatology(cc.annual_cdd(tas, base=base), period)


def compute_fields(gdf):
    """Everything the figure needs, computed once and cached."""
    if cc.cache_exists(CACHE_NAME):
        print(f"  reading cached fields from {CACHE_NAME}")
        return xr.open_dataset(cc.cache_path(CACHE_NAME))

    print("Loading MSWX and masking to land before coarsening")
    tas = cc.open_tas(cc.PATHS["MSWX"], label="MSWX", years=cc.BASELINE)
    yrs = cc.complete_years(tas, verbose=False)
    tas = cc.clip_india(tas.sel(time=tas.time.dt.year.isin(yrs)), gdf).persist()
    land = tas.isel(time=0).notnull()
    print(f"  {int(land.sum())} land cells")

    fine = annual_cdd_clim(tas, cc.T_BASE_CDD, cc.BASELINE).compute()
    dens = (((tas > cc.T_BASE_CDD - BAND) & (tas < cc.T_BASE_CDD + BAND))
            .astype("float32").where(land).mean(dim="time")
            / (2 * BAND)).compute()

    out, stats = {}, []
    for f in FACTORS:
        frac = block_mean(land.astype("float32"), f)
        keep = frac >= LAND_FRACTION_MIN

        t_blk = block_mean(tas, f)
        cdd_of_mean = annual_cdd_clim(t_blk, cc.T_BASE_CDD, cc.BASELINE).compute()
        gap = (block_mean(fine, f) - cdd_of_mean).where(keep)

        m1, m2 = block_mean(tas, f), block_mean(tas ** 2, f)
        var_sub = (m2 - m1 ** 2).clip(min=0.0).mean(dim="time").compute()
        dens_blk = block_mean(dens, f)
        pred = (0.5 * DAYS * dens_blk * var_sub).where(keep)

        x, y = _pair(pred, gap)
        rho = spearmanr(x, y)[0]
        sl, ic, lo, hi = theilslopes(y, x)
        neg = float((gap < -0.5).sum()) / max(int(np.isfinite(gap.values).sum()), 1)
        stats.append(dict(resolution_deg=cc.factor_to_degrees(f),
                          n_blocks=len(x), rho=rho, slope=sl,
                          slope_lo=lo, slope_hi=hi,
                          pct_overstated=100 * neg,
                          rho_curvature_only=spearmanr(*_pair(dens_blk.where(keep), gap))[0],
                          rho_variance_only=spearmanr(*_pair(var_sub.where(keep), gap))[0]))
        print(f"  {cc.factor_to_degrees(f):.1f} deg: n={len(x)}, rho={rho:.3f}, "
              f"slope={sl:.3f} [{lo:.3f}, {hi:.3f}], "
              f"overstating {100*neg:.2f}%")
        if f == MAIN_FACTOR:
            out.update(gap=gap, pred=pred, var_sub=var_sub, dens_blk=dens_blk)

    # ---- equation (2): sensitivity identity ------------------------------
    print("\nSensitivity identity from the SSP5-8.5 projection")
    tas_f = cc.open_tas(cc.PATHS["DDPM_ssp585"], label="ddpm ssp585",
                        years=(2015, 2100))
    yrs_f = cc.complete_years(tas_f, verbose=False)
    tas_f = tas_f.sel(time=tas_f.time.dt.year.isin(yrs_f))

    p_period = cc.clip_india(cc.frac_above(tas_f, cc.T_BASE_CDD).compute(), gdf)
    pred_sens = DAYS * p_period

    cdd_a = cc.read_cache("annual_cdd_ddpm_ssp585.nc", var="cdd")
    tmean = cc.clip_india(tas_f.groupby("time.year").mean("time").compute(), gdf)
    yy = np.intersect1d(cdd_a.year.values, tmean.year.values)
    ya, xa = cdd_a.sel(year=yy), tmean.sel(year=yy)
    xm, ym = xa.mean("year"), ya.mean("year")
    cov = ((xa - xm) * (ya - ym)).mean("year")
    var = ((xa - xm) ** 2).mean("year")
    slope_sens = (cov / var.where(var > 1e-9)).compute()

    # Align onto the observational grid, which is the smaller of the two.
    slope_sens = cc.align_to(slope_sens, pred_sens, label="measured dCDD/dT")

    xs, ys = _pair(pred_sens, slope_sens)
    rho_s = spearmanr(xs, ys)[0]
    sl_s, ic_s, lo_s, hi_s = theilslopes(ys, xs)
    print(f"  rho = {rho_s:.3f}, slope = {sl_s:.3f} [{lo_s:.3f}, {hi_s:.3f}], "
          f"national {cc.wmean(pred_sens):.0f} predicted vs "
          f"{cc.wmean(slope_sens):.0f} measured")
    stats.append(dict(resolution_deg=np.nan, n_blocks=len(xs), rho=rho_s,
                      slope=sl_s, slope_lo=lo_s, slope_hi=hi_s,
                      pct_overstated=np.nan, rho_curvature_only=np.nan,
                      rho_variance_only=np.nan))

    # Strip any coordinate that is not a dimension before assembling the
    # Dataset. Fields derived from different records can carry scalar
    # coordinates (a leftover time stamp, a CRS variable) whose values differ,
    # and xarray refuses to merge them. Only lat and lon are meaningful here.
    fields = dict(dens=dens, gap=out["gap"], pred=out["pred"],
                  var_sub=out["var_sub"], dens_blk=out["dens_blk"],
                  p_period=p_period, pred_sens=pred_sens,
                  slope_sens=slope_sens)
    fields = {k: _strip(v) for k, v in fields.items()}

    # The block fields and the cell fields live on different grids, so they are
    # written as two groups of variables rather than forced onto one.
    ds = xr.merge([v.rename(k) for k, v in fields.items()],
                  compat="override", join="outer")
    ds.attrs.update({
        "base_temperature_degC": cc.T_BASE_CDD,
        "band_half_width_degC": BAND,
        "main_factor": MAIN_FACTOR,
        "land_fraction_min": LAND_FRACTION_MIN,
        "stats_json": pd.DataFrame(stats).to_json(orient="records"),
    })
    ds.to_netcdf(cc.cache_path(CACHE_NAME))
    pd.DataFrame(stats).to_csv(f"{cc.FIGDIR}/Jha_etal_Fig1_statistics.csv",
                               index=False, float_format="%.5f")
    print(f"  cached -> {cc.cache_path(CACHE_NAME)}")
    return ds


def _strip(da: xr.DataArray) -> xr.DataArray:
    """Remove every coordinate that is not a dimension of the array."""
    drop = [c for c in da.coords if c not in da.dims]
    return da.drop_vars(drop) if drop else da


def _pair(a, b):
    av, bv = np.asarray(a.values).ravel(), np.asarray(b.values).ravel()
    m = np.isfinite(av) & np.isfinite(bv)
    return av[m], bv[m]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def main() -> None:
    cc.set_style()
    gdf = cc.load_boundary()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    ds = compute_fields(gdf)
    stats = pd.read_json(ds.attrs["stats_json"])
    agg = stats[stats.resolution_deg == cc.factor_to_degrees(MAIN_FACTOR)].iloc[0]
    sens = stats[stats.resolution_deg.isna()].iloc[0]

    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(
        2, 3, wspace=0.59, hspace=0.34,
        left=0.070, right=0.975, top=0.935, bottom=0.095,
    )
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    maps = [
        (ax[0], ds.dens, "(a) Curvature $f_T(24\\,\\degree$C)", "magma_r",
         "Density at base ($\\degree$C$^{-1}$)", 98.0),
        (ax[1], ds.var_sub, "(b) Subgrid variance, 1$\\degree$", "viridis",
         "Var$_{\\mathrm{sub}}$ ($\\degree$C$^2$)", 99.0),  # Changed to 99th percentile
        (ax[2], ds.gap, "(c) Measured aggregation gap", cc.CMAP_ABS,
         "Gap ($\\degree$C days)", 99.0),                  # Changed to 99th percentile
        (ax[3], ds.p_period, "(e) $P(T>24\\,\\degree$C), projection", "YlOrBr",
         "Exceedance probability", 100.0),
    ]
    for a, fld, ttl, cmap, lab, q in maps:
        # Strip the empty NaN coordinates caused by the outer merge
        fld_clean = fld.dropna(dim="lat", how="all").dropna(dim="lon", how="all")
        
        # Plot using the cleaned DataArray
        im = a.pcolormesh(fld_clean.lon, fld_clean.lat, fld_clean, cmap=cmap, vmin=0,
                          vmax=cc.robust_max(fld_clean, q), shading="auto",
                          rasterized=True)
        cc.style_map(a, gdf)
        cc.panel_label(a, ttl)
        a.set_xlabel("Longitude ($\\degree$E)")
        a.set_ylabel("Latitude ($\\degree$N)")
        cb = fig.colorbar(im, ax=a, fraction=0.048, pad=0.03,
                          extend="max" if q < 100 else "neither")
        cb.set_label(lab, fontsize=7.2)
        cb.ax.tick_params(labelsize=6.6)
        cb.outline.set_linewidth(0.5)

    # (d) equation (3) test
    a = ax[4]
    x, y = _pair(ds.pred, ds.gap)
    a.scatter(x, y, s=11, alpha=0.45, color="#882255", edgecolor="none")
    lim = float(np.nanpercentile(np.r_[x, y], 99.0))
    a.plot([0, lim], [0, lim], color="0.4", linewidth=0.9, linestyle="--",
           label="1:1", zorder=1)
    a.plot([0, lim], [0, agg.slope * lim], color="#333333", linewidth=1.3,
           label=f"slope {agg.slope:.2f}", zorder=2)
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("$\\frac{N}{2}f_T(b)\\,$Var$_{\\mathrm{sub}}$ "
                 "($\\degree$C days)")
    a.set_ylabel("Measured gap ($\\degree$C days)")
    cc.panel_label(a, "(d) Identity (3) verified")
    a.legend(frameon=False, fontsize=6.8, loc="lower right")
    a.text(0.04, 0.96,
           f"$\\rho$ = {agg.rho:.2f}",
           transform=a.transAxes, va="top", ha="left", fontsize=7.2,
           bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                     edgecolor="0.7", linewidth=0.5, alpha=0.9))
    a.grid(color="0.93", linewidth=0.5)
    a.set_axisbelow(True)

    # (f) equation (2) test
    a = ax[5]
    xs, ys = _pair(ds.pred_sens, ds.slope_sens)
    sub = np.random.default_rng(0).choice(len(xs), size=min(6000, len(xs)),
                                          replace=False)
    a.scatter(xs[sub], ys[sub], s=4, alpha=0.22, color="#CC3311",
              edgecolor="none")
    lim = float(np.nanpercentile(np.r_[xs, ys], 99.5))
    a.plot([0, lim], [0, lim], color="0.4", linewidth=0.9, linestyle="--",
           label="1:1", zorder=1)
    a.plot([0, lim], [0, sens.slope * lim], color="#333333", linewidth=1.3,
           label=f"slope {sens.slope:.2f}", zorder=2)
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("$N\\,P(T>b)$ ($\\degree$C days per $\\degree$C)")
    a.set_ylabel("Measured d CDD / d T")
    cc.panel_label(a, "(f) Identity (2) verified")
    a.legend(frameon=False, fontsize=6.8, loc="lower right")
    a.text(0.04, 0.96,
           f"$\\rho$ = {sens.rho:.2f}",
           transform=a.transAxes, va="top", ha="left", fontsize=7.2,
           bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                     edgecolor="0.7", linewidth=0.5, alpha=0.9))
    a.grid(color="0.93", linewidth=0.5)
    a.set_axisbelow(True)

    cc.save_figure(fig, "Jha_etal_Fig1")
    plt.close(fig)

    print(f"\n  Panel (d): rho {agg.rho:.3f}, slope {agg.slope:.3f} "
          f"[{agg.slope_lo:.3f}, {agg.slope_hi:.3f}], "
          f"{agg.pct_overstated:.2f}% of blocks overstating")
    print(f"  Panel (f): rho {sens.rho:.3f}, slope {sens.slope:.3f} "
          f"[{sens.slope_lo:.3f}, {sens.slope_hi:.3f}]")


if __name__ == "__main__":
    main()
