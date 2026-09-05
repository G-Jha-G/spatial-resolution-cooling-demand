#!/usr/bin/env python3
"""
Resolution uncertainty versus CMIP6 driving-model uncertainty
for national cooling degree days over India.

This analysis is intentionally paired by driving model. For every model and
scenario, the script computes CDD from the same 0.1-degree daily QDM field
and from the same field after daily temperature aggregation to 1 degree.
Thus the resolution effect is separated from the model-to-model spread.

Outputs
-------
1. CSV with model-level and summary results.
2. Publication-quality PNG/PDF figure.

Main quantities
---------------
Baseline level:
    resolution effect = CDD_1deg - CDD_0p1deg

End-century level:
    resolution effect = CDD_1deg - CDD_0p1deg

Projected growth:
    growth_0p1 = CDD_far_0p1 - CDD_base_0p1
    growth_1deg = CDD_far_1deg - CDD_base_1deg
    resolution effect on growth = growth_1deg - growth_0p1

Climate-model uncertainty is represented by the standard deviation across
the ten driving models of the 0.1-degree result. The figure also reports the
full model range.

IMPORTANT
---------
The daily field is coarsened BEFORE applying the CDD threshold. Coarsening
annual CDD would not represent a coarse climate model and would remove the
nonlinearity that this paper studies.

The national mean is cosine-latitude weighted. India is clipped only after
the CDD index is calculated, following cdd_common.py.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import the manuscript's common analysis foundation.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cdd_common import (
    BASELINE,
    FUTURE_PERIODS,
    MODELS,
    SCENARIOS,
    SCENARIO_LABEL,
    T_BASE_CDD,
    INDIA_BBOX,
    member_path,
    open_tas,
    annual_cdd,
    block_coarsen,
    load_boundary,
    clip_india,
    wmean,
    set_style,
    save_figure,
    FIGDIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------
COARSEN_FACTOR = 10                  # 0.1° x 10 = 1.0°
FUTURE = FUTURE_PERIODS["far"]       # 2081-2100
BASE = BASELINE                       # 1985-2014

OUT_CSV = os.path.join(
    FIGDIR, "Jha_etal_resolution_vs_model_uncertainty.csv"
)
FIG_STEM = "Jha_etal_resolution_vs_model_uncertainty"

# Set True for a quick test using only the first two models.
# Leave False for the manuscript result.
TEST_MODE = False
TEST_MODELS = MODELS[:2]


def national_cdd_for_period(tas, period):
    """Area-weighted national CDD after the index has been calculated."""
    cdd = annual_cdd(tas, base=T_BASE_CDD)
    cdd = cdd.sel(year=slice(period[0], period[1])).mean("year")
    return wmean(cdd)


def calculate_model(model: str, scenario: str) -> dict | None:
    """
    Calculate paired 0.1° and 1° national CDD for one model/scenario.

    The same daily QDM temperature field is used for both resolutions.
    """
    path = member_path(model, scenario)
    if path is None:
        return None

    print(f"\n{'=' * 72}")
    print(f"{model} | {SCENARIO_LABEL[scenario]}")
    print(f"{'=' * 72}")
    print(f"  file: {path}")

    # Load the complete period needed by this comparison. Historical members
    # are used only for the common 1985-2014 baseline; scenario files supply
    # 2015-2100. For a scenario member, we therefore need both files.
    hist_path = member_path(model, "historical")
    if hist_path is None:
        print(f"  missing historical member: {model}")
        return None

    # Historical baseline.
    tas_hist = open_tas(
        hist_path,
        label=f"{model} historical",
        bbox=INDIA_BBOX,
        years=BASE,
        chunks={"time": 730},
    )

    # Scenario future.
    tas_scen = open_tas(
        path,
        label=f"{model} {scenario}",
        bbox=INDIA_BBOX,
        years=FUTURE,
        chunks={"time": 730},
    )

    # -----------------------------------------------------------------------
    # 0.1°: calculate CDD at native resolution.
    # -----------------------------------------------------------------------
    print("  calculating native 0.1° CDD...")
    base_native = national_cdd_for_period(tas_hist, BASE)
    far_native = national_cdd_for_period(tas_scen, FUTURE)

    # -----------------------------------------------------------------------
    # 1°: aggregate DAILY temperature first, then apply threshold.
    # This is the critical operation for the resolution experiment.
    # -----------------------------------------------------------------------
    print("  coarsening daily temperature to 1°...")
    tas_hist_1deg = block_coarsen(tas_hist, COARSEN_FACTOR)
    tas_scen_1deg = block_coarsen(tas_scen, COARSEN_FACTOR)

    print("  calculating 1° CDD...")
    base_1deg = national_cdd_for_period(tas_hist_1deg, BASE)
    far_1deg = national_cdd_for_period(tas_scen_1deg, FUTURE)

    # Paired resolution effects.
    level_res_base = base_1deg - base_native
    level_res_far = far_1deg - far_native

    growth_native = far_native - base_native
    growth_1deg = far_1deg - base_1deg
    growth_res = growth_1deg - growth_native

    return {
        "model": model,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABEL[scenario],
        "baseline_native_0p1": base_native,
        "baseline_1deg": base_1deg,
        "far_native_0p1": far_native,
        "far_1deg": far_1deg,
        "growth_native_0p1": growth_native,
        "growth_1deg": growth_1deg,
        "resolution_effect_baseline": level_res_base,
        "resolution_effect_far": level_res_far,
        "resolution_effect_growth": growth_res,
        "resolution_error_baseline_pct": 100.0 * level_res_base / base_native,
        "resolution_error_far_pct": 100.0 * level_res_far / far_native,
        "resolution_error_growth_pct": (
            100.0 * growth_res / growth_native
            if growth_native != 0 else np.nan
        ),
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise model spread and paired resolution effects by scenario."""
    rows = []

    for scenario in SCENARIOS:
        d = results.loc[results["scenario"] == scenario].copy()
        if d.empty:
            continue

        for quantity, native_col, coarse_col, res_col in [
            (
                "baseline_level",
                "baseline_native_0p1",
                "baseline_1deg",
                "resolution_effect_baseline",
            ),
            (
                "far_level",
                "far_native_0p1",
                "far_1deg",
                "resolution_effect_far",
            ),
            (
                "growth",
                "growth_native_0p1",
                "growth_1deg",
                "resolution_effect_growth",
            ),
        ]:
            native = d[native_col].to_numpy(float)
            coarse = d[coarse_col].to_numpy(float)
            res = d[res_col].to_numpy(float)

            # The primary climate-model uncertainty is the SD of the native
            # high-resolution result across the driving models.
            model_sd = float(np.std(native, ddof=1)) if len(native) > 1 else np.nan
            model_mean = float(np.mean(native))
            model_min = float(np.min(native))
            model_max = float(np.max(native))

            # Paired resolution effect. Mean signed effect describes systematic
            # bias; mean absolute effect describes typical magnitude.
            mean_res = float(np.mean(res))
            mean_abs_res = float(np.mean(np.abs(res)))

            ratio = (
                mean_abs_res / model_sd
                if np.isfinite(model_sd) and model_sd > 0
                else np.nan
            )

            rows.append({
                "scenario": scenario,
                "scenario_label": SCENARIO_LABEL[scenario],
                "quantity": quantity,
                "n_models": len(native),
                "native_model_mean": model_mean,
                "native_model_sd": model_sd,
                "native_model_min": model_min,
                "native_model_max": model_max,
                "model_range": model_max - model_min,
                "mean_resolution_effect": mean_res,
                "mean_abs_resolution_effect": mean_abs_res,
                "resolution_to_model_sd": ratio,
                "resolution_effect_pct_of_native": (
                    100.0 * mean_res / model_mean
                    if model_mean != 0 else np.nan
                ),
                "resolution_abs_effect_pct_of_native": (
                    100.0 * mean_abs_res / model_mean
                    if model_mean != 0 else np.nan
                ),
            })

    return pd.DataFrame(rows)


def make_figure(results: pd.DataFrame, summary: pd.DataFrame):
    """
    Main figure:
      (a) model uncertainty versus paired resolution effect
      (b) resolution/model uncertainty ratio
      (c) model-specific resolution effect on projected growth

    Panel (a) uses absolute units and distinguishes level from growth.
    Panel (b) shows the ratio to model SD; R=1 means equal magnitude.
    Panel (c) makes the paired nature explicit for end-century growth.
    """
    set_style()

    fig, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.35), constrained_layout=True
    )

    scenario_order = SCENARIOS
    marker_map = {"ssp245": "o", "ssp370": "s", "ssp585": "^"}

    # -----------------------------------------------------------------------
    # (a) Resolution effect versus CMIP6 model spread
    # -----------------------------------------------------------------------
    ax = axes[0]

    for scenario in scenario_order:
        d = summary[
            (summary["scenario"] == scenario)
            & (summary["quantity"].isin(["baseline_level", "far_level"]))
        ]

        for _, r in d.iterrows():
            x = r["native_model_sd"]
            y = r["mean_abs_resolution_effect"]
            label = (
                f"{r['scenario_label']} "
                f"{'baseline' if r['quantity']=='baseline_level' else '2081–2100'}"
            )
            ax.scatter(
                x, y,
                marker=marker_map[scenario],
                s=38,
                alpha=0.85,
                label=label,
            )

    lims = ax.get_xlim()
    ax.plot(lims, lims, "--", linewidth=0.8, color="0.45", zorder=0)
    ax.set_xlim(lims)
    ax.set_xlabel("CMIP6 model SD (degree days)")
    ax.set_ylabel("Mean absolute resolution effect (degree days)")
    ax.set_title("(a) Resolution effect versus model spread", loc="left",
                 fontweight="bold")

    # Avoid an oversized legend by using the scenario markers only.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6.5, frameon=False,
                  loc="upper left")

    # -----------------------------------------------------------------------
    # (b) Ratio
    # -----------------------------------------------------------------------
    ax = axes[1]

    width = 0.22
    x = np.arange(3)
    quantities = ["baseline_level", "far_level", "growth"]
    qlabels = ["Baseline", "2081–2100", "Growth"]

    for i, scenario in enumerate(scenario_order):
        vals = []
        for q in quantities:
            row = summary[
                (summary["scenario"] == scenario)
                & (summary["quantity"] == q)
            ]
            vals.append(
                float(row["resolution_to_model_sd"].iloc[0])
                if not row.empty else np.nan
            )
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=SCENARIO_LABEL[scenario],
        )

    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="0.45")
    ax.set_xticks(x)
    ax.set_xticklabels(qlabels)
    ax.set_ylabel("Resolution effect / CMIP6 model SD")
    ax.set_title("(b) Relative importance", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False)

    # -----------------------------------------------------------------------
    # (c) Model-by-model resolution effect on projected growth
    # -----------------------------------------------------------------------
    ax = axes[2]

    plot_rows = results.copy()
    xpos = {"ssp245": 0, "ssp370": 1, "ssp585": 2}

    for scenario in scenario_order:
        d = plot_rows[plot_rows["scenario"] == scenario]
        x0 = xpos[scenario]

        rng = np.random.default_rng(42 + x0)
        jitter = rng.uniform(-0.14, 0.14, len(d))

        ax.scatter(
            np.full(len(d), x0) + jitter,
            100.0 * d["resolution_effect_growth"]
            / d["growth_native_0p1"],
            s=28,
            alpha=0.8,
        )

        mean = (
            100.0 * d["resolution_effect_growth"]
            / d["growth_native_0p1"]
        ).mean()
        ax.plot(
            [x0 - 0.18, x0 + 0.18],
            [mean, mean],
            linewidth=1.5,
            color="black",
        )

    ax.axhline(0, linewidth=0.7, color="0.45")
    ax.set_xticks(range(3))
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in scenario_order])
    ax.set_ylabel("Resolution effect on growth (%)")
    ax.set_title("(c) Paired effect on projected growth", loc="left",
                 fontweight="bold")

    fig.suptitle(
        "Spatial-resolution effects versus CMIP6 driving-model spread",
        fontsize=10.5,
        fontweight="bold",
    )

    return fig


def main():
    set_style()
    os.makedirs(FIGDIR, exist_ok=True)

    models = TEST_MODELS if TEST_MODE else MODELS
    print("\nResolution versus climate-model uncertainty")
    print(f"CDD base: {T_BASE_CDD} °C")
    print(f"Baseline: {BASE[0]}–{BASE[1]}")
    print(f"Future: {FUTURE[0]}–{FUTURE[1]}")
    print(f"Native resolution: 0.1°")
    print(f"Coarse resolution: {0.1 * COARSEN_FACTOR:.1f}°")
    print(f"Models: {len(models)}")
    print(f"Scenarios: {', '.join(SCENARIOS)}")

    rows = []

    for scenario in SCENARIOS:
        for model in models:
            try:
                result = calculate_model(model, scenario)
                if result is not None:
                    rows.append(result)
            except Exception as exc:
                print(f"\nERROR: {model} {scenario}: {exc}")
                raise

    if not rows:
        raise RuntimeError("No model/scenario results were produced.")

    results = pd.DataFrame(rows)
    summary = summarize(results)

    # Save both model-level and summary-level results in one CSV with a
    # 'record_type' column so the complete provenance is preserved.
    model_out = results.copy()
    model_out.insert(0, "record_type", "model")
    summary_out = summary.copy()
    summary_out.insert(0, "record_type", "summary")

    combined = pd.concat([model_out, summary_out], ignore_index=True, sort=False)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nWritten: {OUT_CSV}")

    print("\nSUMMARY")
    print(
        summary[
            [
                "scenario_label",
                "quantity",
                "n_models",
                "native_model_mean",
                "native_model_sd",
                "model_range",
                "mean_resolution_effect",
                "mean_abs_resolution_effect",
                "resolution_to_model_sd",
                "resolution_abs_effect_pct_of_native",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    fig = make_figure(results, summary)
    save_figure(fig, FIG_STEM)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
