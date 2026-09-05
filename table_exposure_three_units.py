"""
table_exposure_three_units.py

Manuscript table: end of century cooling exposure reported in three
complementary units, plus the same across global warming levels.

Why three units. Person degree days is an extensive quantity and is the correct
thing to decompose into drivers, but it is abstract and a reader cannot judge
whether a number is large. Per capita degree days answers "what does the average
resident experience". Headcount above a threshold answers "how many people",
which is the unit policy is written in. Reporting all three at once heads off the
standard objection that person degree days is uninterpretable, and it costs one
table.

Population is held static at the 2010 grid throughout, so every number isolates
the climate signal from demographic change. Figure 7 handles the demographic
contribution separately.

Two outputs:
    Table A  end of century, 2081-2100, by scenario
    Table B  SSP5-8.5 by global warming level

Both are written as CSV and as a pipe delimited table for pasting into Word.

Requires: preprocess_build_cache.py stage 1.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

import cdd_common as cc


THRESHOLDS = [300, 500, 800, 1500]     # degC days per year
FAR = cc.FUTURE_PERIODS["far"]


def exposure_row(cdd, cdd_base, pop, label):
    """Exposure in three units, against a fixed baseline field."""
    valid = cdd.notnull() & cdd_base.notnull() & pop.notnull()
    c, c0, p = cdd.where(valid), cdd_base.where(valid), pop.where(valid)
    tot_pop = float(p.sum())

    person_dd = float(((c - c0) * p).sum())
    per_capita = person_dd / tot_pop
    pw_level = float((c * p).sum() / tot_pop)

    row = {"scenario": label,
           "person_degC_days_billion": person_dd / 1e9,
           "per_capita_delta_CDD": per_capita,
           "pop_weighted_CDD_level": pw_level}
    for t in THRESHOLDS:
        above = float(p.where(c > t).sum())
        row[f"people_above_{t}_M"] = above / 1e6
        row[f"pct_above_{t}"] = 100 * above / tot_pop
    return row


def as_pipe_table(df, floatfmt="{:,.1f}"):
    """Pipe delimited, which pastes cleanly into Word."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [r[c] if isinstance(r[c], str) else floatfmt.format(r[c])
                 for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    cc.set_style()
    gdf = cc.load_boundary()
    os.makedirs(cc.FIGDIR, exist_ok=True)

    hist = cc.read_cache("annual_cdd_ddpm_historical.nc", var="cdd")
    cdd_base = cc.climatology(hist, cc.BASELINE)

    pop = cc.load_population(cc.PATHS["POP_2010"], gdf,
                             expect_total=cc.CENSUS_2011_POP, label="WorldPop 2010")
    pop = pop.reindex_like(cdd_base, method="nearest", tolerance=0.051)
    print(f"  static population {float(pop.sum())/1e6:,.0f} million\n")

    # ---- Table A: end of century by scenario -----------------------------
    rows = [exposure_row(cdd_base, cdd_base, pop,
                         f"Historical {cc.BASELINE[0]}-{cc.BASELINE[1]}")]
    for sc in cc.SCENARIOS:
        fut = cc.read_cache(f"annual_cdd_ddpm_{sc}.nc", var="cdd")
        rows.append(exposure_row(cc.climatology(fut, FAR), cdd_base, pop,
                                 f"{cc.SCENARIO_LABEL[sc]} ({FAR[0]}-{FAR[1]})"))
    tA = pd.DataFrame(rows).set_index("scenario")

    # ---- Table B: SSP5-8.5 by warming level ------------------------------
    fut585 = cc.read_cache("annual_cdd_ddpm_ssp585.nc", var="cdd")
    rowsB = []
    for gwl, epoch in cc.WARMING_EPOCHS.items():
        rowsB.append(exposure_row(cc.climatology(fut585, epoch), cdd_base, pop,
                                  f"{gwl} $\\degree$C ({epoch[0]}-{epoch[1]})"))
    tB = pd.DataFrame(rowsB).set_index("scenario")

    for name, t in (("A_by_scenario", tA), ("B_by_warming_level", tB)):
        csv = f"{cc.FIGDIR}/Jha_etal_TableExposure_{name}.csv"
        t.to_csv(csv, float_format="%.3f")
        print(f"Table {name} -> {csv}")

    display = ["person_degC_days_billion", "per_capita_delta_CDD",
               "pop_weighted_CDD_level", "people_above_800_M", "pct_above_800"]
    print("\nTable A: end of century exposure in three units")
    print(tA[display].round(1).to_string())
    print("\nTable B: SSP5-8.5 by global warming level")
    print(tB[display].round(1).to_string())

    with open(f"{cc.FIGDIR}/Jha_etal_TableExposure.md", "w") as fh:
        for title, t in (("Table A. End of century (2081-2100) exposure "
                          "in three units", tA),
                         ("Table B. SSP5-8.5 by global warming level", tB)):
            fh.write(f"**{title}**\n\n")
            d = t[display].reset_index()
            d.columns = ["Scenario", "Person degC days (billion)",
                         "Per capita delta CDD", "Pop-weighted CDD level",
                         "People above 800 CDD (M)", "Share above 800 CDD (%)"]
            fh.write(as_pipe_table(d) + "\n\n")
        fh.write("Population held static at the WorldPop 2010 grid, so every "
                 "column isolates the climate signal from demographic change. "
                 "Person degree days and per capita values are increments "
                 "relative to the 1985-2014 baseline; the level and headcount "
                 "columns are absolute.\n")
    print(f"\nPipe delimited version -> {cc.FIGDIR}/Jha_etal_TableExposure.md")

    far585 = tA.loc[f"{cc.SCENARIO_LABEL['ssp585']} ({FAR[0]}-{FAR[1]})"]
    hist_row = tA.iloc[0]
    print(f"\n  Headline: the average resident gains "
          f"{far585.per_capita_delta_CDD:,.0f} degC days, and the share living "
          f"above 800 CDD rises from {hist_row.pct_above_800:.1f}% to "
          f"{far585.pct_above_800:.1f}%.")


if __name__ == "__main__":
    main()
