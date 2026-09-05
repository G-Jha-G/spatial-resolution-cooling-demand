"""
cdd_common.py

Shared foundation for every figure in Jha and Gupta, Earth's Future.

Every figure script imports from here, so that the path from disk to analysis is
defined once, is identical across products, and can be deposited to Zenodo as a
single auditable ingest routine.

Design decisions this module enforces, each because the source files disagree:

  Variable naming. Per-member QDM files store 'tas'; the QDM and DDPM ensemble
  means store 'tas_mean'; the raw ensemble stores 'tas_ensmean'; MSWX stores
  'air_temperature'. The loader resolves whichever is present and returns a
  DataArray always named 'tas', so no downstream script hardcodes a name.

  Units. The per-member QDM files are already in Celsius (global attribute
  bc_output_units = degree_Celsius) but inherit a CMIP6 variable attribute that
  may still read 'K'. Trusting the variable attribute would subtract 273.15 from
  Celsius data and yield a CDD field of zeros, which still plots and would not
  be caught by eye. The loader therefore decides on the data themselves, using a
  robust median above 200 as the Kelvin test, and cross-checks against metadata,
  reporting any disagreement rather than silently choosing.

  Timestamps. Per-member files are stamped at 12:00, ensemble means at midnight.
  Any operation that aligns members against an ensemble-mean series on the time
  axis would silently return an empty intersection. The loader normalises all
  timestamps to midnight, and all aggregation is by integer calendar year.

  Latitude orientation. Climate files run north to south; the SSP2 population
  grids run south to north. Arithmetic is label-aligned by xarray and therefore
  safe, but block coarsening, pcolormesh and positional indexing are not. The
  loader sorts every field to ascending latitude and asserts monotonicity.

  Clipping order. The index is always computed first and clipped afterwards.
  Clipping a field before spatial coarsening admits zeros from outside the
  boundary into block means and manufactures an apparent resolution gap
  (Methods 3.3, SI Text S2).

Author: Shailesh Kumar Jha, HIMPACT Lab, IIT Mandi.
"""

from __future__ import annotations

import os
import warnings

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 1. Configuration: single source of truth
# =============================================================================

T_BASE_CDD = 24.0     # BEE mandated default air conditioner setpoint
T_BASE_HDD = 18.0     # heating base, swept 15 to 18 in the sensitivity analysis

BASELINE = (1985, 2014)
FUTURE_PERIODS = {
    "near": (2021, 2040),
    "mid": (2041, 2060),
    "far": (2081, 2100),
}

# Warming level epochs for the CMIP6 ensemble mean under SSP5-8.5.
# Per model GWLs are not computable here: global mean surface temperature
# requires global fields, and every product in this study covers the India
# domain only. The source of these windows must be cited in Methods.
WARMING_EPOCHS = {
    "1.5": (2021, 2040),
    "2.0": (2032, 2051),
    "2.5": (2041, 2060),
    "3.0": (2051, 2070),
    "4.0": (2068, 2087),
}

ROOT = "/home/shailesh/extra_storage/shailesh_storage"
CACHE = f"{ROOT}/CDD_paper/cache"
FIGDIR = f"{ROOT}/CDD_paper/figures"

MODELS = [
    "ACCESS-CM2", "AWI-CM-1-1-MR", "CNRM-CM6-1", "CNRM-ESM2-1", "IITM-ESM",
    "IPSL-CM6A-LR", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-LR", "MRI-ESM2-0",
]

SCENARIOS = ["ssp245", "ssp370", "ssp585"]

SCENARIO_LABEL = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

SCENARIO_COLOR = {          # colour vision safe, ordered by forcing
    "ssp245": "#4477AA",
    "ssp370": "#EE7733",
    "ssp585": "#CC3311",
}

PATHS = {
    "RAW_HIST": (f"{ROOT}/raw-data/nc-files/cmip6/tas/historical_1850-2014/"
                 "ensembles/ensemble_tas_ensmean_0p1deg_historical.nc"),
    "QDM_HIST": (f"{ROOT}/qdm_output/tas/"
                 "MME_ensemble_10models_historical_1979-2014_0p1deg.nc"),
    "QDM_ssp245": (f"{ROOT}/qdm_output/tas/"
                   "MME_ensemble_10models_ssp245_2015-2100_0p1deg.nc"),
    "QDM_ssp370": (f"{ROOT}/qdm_output/tas/"
                   "MME_ensemble_10models_ssp370_2015-2100_0p1deg.nc"),
    "QDM_ssp585": (f"{ROOT}/qdm_output/tas/"
                   "MME_ensemble_10models_ssp585_2015-2100_0p1deg.nc"),
    "DDPM_HIST": (f"{ROOT}/Results/DDPM_hist_MME_tas_mean/"
                  "DDPM_tas_mean_MME_historical_1979-2014.nc"),
    "DDPM_ssp245": (f"{ROOT}/Results/DDPM_ssp245_MME_tas_mean/"
                    "Final_DDPM_tas_mean_MME_ssp245_2015-2100.nc"),
    "DDPM_ssp370": (f"{ROOT}/Results/DDPM_ssp370_MME_tas_mean/"
                    "Final_DDPM_tas_mean_MME_ssp370_2015-2100.nc"),
    "DDPM_ssp585": (f"{ROOT}/Results/DDPM_ssp585_MME_tas_mean/"
                    "Final_DDPM_tas_mean_MME_ssp585_2015-2100.nc"),
    "MSWX": "/home/shailesh/my_data/downscaling/dataset/MSWX_Temp_Daily_Ind_clipped.nc",
    "POP_2010": f"{ROOT}/ERL_GRL/CDD/ind_ppp_2010_UNadj_0p1deg_MME_Aligned.nc",
    "POP_SSP2_DIR": f"{ROOT}/ERL_GRL/CDD/SSP_ssp2_1km",
    "POP_SSP5_DIR": f"{ROOT}/ERL_GRL/CDD/SSP_ssp5_1km",
    "NTL": f"{ROOT}/ERL_GRL/DMSP_Nightlights_India_0p1deg.tif",
    "SHP": f"{ROOT}/raw-data/shapefiles/India_Boundary.shp",
}

# Effective resolution ladder. The analysis grid is 0.1 degree, so a coarsening
# factor f gives an effective resolution of 0.1 x f degrees. Integer factors
# only, since block averaging requires them; 0.25 degree is therefore not
# directly attainable and is bracketed by factors 2 and 3.
COARSEN_FACTORS = [2, 3, 4, 5, 6, 8, 10]

def factor_to_degrees(f: int) -> float:
    return 0.1 * f

# Building climate zones, ASHRAE Standard 169 style with a heating based cold
# override, as specified in Methods 3.7 and the Figure 6 caption.
#
# IMPORTANT: the standard is keyed to CDD base 10 C and HDD base 18 C, NOT to
# the 24 C demand base used everywhere else in this study. The 24 C base is
# India's mandated air conditioner setpoint and is the right anchor for demand;
# the zone tiers are a separate, externally defined classification and must use
# the standard's own bases or the tier shares will not be comparable with the
# published code zones. Both indices are therefore cached.
#
# Thresholds below follow ASHRAE 169. VERIFY AGAINST THE STANDARD before
# submission and cite the edition used.
# Base temperature sweep. The literature treats the base as a choice matched to
# the application rather than a constant: IEA uses 18 C; Bhatnagar, Mathur and
# Garg (2018) fit 18 C for India from building energy signatures; Chen (2017)
# and Haase (2009) use 26 C for hot humid Asia; Borah and colleagues compute 20
# to 28 C for North East India; Priya and colleagues sweep 18 to 28 C for
# Tiruchirappalli. The 24 C headline is India's mandated air conditioner
# setpoint, which is a regulatory anchor rather than a convention. Reporting the
# full sweep makes the base an axis of the result instead of an assumption.
CDD_BASES = [18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
HDD_BASES = [12.0, 14.0, 16.0, 18.0, 20.0]

T_BASE_ZONE_CDD = 10.0
T_BASE_ZONE_HDD = 18.0

# (name, CDD10 lower, CDD10 upper). Evaluated in order; the first match wins.
# Zones 3C and colder are reached only through the heating override below.
ASHRAE_CDD_TIERS = [
    ("Extremely hot",  6000.0, np.inf),   # zone 0
    ("Very hot",       5000.0, 6000.0),   # zone 1
    ("Hot",            3500.0, 5000.0),   # zone 2
    ("Warm",           2500.0, 3500.0),   # zone 3A/3B
]

# Heating based cold override, applied where CDD10 <= 2500.
# (name, HDD18 lower, HDD18 upper)
ASHRAE_HDD_TIERS = [
    ("Warm marine",       -np.inf, 2000.0),   # zone 3C
    ("Mixed",              2000.0, 3000.0),   # zone 4
    ("Cool",               3000.0, 4000.0),   # zone 5
    ("Cold",               4000.0, 5000.0),   # zone 6
    ("Very cold",          5000.0, 7000.0),   # zone 7
    ("Subarctic",          7000.0, np.inf),   # zone 8
]

# Ordered hottest to coldest, for the categorical palette and for testing
# whether a cell has moved to a hotter tier.
ZONE_ORDER = ([n for n, _, _ in ASHRAE_CDD_TIERS]
              + [n for n, _, _ in ASHRAE_HDD_TIERS])


def ashrae_zone(cdd10, hdd18):
    """
    ASHRAE 169 style zone index, hottest = 0, using CDD base 10 C and HDD base
    18 C. Returns a float index into ZONE_ORDER so it can be differenced.
    """
    out = xr.full_like(cdd10, np.nan)
    for i, (_, lo, hi) in enumerate(ASHRAE_CDD_TIERS):
        out = xr.where((cdd10 >= lo) & (cdd10 < hi), float(i), out)

    cold = cdd10 < ASHRAE_CDD_TIERS[-1][1]
    offset = len(ASHRAE_CDD_TIERS)
    for j, (_, lo, hi) in enumerate(ASHRAE_HDD_TIERS):
        out = xr.where(cold & (hdd18 >= lo) & (hdd18 < hi), float(offset + j), out)

    return out.where(cdd10.notnull() & hdd18.notnull())


POP_DECADES = [2030, 2040, 2050, 2060, 2080]
EPOCH_TO_DECADE = {"1.5": 2030, "2.0": 2040, "2.5": 2050, "3.0": 2060, "4.0": 2080}

CENSUS_2011_POP = 1.2106e9   # Census of India 2011, for the load-time assertion

# Generous India bounding box. Slicing at open time cuts the read volume of the
# per member scenario files from 930 GB to about 331 GB without touching any
# cell inside the national boundary.
INDIA_BBOX = dict(lat=(5.5, 38.5), lon=(66.5, 98.5))

# Candidate temperature variable names, in resolution order.
TAS_CANDIDATES = ["tas", "tas_mean", "tas_ensmean", "air_temperature", "t2m"]

# Variables that must never be used as a temperature field. The combined QDM
# MME files store tas_min, tas_max, tas_std and tas_spread, which are statistics
# ACROSS THE TEN MODELS on each day, not diurnal minima and maxima and not a
# temporal standard deviation. Using tas_min or tas_max as if it were a daily
# extreme would silently produce a physically meaningless index, so they are
# blocked by name and the loader refuses them.
FORBIDDEN_VARS = {
    "tas_min": "across-model daily minimum, not the diurnal minimum",
    "tas_max": "across-model daily maximum, not the diurnal maximum",
    "tas_std": "across-model daily standard deviation, not temporal variability",
    "tas_spread": "across-model daily range, not a diurnal range",
}

KELVIN_TEST = 200.0   # a median above this can only be Kelvin


# =============================================================================
# 2. Ingest
# =============================================================================

def _resolve_var(ds: xr.Dataset, requested: str | None = None) -> str:
    """
    Resolve the temperature variable, refusing across-model statistics.

    The combined QDM MME files carry five 3-D variables. Only tas_mean is a
    temperature field in the sense this study needs; the others are statistics
    across the ten driving models on each day. Selecting one of them by accident
    would produce an index that still plots and still has plausible units, which
    is exactly the class of error that survives review.
    """
    if requested is not None:
        if requested in FORBIDDEN_VARS:
            raise ValueError(
                f"'{requested}' is {FORBIDDEN_VARS[requested]}. It must not be "
                f"used as a temperature field."
            )
        if requested not in ds.data_vars:
            raise KeyError(f"'{requested}' absent. Present: {list(ds.data_vars)}")
        return requested

    for name in TAS_CANDIDATES:
        if name in ds.data_vars:
            return name

    three_d = [v for v in ds.data_vars
               if ds[v].ndim == 3 and v not in FORBIDDEN_VARS]
    if len(three_d) == 1:
        return three_d[0]
    raise KeyError(
        f"No unambiguous temperature variable. Present: {list(ds.data_vars)}. "
        f"Pass var= explicitly or extend TAS_CANDIDATES in cdd_common.py."
    )


def to_celsius(da: xr.DataArray, ds_attrs: dict | None = None,
               label: str = "") -> xr.DataArray:
    """
    Convert to Celsius, deciding on the data rather than on metadata.

    A robust median above 200 can only be Kelvin for near surface air
    temperature over India, so that is the test. Metadata is read as well and
    any disagreement is reported, because a metadata conflict is a sign that
    something upstream changed and should not pass silently.
    """
    ds_attrs = ds_attrs or {}

    # Robust sample: a slab of days, median over all cells. Median rather than
    # mean so that fill values or a few corrupted cells cannot flip the test.
    if "time" in da.dims:
        n = min(da.sizes["time"], 400)
        sample = da.isel(time=slice(0, n))
    else:
        sample = da
    median = float(np.nanmedian(np.asarray(sample.compute())))

    is_kelvin = median > KELVIN_TEST

    # Cross-check against metadata. bc_output_units is authoritative where it
    # exists, because the per member files are written in Celsius while
    # retaining an inherited CMIP6 variable attribute of 'K'.
    declared = str(
        ds_attrs.get("bc_output_units", da.attrs.get("units", ""))
    ).strip().lower()
    declared_kelvin = declared in ("k", "kelvin", "degk", "deg_k")
    declared_celsius = declared in (
        "c", "celsius", "degc", "deg_c", "degree_celsius", "degrees_celsius",
    )

    if declared and ((declared_kelvin and not is_kelvin)
                     or (declared_celsius and is_kelvin)):
        print(f"    [units] {label}: metadata says '{declared}' but the domain "
              f"median is {median:.1f}. Trusting the data.")

    if is_kelvin:
        print(f"    [units] {label}: median {median:.1f} -> Kelvin, "
              f"subtracting 273.15")
        out = da - 273.15
    else:
        print(f"    [units] {label}: median {median:.1f} -> already Celsius")
        out = da

    # Physical range assertion. This is the guard that catches a double
    # conversion, which would otherwise produce a CDD field of zeros that still
    # plots and would survive visual inspection.
    if "time" in out.dims:
        chk = out.isel(time=slice(0, min(out.sizes["time"], 400)))
    else:
        chk = out
    med_c = float(np.nanmedian(np.asarray(chk.compute())))
    if not (-40.0 < med_c < 50.0):
        raise ValueError(
            f"{label}: post-conversion median {med_c:.1f} C is outside the "
            f"physical range for near surface air temperature over this domain. "
            f"Refusing to continue."
        )

    out.attrs = dict(da.attrs)
    out.attrs["units"] = "degC"
    return out


def open_tas(path: str, label: str = "", bbox: dict | None = INDIA_BBOX,
             years: tuple[int, int] | None = None,
             chunks: dict | None = None,
             var: str | None = None) -> xr.DataArray:
    """
    Canonical temperature loader. Returns a DataArray named 'tas' in Celsius,
    on ascending latitude, with timestamps normalised to midnight.

    Parameters
    ----------
    bbox   : lat/lon subset applied at open time. Pass None for the full domain.
    years  : inclusive calendar year range, applied before any computation.
    """
    label = label or os.path.basename(path)
    print(f"  opening {label}")

    chunks = chunks or {"time": 730}
    ds = xr.open_dataset(path, chunks=chunks)

    var = _resolve_var(ds, requested=var)
    others = [v for v in ds.data_vars if v != var and ds[v].ndim == 3]
    if others:
        print(f"    [vars] using '{var}'; ignoring {others}")
    else:
        print(f"    [vars] using '{var}'")
    da = ds[var]

    # Drop scalar coordinates that some products carry and others do not.
    for coord in ("height", "band", "spatial_ref", "crs"):
        if coord in da.coords:
            da = da.drop_vars(coord)
        if coord in ds.coords and coord in da.dims:
            da = da.squeeze(coord, drop=True)

    da = _standardise_spatial_names(da)

    # Ascending latitude, always. Asserted rather than assumed.
    da = da.sortby("lat")
    assert bool(np.all(np.diff(da.lat.values) > 0)), f"{label}: lat not ascending"
    assert bool(np.all(np.diff(da.lon.values) > 0)), f"{label}: lon not ascending"

    # Midnight timestamps, so per member files (12:00) and ensemble means
    # (00:00) can be combined on the time axis without an empty intersection.
    if "time" in da.dims:
        idx = pd.DatetimeIndex(da.indexes["time"])
        if (idx.hour != 0).any() or (idx.minute != 0).any():
            print(f"    [time] normalising timestamps to midnight")
            da = da.assign_coords(time=idx.normalize())

    if bbox is not None:
        da = da.sel(lat=slice(*bbox["lat"]), lon=slice(*bbox["lon"]))

    if years is not None:
        da = da.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))

    da = to_celsius(da, ds.attrs, label=label)
    da.name = "tas"
    da.attrs["source_file"] = path
    da.attrs["source_variable"] = var
    return da


def _standardise_spatial_names(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "y" in da.dims and "x" in da.dims:
        rename.update({"y": "lat", "x": "lon"})
    if "latitude" in da.dims:
        rename.update({"latitude": "lat", "longitude": "lon"})
    return da.rename(rename) if rename else da


def load_boundary() -> gpd.GeoDataFrame:
    """
    Government of India approved boundary. This is the only boundary source used
    anywhere in the manuscript; no Cartopy political border features are drawn.
    """
    gdf = gpd.read_file(PATHS["SHP"])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def clip_india(da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.DataArray:
    """Clip to the national boundary. Always applied after index computation."""
    da = _standardise_spatial_names(da).sortby("lat")
    da = (da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.write_crs("EPSG:4326"))
    return da.rio.clip(gdf.geometry, gdf.crs, drop=True, all_touched=False)


def load_population(path: str, gdf: gpd.GeoDataFrame | None = None,
                    expect_total: float | None = None,
                    label: str = "") -> xr.DataArray:
    """
    Population on the 0.1 degree analysis grid, ascending latitude, zeros
    preserved. Negative values are treated as nodata.
    """
    label = label or os.path.basename(path)
    print(f"  opening population {label}")
    if path.endswith((".tif", ".tiff")):
        ds = xr.open_dataset(path, engine="rasterio")
    else:
        ds = xr.open_dataset(path)

    var = "population" if "population" in ds.data_vars else list(ds.data_vars)[0]
    pop = _standardise_spatial_names(ds[var])
    if "band" in pop.dims:
        pop = pop.squeeze("band", drop=True)
    pop = pop.sortby("lat")
    assert bool(np.all(np.diff(pop.lat.values) > 0)), f"{label}: lat not ascending"

    pop = pop.where(pop >= 0)          # negatives are nodata; zeros are real
    if gdf is not None:
        pop = clip_india(pop, gdf)

    total = float(pop.sum())
    print(f"    total on the India mask: {total/1e6:,.1f} million")
    if expect_total is not None:
        rel = abs(total - expect_total) / expect_total
        if rel > 0.05:
            raise ValueError(
                f"{label}: total {total/1e6:.1f} M departs from the expected "
                f"{expect_total/1e6:.1f} M by {100*rel:.1f} percent. Check the "
                f"aggregation to 0.1 degree before using this file."
            )
        print(f"    within {100*rel:.1f} percent of the expected total")

    pop.name = "population"
    return pop


# ---------------------------------------------------------------------------
# Regional aggregation
# ---------------------------------------------------------------------------
#
# Regions are the fifteen agro-climatic regions of India (ICAR / Planning
# Commission), delineated on physiography, soil, climate and cropping pattern.
# They are used here in preference to latitude and longitude boxes for three
# reasons: they are a published and citable definition rather than one invented
# for this study; their boundaries follow the terrain that generates the subgrid
# heterogeneity under investigation, so they do not cut across the Western Ghats
# or the Himalayan front; and they are the units in which Indian agricultural and
# climate planning is already expressed.
#
# Region names in the shapefile carry HTML entities such as &amp;, which are
# unescaped on load.
REGION_SHP = (f"{ROOT}/raw-data/shapefiles/Agro_Climate/AgroClimatic_Regions.shp")
REGION_NAME_FIELD = "regionname"
REGION_CODE_FIELD = "regioncode"

# Regions too small to support a mean over whole coarse blocks. They are
# retained in cell-level analyses and reported with their block count in
# block-level ones, so a thin sample is visible rather than silent.
SMALL_REGIONS = {"Island region"}


def load_regions(verbose: bool = True) -> gpd.GeoDataFrame:
    """The fifteen agro-climatic regions, on EPSG:4326, with names unescaped."""
    import html as _html

    gdf = gpd.read_file(REGION_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["region"] = (gdf[REGION_NAME_FIELD].astype(str)
                     .map(_html.unescape).str.replace(r"\s+", " ", regex=True)
                     .str.strip())
    gdf["code"] = gdf[REGION_CODE_FIELD].astype(int)
    gdf = gdf.sort_values("code").reset_index(drop=True)
    if verbose:
        print(f"  [regions] {len(gdf)} agro-climatic regions loaded")
    return gdf


def region_index(field: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Integer region code per cell, rasterized onto a field's grid.

    Rasterizing once and selecting by code is far cheaper than clipping each
    region separately, and it guarantees that every cell belongs to exactly one
    region, so regional totals sum to the national total without double
    counting at shared boundaries.
    """
    from rasterio import features
    from affine import Affine

    lat = np.asarray(field.lat.values, dtype=float)
    lon = np.asarray(field.lon.values, dtype=float)
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    transform = Affine.translation(lon[0] - dlon / 2, lat[0] - dlat / 2) \
        * Affine.scale(dlon, dlat)

    shapes = [(geom, int(code)) for geom, code in zip(gdf.geometry, gdf.code)]
    arr = features.rasterize(shapes, out_shape=(len(lat), len(lon)),
                             transform=transform, fill=0, dtype="int32",
                             all_touched=False)
    out = xr.DataArray(arr, coords={"lat": field.lat, "lon": field.lon},
                       dims=("lat", "lon"), name="region_code")
    return out.where(out > 0)


def region_table(fields: dict, field_for_grid: xr.DataArray,
                 gdf: gpd.GeoDataFrame, weights: xr.DataArray | None = None):
    """
    Area or population weighted regional means of several fields at once.

    fields  : mapping of column name to DataArray on the same grid
    weights : optional weighting field; cosine latitude is used when omitted
    """
    idx = region_index(field_for_grid, gdf)
    if weights is None:
        weights = np.cos(np.deg2rad(field_for_grid.lat)).broadcast_like(
            field_for_grid)

    rows = []
    for _, r in gdf.iterrows():
        m = idx == r.code
        n = int((m & field_for_grid.notnull()).sum())
        row = {"region": r.region, "code": int(r.code), "n_cells": n}
        for name, da in fields.items():
            w = weights.where(m & da.notnull())
            tot = float(w.sum())
            row[name] = float((da.where(m) * w).sum() / tot) if tot > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def normalized_added_value(err_ref, err_test):
    """
    Normalized added value of a test product against a reference comparator,
    from per cell absolute errors.

        NAV = (|e_ref| - |e_test|) / (|e_ref| + |e_test|)

    Bounded on [-1, 1]. Positive means the test product is closer to the
    observations in that cell. Unlike a correlation across years it needs no
    temporal synchrony, and unlike a difference of MAEs it is not dominated by
    the cells with the largest absolute errors, so it answers the per cell
    question directly: is this product better here.
    """
    a, b = abs(err_ref), abs(err_test)
    den = a + b
    return xr.where(den > 0, (a - b) / den, 0.0).where(err_ref.notnull()
                                                       & err_test.notnull())



# =============================================================================
# 3. Indices
# =============================================================================

def complete_years(da: xr.DataArray, verbose: bool = True) -> np.ndarray:
    """Calendar years with a full record, so climatologies are not diluted."""
    counts = da.time.groupby("time.year").count()
    years = counts.where(counts >= 365, drop=True).year.values
    dropped = np.setdiff1d(counts.year.values, years)
    if verbose and dropped.size:
        print(f"    dropping incomplete years: {list(dropped)}")
    return years


def frac_above(tas: xr.DataArray, threshold: float,
               dim: str = "time") -> xr.DataArray:
    """
    Fraction of the record above a threshold, with the input mask preserved.

    Use this rather than (tas > threshold).mean(dim). A direct comparison
    returns False, not NaN, wherever tas is NaN, because NaN > x is False. The
    time mean over a masked cell is then 0 rather than NaN, the cell counts as
    valid, and any subsequent spatial mean silently averages the whole domain
    instead of the land. Over India, whose land covers 34 percent of the
    analysis box, that understates a national exceedance probability by a factor
    of about three. This is the third place in this project where the same trap
    appeared, hence a named function.
    """
    valid = tas.isel({dim: 0}, drop=True).notnull()
    return xr.where(tas > threshold, 1.0, 0.0).mean(dim=dim).where(valid)


def frac_below(tas: xr.DataArray, threshold: float,
               dim: str = "time") -> xr.DataArray:
    """Fraction of the record below a threshold, with the mask preserved."""
    valid = tas.isel({dim: 0}, drop=True).notnull()
    return xr.where(tas < threshold, 1.0, 0.0).mean(dim=dim).where(valid)


def annual_cdd(tas: xr.DataArray, base: float = T_BASE_CDD) -> xr.DataArray:
    """
    Annual accumulated cooling degree days from daily mean temperature.

    The input mask is restored explicitly at the end. xr.where(tas > base, ...)
    returns the zero branch wherever tas is NaN, because NaN > base evaluates
    False, so the annual sum over a masked cell is 0 rather than NaN and
    .notnull() would be True everywhere. Any later operation that builds a mask
    from the result, takes a spatial mean, or block averages across a coastline
    would then silently include those zeros. This is the same failure that
    corrupted the national mean temperature in the seasonal cache, and it is
    fixed here at the source rather than at each call site.
    """
    years = complete_years(tas)
    tas = tas.sel(time=tas.time.dt.year.isin(years))
    # drop=True is required, not cosmetic. Without it the scalar time
    # coordinate of the first record rides along on the mask and therefore on
    # every field masked with it. Two fields derived from records with
    # different start dates then carry different scalar time values, and any
    # later attempt to combine them into a Dataset fails with a merge conflict.
    valid = tas.isel(time=0, drop=True).notnull()
    exceed = xr.where(tas > base, tas - base, 0.0)
    out = exceed.groupby("time.year").sum(dim="time", skipna=False).where(valid)
    out.name = "cdd"
    out.attrs = {
        "long_name": f"Annual cooling degree days, base {base:g} C",
        "units": "degC days",
        "base_temperature_degC": base,
    }
    return out


def annual_hdd(tas: xr.DataArray, base: float = T_BASE_HDD) -> xr.DataArray:
    """Annual accumulated heating degree days from daily mean temperature."""
    years = complete_years(tas, verbose=False)
    tas = tas.sel(time=tas.time.dt.year.isin(years))
    valid = tas.isel(time=0, drop=True).notnull()
    deficit = xr.where(tas < base, base - tas, 0.0)
    out = deficit.groupby("time.year").sum(dim="time", skipna=False).where(valid)
    out.name = "hdd"
    out.attrs = {
        "long_name": f"Annual heating degree days, base {base:g} C",
        "units": "degC days",
        "base_temperature_degC": base,
    }
    return out


def climatology(annual: xr.DataArray, period: tuple[int, int]) -> xr.DataArray:
    """
    Mean over the calendar years of a period that are actually present.

    Records that end before the nominal period end, notably IITM-ESM, are
    averaged over the years available. The number of years used is recorded in
    the attributes so it can be reported. Because the missing years are the
    warmest, a short window biases that member low by roughly 0.2 to 0.5 percent
    of the end of century level, which is negligible against ensemble spread but
    is stated rather than hidden.
    """
    y0, y1 = period
    sel = annual.sel(year=slice(y0, y1))
    n = int(sel.sizes["year"])
    yrs = sel.year.values
    out = sel.mean(dim="year")
    out.attrs = dict(annual.attrs)
    out.attrs.update({
        "period_requested": f"{y0}-{y1}",
        "period_used": f"{int(yrs.min())}-{int(yrs.max())}",
        "n_years": n,
        "n_years_requested": y1 - y0 + 1,
    })
    if n < (y1 - y0 + 1):
        print(f"    short record: {n} of {y1-y0+1} years "
              f"({int(yrs.min())}-{int(yrs.max())})")
    return out


# =============================================================================
# 4. Weighting and statistics
# =============================================================================

def area_weights(da: xr.DataArray) -> xr.DataArray:
    """cos(latitude) weights, masked to the valid cells of the field."""
    w = np.cos(np.deg2rad(da.lat))
    return w.broadcast_like(da).where(da.notnull())


def wmean(da: xr.DataArray) -> float:
    """Area weighted spatial mean over the India mask."""
    w = area_weights(da)
    return float((da * w).sum() / w.sum())


def pop_weighted(field: xr.DataArray, pop: xr.DataArray) -> float:
    """
    Population weighted mean. Area weights are not applied: population counts
    are already per cell totals, so weighting by them alone is correct.
    """
    valid = field.notnull() & pop.notnull()
    f, p = field.where(valid), pop.where(valid)
    return float((f * p).sum() / p.sum())


def exposure(field: xr.DataArray, pop: xr.DataArray) -> float:
    """Total person degree days."""
    valid = field.notnull() & pop.notnull()
    return float((field.where(valid) * pop.where(valid)).sum())


def align_to(source: xr.DataArray, target: xr.DataArray,
             label: str = "") -> xr.DataArray:
    """
    Put a field on the reference grid. The 0.1 degree grids in this study share
    cell centres exactly, so this is a subset rather than an interpolation, and
    the assertion makes any future coordinate drift fail loudly.
    """
    if source.sizes.get("lat") == target.sizes.get("lat") and \
       source.sizes.get("lon") == target.sizes.get("lon") and \
       np.allclose(source.lat.values, target.lat.values, atol=1e-4) and \
       np.allclose(source.lon.values, target.lon.values, atol=1e-4):
        return source
    out = source.sel(lat=target.lat, lon=target.lon, method="nearest",
                     tolerance=0.051)
    out = out.assign_coords(lat=target.lat, lon=target.lon)
    print(f"    aligned {label} onto the reference grid "
          f"({dict(source.sizes)} -> {dict(out.sizes)})")
    return out


def weighted_stats(product: xr.DataArray, reference: xr.DataArray) -> dict:
    """Area weighted bias, MAE, RMSE, pattern correlation and Wasserstein-1."""
    valid = product.notnull() & reference.notnull()
    p, r = product.where(valid), reference.where(valid)
    w = area_weights(r)
    ws = float(w.sum())

    d = p - r
    bias = float((d * w).sum() / ws)
    mae = float((abs(d) * w).sum() / ws)
    rmse = float(np.sqrt(float(((d ** 2) * w).sum() / ws)))

    pm, rm = float((p * w).sum() / ws), float((r * w).sum() / ws)
    cov = float(((p - pm) * (r - rm) * w).sum() / ws)
    sp = np.sqrt(float(((p - pm) ** 2 * w).sum() / ws))
    sr = np.sqrt(float(((r - rm) ** 2 * w).sum() / ws))

    return {
        "mean_product": pm,
        "mean_reference": rm,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "pattern_r": cov / (sp * sr),
        "w1": wasserstein1(p, r),
        "n_cells": int(valid.sum()),
    }


def wasserstein1(p: xr.DataArray, r: xr.DataArray) -> float:
    """
    Wasserstein-1 distance between the spatial distributions of two fields.
    Computed as the mean absolute difference between matched quantiles, which is
    the standard closed form for equal sample sizes.
    """
    a = np.sort(p.values[np.isfinite(p.values)])
    b = np.sort(r.values[np.isfinite(r.values)])
    q = np.linspace(0.0, 1.0, min(len(a), len(b), 20000))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def block_coarsen(da: xr.DataArray, factor: int) -> xr.DataArray:
    """
    Block average to a coarser effective resolution.

    Applied to the unclipped field, because a genuine coarse cell blends all the
    air within it. Clipping first would admit zeros from outside the boundary
    and manufacture an apparent resolution gap (Methods 3.3).
    """
    return da.coarsen(lat=factor, lon=factor, boundary="trim").mean()


def block_coarsen_then_refine(da: xr.DataArray, factor: int) -> xr.DataArray:
    """Coarsen then return to the fine grid, for like for like comparison."""
    coarse = block_coarsen(da, factor)
    return coarse.interp(lat=da.lat, lon=da.lon, method="nearest",
                         kwargs={"fill_value": None})


def block_bootstrap_ci(series: np.ndarray, block: int = 3, n: int = 5000,
                       alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Circular block bootstrap confidence interval for a mean over years."""
    rng = np.random.default_rng(seed)
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    m = len(x)
    nb = int(np.ceil(m / block))
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m, nb)
        idx = np.concatenate([(np.arange(s, s + block) % m) for s in starts])[:m]
        means[i] = x[idx].mean()
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def theil_sen(years: np.ndarray, values: np.ndarray) -> float:
    """Theil-Sen slope per year, robust to outliers."""
    from scipy.stats import theilslopes
    return float(theilslopes(values, years)[0])


# =============================================================================
# 5. Caching
# =============================================================================

def cache_path(name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)


def write_cache(da: xr.DataArray, name: str, **attrs) -> str:
    path = cache_path(name)
    da = da.copy()
    da.attrs.update(attrs)
    da.attrs["created_by"] = "cdd_common.write_cache"
    da.attrs["base_temperature_cdd"] = T_BASE_CDD
    da.to_netcdf(path, encoding={da.name: {"zlib": True, "complevel": 4}})
    print(f"    cached -> {path}")
    return path


# Variables written alongside the data by rioxarray, which are metadata rather
# than fields and must never be returned as the payload of a cache file.
_METADATA_VARS = {"spatial_ref", "crs", "transverse_mercator", "lambert_conformal_conic"}


def read_cache(name: str, var: str | None = None) -> xr.DataArray:
    """
    Read a cached field.

    rioxarray writes a scalar 'spatial_ref' variable to carry the CRS. Taking
    the first entry of data_vars can therefore return that scalar instead of the
    field, which fails later with a confusing message about missing dimensions.
    The payload is identified as the variable with the most dimensions, ignoring
    known metadata variables, and the choice is asserted.
    """
    path = cache_path(name)
    ds = xr.open_dataset(path)

    if var is not None:
        if var not in ds.data_vars:
            raise KeyError(f"{name}: '{var}' absent. Present: {list(ds.data_vars)}")
        return ds[var]

    candidates = [v for v in ds.data_vars
                  if v not in _METADATA_VARS and ds[v].ndim >= 2]
    if not candidates:
        raise KeyError(
            f"{name}: no field variable found. Present: {list(ds.data_vars)}"
        )
    payload = max(candidates, key=lambda v: ds[v].ndim)
    return ds[payload]


def read_cache_dataset(name: str) -> xr.Dataset:
    """Read a cache file that holds several fields, for example daily moments."""
    return xr.open_dataset(cache_path(name))


def cache_exists(name: str) -> bool:
    return os.path.exists(cache_path(name))


# =============================================================================
# 6. Plotting
# =============================================================================

CMAP_ABS = "YlOrRd"
CMAP_BIAS = "RdBu_r"
CMAP_DIVERGE = "RdBu_r"

MEAN_LAT_INDIA = 22.0


def set_style() -> None:
    """AGU compatible defaults. Applied once at the top of every figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 400,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
    })


def style_map(ax, gdf: gpd.GeoDataFrame, mean_lat: float = MEAN_LAT_INDIA) -> None:
    """Boundary overlay and the correct aspect for a geographic lon/lat plot."""
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=5)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    ax.tick_params(length=2.5)


def panel_label(ax, text: str, **kw) -> None:
    ax.set_title(text, loc="left", fontweight="bold", **kw)


def stat_box(ax, text: str, loc: str = "upper left") -> None:
    x, y, ha, va = (0.025, 0.975, "left", "top")
    if loc == "upper right":
        x, ha = 0.975, "right"
    ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha,
            fontsize=7.5, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="0.7", linewidth=0.5, alpha=0.88))


def hist_inset(ax, data: xr.DataArray, color: str, centred: bool = False,
               rect=(0.60, 0.06, 0.36, 0.20)):
    v = data.values[np.isfinite(data.values)]
    ins = ax.inset_axes(rect)
    ins.hist(v, bins=40, color=color, alpha=0.9, edgecolor="none")
    ins.set_yticks([])
    ins.tick_params(axis="x", labelsize=6, length=2, pad=1)
    for side in ("top", "right", "left"):
        ins.spines[side].set_visible(False)
    ins.spines["bottom"].set_linewidth(0.5)
    ins.patch.set_alpha(0.75)
    if centred:
        ins.axvline(0, color="black", linestyle="--", linewidth=0.7)
    return ins


def robust_max(da: xr.DataArray, q: float = 99.0) -> float:
    v = da.values[np.isfinite(da.values)]
    return float(np.percentile(v, q))


def save_figure(fig, stem: str) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  written -> {path}")


# =============================================================================
# 7. Path helpers
# =============================================================================

def member_path(model: str, scenario: str) -> str | None:
    """
    Per member QDM scenario file. Filenames encode each model's record length,
    so the year range is discovered rather than assumed.
    """
    import glob
    pattern = (f"{ROOT}/qdm_output/tas/{model}/"
               f"tas_day_{model}_{scenario}_qdm_*_0p1deg.nc")
    hits = sorted(glob.glob(pattern))
    if not hits:
        print(f"    missing: {model} {scenario}")
        return None
    return hits[0]


def ddpm_path(scenario: str) -> str:
    return PATHS["DDPM_HIST"] if scenario == "historical" else PATHS[f"DDPM_{scenario}"]


def pop_path(pathway: str, decade: int) -> str:
    d = PATHS["POP_SSP2_DIR"] if pathway.lower() == "ssp2" else PATHS["POP_SSP5_DIR"]
    return f"{d}/pop_{pathway.lower()}_{decade}_0p1deg_MME_Aligned.nc"


if __name__ == "__main__":
    set_style()
    print("cdd_common configuration")
    print(f"  CDD base            {T_BASE_CDD} C")
    print(f"  HDD base            {T_BASE_HDD} C")
    print(f"  baseline            {BASELINE[0]}-{BASELINE[1]}")
    print(f"  models              {len(MODELS)}")
    print(f"  scenarios           {SCENARIOS}")
    print(f"  cache               {CACHE}")
    print(f"  figures             {FIGDIR}")
    print("\nfile availability")
    for k, v in PATHS.items():
        print(f"  {'OK ' if os.path.exists(v) else 'MISSING'}  {k}")
    print("\nper member scenario files")
    for m in MODELS:
        row = [("OK" if member_path(m, s) else "--") for s in SCENARIOS]
        print(f"  {m:<16} {' '.join(row)}")
