#!/usr/bin/env python3
"""Out-of-sample validation of Equation (4), using MSWX only."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr, theilslopes
import cdd_common as cc

BASE=cc.T_BASE_CDD
BAND=1.0
LAND_FRACTION_MIN=0.99
PRIMARY_SPLITS=[("1985-1999",(1985,1999),"2000-2014",(2000,2014)),
                ("2000-2014",(2000,2014),"1985-1999",(1985,1999))]
ROLLING_TEST_WINDOWS=[(1995,1999),(2000,2004),(2005,2009),(2010,2014)]
FACTORS=[5,10]
OUTDIR=Path(cc.FIGDIR)
CSV_NAME="Jha_etal_Eq4_OOS_validation.csv"
NC_NAME="eq4_oos_validation.nc"
FIG_NAME="Jha_etal_Eq4_OOS_validation"

def block_mean(da,f): return da.coarsen(lat=f,lon=f,boundary="trim").mean()

def prepare(gdf):
    tas=cc.open_tas(cc.PATHS["MSWX"],label="MSWX",years=cc.BASELINE)
    years=cc.complete_years(tas,verbose=True)
    tas=tas.sel(time=tas.time.dt.year.isin(years))
    tas=cc.clip_india(tas,gdf).persist()
    land=tas.isel(time=0,drop=True).notnull()
    print(f"Retained land cells: {int(land.sum())}")
    return tas,land

def density(tas,land):
    inside=((tas>BASE-BAND)&(tas<BASE+BAND)).astype("float32").where(land)
    return (inside.mean("time")/(2*BAND)).compute()

def variance(tas,f):
    m1=block_mean(tas,f); m2=block_mean(tas**2,f)
    return (m2-m1**2).clip(min=0).mean("time").compute()

def keepmask(land,f): return block_mean(land.astype("float32"),f)>=LAND_FRACTION_MIN

def annual_cdd(tas): return cc.annual_cdd(tas,base=BASE)

def train_prediction(train,land,f):
    dens=density(train,land)
    var=variance(train,f)
    db=block_mean(dens,f)
    nd=float(train.time.groupby("time.year").count().mean().compute())
    pred=(0.5*nd*db*var).where(keepmask(land,f))
    return pred,db,var,nd

def observed_gap(test,land,f):
    fine=annual_cdd(test).mean("year").compute()
    tb=block_mean(test,f)
    coarse=annual_cdd(tb).mean("year").compute()
    gap=(block_mean(fine,f)-coarse).where(keepmask(land,f))
    return gap

def pair(a,b):
    av=np.asarray(a.values).ravel(); bv=np.asarray(b.values).ravel()
    m=np.isfinite(av)&np.isfinite(bv)
    return av[m],bv[m]

def origin_slope(x,y):
    return float(np.dot(x,y)/np.dot(x,x))

def stats(x,y):
    rho,p=spearmanr(x,y)
    ts,inter,lo,hi=theilslopes(y,x)
    r=y-x
    return dict(n_blocks=len(x),spearman_rho=float(rho),spearman_p=float(p),
                slope_origin=origin_slope(x,y),theil_sen_slope=float(ts),
                theil_sen_intercept=float(inter),theil_sen_lo=float(lo),theil_sen_hi=float(hi),
                rmse_degC_days=float(np.sqrt(np.mean(r*r))),
                mae_degC_days=float(np.mean(np.abs(r))),
                mean_error_degC_days=float(np.mean(r)),
                observed_mean_degC_days=float(np.mean(y)),
                predicted_mean_degC_days=float(np.mean(x)))

def run_split(tas,land,train_period,test_period,f,validation):
    train=tas.sel(time=slice(f"{train_period[0]}-01-01",f"{train_period[1]}-12-31"))
    test=tas.sel(time=slice(f"{test_period[0]}-01-01",f"{test_period[1]}-12-31"))
    pred,db,var,nd=train_prediction(train,land,f)
    obs=observed_gap(test,land,f)
    x,y=pair(pred,obs)
    s=stats(x,y)
    s.update(validation=validation,train_start=train_period[0],train_end=train_period[1],
             test_start=test_period[0],test_end=test_period[1],effective_resolution_deg=f/10,
             base_temperature_degC=BASE,density_band_half_width_degC=BAND,
             training_mean_days_per_year=nd,training_mean_density_degC_inv=float(db.mean()),
             training_mean_variance_degC2=float(var.mean()))
    print(f"{validation} {f/10:.1f}°: n={len(x)}, rho={s['spearman_rho']:.3f}, origin={s['slope_origin']:.3f}, TS={s['theil_sen_slope']:.3f}")
    return s,x,y

def main():
    cc.set_style(); OUTDIR.mkdir(parents=True,exist_ok=True)
    gdf=cc.load_boundary(); tas,land=prepare(gdf)
    rows=[]; primary={}
    for sl,train,tl,test in PRIMARY_SPLITS:
        name=f"{sl}_to_{tl}"
        for f in FACTORS:
            s,x,y=run_split(tas,land,train,test,f,name)
            rows.append(s); primary[(name,f)]=(x,y)
    for f in FACTORS:
        for ts,te in ROLLING_TEST_WINDOWS:
            train=(1985,ts-1); test=(ts,te)
            name=f"rolling_{ts}_{te}"
            s,_,_=run_split(tas,land,train,test,f,name); rows.append(s)
    df=pd.DataFrame(rows); csv=OUTDIR/CSV_NAME; df.to_csv(csv,index=False,float_format="%.6f")
    print(f"Written {csv}")
    # Primary diagnostic figure
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2))
    order=[("1985-1999_to_2000-2014",10),("2000-2014_to_1985-1999",10),
           ("1985-1999_to_2000-2014",5),("2000-2014_to_1985-1999",5)]
    for ax,(name,f) in zip(axs.flat,order):
        x,y=primary[(name,f)]; lim=max(float(np.nanpercentile(np.r_[x,y],99.5)),1)
        sl=origin_slope(x,y); rho=spearmanr(x,y)[0]
        ax.scatter(x,y,s=8,alpha=.45,edgecolor="none"); ax.plot([0,lim],[0,lim],'--',lw=.8); ax.plot([0,lim],[0,sl*lim],lw=1.2)
        ax.set(xlim=(0,lim),ylim=(0,lim),xlabel="Predicted gap (°C days)",ylabel="Observed gap (°C days)")
        ax.grid(alpha=.25,lw=.4); ax.text(.04,.96,f"ρ = {rho:.2f}\nn = {len(x)}",transform=ax.transAxes,va="top",fontsize=7)
        direction="1985–1999 → 2000–2014" if name.startswith("1985") else "2000–2014 → 1985–1999"
        ax.set_title(f"{direction}, {f/10:.1f}°",loc="left",fontweight="bold",fontsize=9)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(OUTDIR/f"{FIG_NAME}.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)
    print("\nPRIMARY RESULTS")
    print(df[df.validation.str.contains("_to_")][['validation','effective_resolution_deg','n_blocks','spearman_rho','slope_origin','theil_sen_slope','theil_sen_lo','theil_sen_hi','rmse_degC_days']].to_string(index=False,float_format=lambda z:f"{z:.3f}"))

if __name__=="__main__": main()
