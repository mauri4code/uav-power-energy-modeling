import importlib.util, glob, os, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
spec = importlib.util.spec_from_file_location('a', 'exp_arming_window.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

def build(margin):
    out = []
    for f in sorted(glob.glob(os.path.join(A.find_flights_dir(), 'F*', 'flight_resampled.csv'))):
        d = pd.read_csv(f).sort_values('timestamp').copy()
        thr = 0.5*(np.percentile(d.power,5)+np.percentile(d.power,95))
        d['t'] = d.timestamp - d.timestamp.iloc[0]; d['sec'] = np.floor(d.t).astype(int)
        a = d.groupby('sec', as_index=False)[A.MOTORS+['payload_mass','power','t']].mean()
        a['flight_id'] = d.flight_id.iloc[0]
        hi = (a.power > thr).values; i0, i1 = hi.argmax(), len(hi)-1-hi[::-1].argmax()
        idx = np.arange(len(a)); a['phase'] = np.where(hi, 'cruise', 'other')
        out.append(a[(idx >= i0-margin) & (idx <= i1+margin)])
    return pd.concat(out, ignore_index=True)

rows = []
for mg in [0, 3, 5, 8, 10, 12, 15, 20, 30]:
    ds = build(mg); p = A.evaluate(ds); y = ds.power.values; ck = (ds.phase=='cruise').values
    r = dict(margin_s=mg, rows=len(ds), pct_other=round(100*(~ck).mean(),1),
             r2=round(r2_score(y,p),3), mae=round(mean_absolute_error(y,p),1),
             cruise=round(mean_absolute_error(y[ck],p[ck]),1),
             other=round(mean_absolute_error(y[~ck],p[~ck]),1) if (~ck).any() else np.nan)
    rows.append(r)
    print(f"{mg:4d} s  {len(ds):5d} rows  {r['pct_other']:5.1f}% other   "
          f"R2 {r['r2']:6.3f}   MAE {r['mae']:5.1f}   cruise {r['cruise']:5.1f}   "
          f"other {r['other']:6.1f}", flush=True)
pd.DataFrame(rows).to_csv('margin_sweep.csv', index=False)
print('saved -> margin_sweep.csv')
