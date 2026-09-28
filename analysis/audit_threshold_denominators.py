from package_paths import SOURCE_ROOT, RUNS_ROOT, read_tile_scores
"""Reconcile event-macro coverage with actual pooled tile-count workload."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from package_paths import OUTPUT_ROOT as ROOT
RUNS=RUNS_ROOT
legacy=pd.read_csv(SOURCE_ROOT/'legacy_r4/threshold_coverage_runs.csv')
rows=[];event_rows=[]
for r in legacy.itertuples(index=False):
    run=RUNS/r.block/r.partition.lower()/f'seed_{r.seed}'/'risk_v2'
    calibration=json.loads((run/'calibration/calibration.json').read_text(encoding='utf-8'))
    threshold=calibration['thresholds']['georisk_v2']
    scores=read_tile_scores(run/'predictions/test_tile_scores.parquet')
    selected=scores.georisk_v2<=threshold
    n=len(scores);retained=int(selected.sum())
    event_coverages=[]
    for event,g in scores.assign(selected=selected).groupby('event_id'):
        c=float(g.selected.mean());event_coverages.append(c)
        event_rows.append(dict(block=r.block,partition=r.partition,seed=r.seed,event_id=event,tiles=len(g),retained=int(g.selected.sum()),coverage=c))
    macro=float(np.mean(event_coverages))
    assert np.isclose(macro,r.realized_coverage,atol=1e-12), (r.block,r.partition,r.seed,macro,r.realized_coverage)
    pooled=retained/n
    rows.append(dict(block=r.block,partition=r.partition,seed=r.seed,tiles=n,retained_tiles=retained,review_tiles=n-retained,
        event_macro_coverage=macro,pooled_tile_coverage=pooled,pooled_coverage_error=pooled-.8,
        event_macro_review_ratio=(1-macro)/.2,pooled_review_ratio=(1-pooled)/.2,threshold=threshold))
out=pd.DataFrame(rows)
out.to_csv(ROOT/'evidence/threshold_denominator_audit.csv',index=False)
pd.DataFrame(event_rows).to_csv(ROOT/'evidence/threshold_event_counts.csv',index=False)
print(out.to_string(index=False))
print('WORST POOLED WORKLOAD',out.loc[out.pooled_review_ratio.idxmax()].to_dict())
