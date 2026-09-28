from package_paths import SOURCE_ROOT, RUNS_ROOT, read_tile_scores
"""Deterministic revision summaries of existing, frozen study outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from package_paths import OUTPUT_ROOT as ROOT
SOURCE = SOURCE_ROOT
OUT = ROOT / 'evidence'
OUT.mkdir(exist_ok=True)

def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                q25=float(np.quantile(x,.25)), q75=float(np.quantile(x,.75)),
                minimum=float(x.min()), maximum=float(x.max()),
                negative=int((x<0).sum()), positive=int((x>0).sum()))

curves = pd.read_csv(SOURCE/'r5_endpoint_audit/endpoint_risk_coverage_by_run_event.csv')
curves = curves[curves['subset'].eq('all_tiles') & curves.score.isin(['georisk_v2','soft_dice_risk_binary'])]
print('EVENT IDENTIFIERS', sorted(curves.event_id.unique()))
keys = ['block','partition','seed','event_id','endpoint','coverage']
paired = curves.pivot(index=keys, columns='score', values='risk').reset_index()
assert paired[['georisk_v2','soft_dice_risk_binary']].notna().all().all()
paired['delta'] = paired.georisk_v2 - paired.soft_dice_risk_binary
rows=[]
for vals,g in paired.groupby(keys[:-1],sort=True):
    g=g.sort_values('coverage'); c=g.coverage.to_numpy(); d=g.delta.to_numpy()
    assert len(g)==19 and np.all(np.diff(c)>0)
    row=dict(zip(keys[:-1], vals))
    i=int(np.argmin(np.abs(c-.8)))
    assert np.isclose(c[i],.8)
    row.update(risk_v2_at_080=float(g.georisk_v2.iloc[i]),
               risk_self_overlap_at_080=float(g.soft_dice_risk_binary.iloc[i]),
               delta_at_080=float(d[i]),delta_aurc_010_100=float(np.trapezoid(d,c)))
    for low,high,label in [(.7,.9,'070_090'),(.8,1.,'080_100')]:
        mask=(c>=low-1e-8)&(c<=high+1e-8)
        integral=float(np.trapezoid(d[mask],c[mask]))
        row['delta_partial_aurc_'+label]=integral
        row['delta_mean_risk_'+label]=integral/(high-low)
    rows.append(row)
effects=pd.DataFrame(rows)
effects.to_csv(OUT/'operating_point_effects_by_run_event.csv',index=False)
macro=effects[effects.event_id.astype(str).str.startswith('__')]
if macro.empty:
    # The source's event-macro rows use this literal identifier.
    macro=effects[effects.event_id.eq('event_macro')]
assert not macro.empty, 'Inspect event identifiers before choosing a macro label'
event=effects[~effects.index.isin(macro.index)]
numeric=[c for c in effects if c.startswith(('delta_','risk_'))]
averaged=event.groupby(['block','event_id','endpoint'],as_index=False)[numeric].mean()
averaged.to_csv(OUT/'operating_point_seed_averaged_events.csv',index=False)
summaries=[]
for unit,frame in [('correlated_run_macro',macro),('seed_averaged_event',averaged)]:
    for (block,endpoint),g in frame.groupby(['block','endpoint']):
        for metric in numeric:
            summaries.append(dict(unit=unit,block=block,endpoint=endpoint,metric=metric,**describe(g[metric])))
pd.DataFrame(summaries).to_csv(OUT/'operating_point_summary.csv',index=False)

metrics=pd.read_csv(SOURCE/'consolidated/georisk_v2_metrics.csv')
external=[]
for block in ['E1','E2','E3']:
    base='self_dice_risk_multiclass' if block=='E1' else 'soft_dice_risk_binary'
    p=metrics[metrics.block.eq(block)&metrics.score.isin(['georisk_v2',base])].pivot(index='seed',columns='score',values='event_macro_aurc')
    assert len(p)==3 and not p.isna().any().any()
    for seed,r in p.iterrows():
        external.append(dict(block=block,seed=int(seed),comparator=base,v2=float(r.georisk_v2),baseline=float(r[base]),delta=float(r.georisk_v2-r[base])))
ex=pd.DataFrame(external)
ex.to_csv(OUT/'external_seed_effects.csv',index=False)
exsum=[dict(block=b,**describe(g.delta)) for b,g in ex.groupby('block')]
pd.DataFrame(exsum).to_csv(OUT/'external_descriptive_summary.csv',index=False)

routing=pd.read_csv(SOURCE/'legacy_r4/threshold_coverage_runs.csv')
routing['event_macro_review_fraction']=1-routing.realized_coverage
routing['event_macro_review_ratio']=routing.event_macro_review_fraction/(1-routing.target_coverage)
routing['coverage_denominator']='equal-weight event mean; not pooled tile count when events differ in size'
routing.to_csv(OUT/'threshold_routing_audit.csv',index=False)
worst=routing.loc[routing.absolute_coverage_error.idxmax()].to_dict()
checks={}
for b,expected in [('P1',-.03434),('P2',-.03998)]:
    v=float(macro[macro.block.eq(b)&macro.endpoint.eq('present_class_miou_risk')].delta_aurc_010_100.mean())
    checks[b]=dict(recomputed=v,published_rounded=expected,within_rounding=bool(abs(v-expected)<1e-5))
assert all(v['within_rounding'] for v in checks.values())
report={'source':'frozen submitted tables','analysis_status':'post_hoc_revision_summary','partial_range':'0.70–0.90; sensitivity 0.80–1.00','published_effect_checks':checks,'worst_event_macro_routing_run':worst,'actual_tile_counts':'See threshold_denominator_audit.csv and threshold_event_counts.csv from the original score files','external':exsum}
(OUT/'revision_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
print(pd.DataFrame(summaries).query("unit == 'correlated_run_macro' and metric in ['delta_at_080','delta_mean_risk_070_090']").to_string(index=False))
