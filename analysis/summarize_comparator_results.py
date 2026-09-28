from package_paths import SOURCE_ROOT, RUNS_ROOT, read_tile_scores
"""Make reviewable tables from the fixed 18-run comparator analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from package_paths import OUTPUT_ROOT as ROOT
OUT=ROOT/'evidence/comparator_analysis'
summary=pd.read_csv(OUT/'paired_summary.csv',float_precision='round_trip')
metrics=pd.read_csv(OUT/'metrics_by_run.csv',float_precision='round_trip')
fidelity=pd.read_csv(OUT/'fidelity_by_run_split.csv',float_precision='round_trip')
ordering=pd.read_csv(OUT/'ordering_changes.csv')
validation=json.loads((OUT/'validation_report.json').read_text())
retention=pd.read_csv(OUT/'retention_by_run.csv',float_precision='round_trip')

def cell(block,arm,endpoint,metric='delta_aurc',basis='stored_logits'):
    q=summary[(summary.block==block)&(summary.arm==arm)&(summary.endpoint==endpoint)&(summary.metric==metric)&(summary.mask_basis==basis)]
    assert len(q)==1
    return q.iloc[0]

rows=[]
for block in ['P1','P2']:
    for arm,label,target in [('frozen_v2','Frozen original v2','miou'),('fit_miou_exact','Matched mIoU','miou'),('fit_dice_exact','Matched foreground Dice','dice'),('fit_iou_exact','Matched foreground IoU','iou')]:
        a=cell(block,arm,'mean_tile_'+target);b=cell(block,arm,'pooled_'+target)
        rows.append(dict(block=block,arm=arm,fitting_target=label,
                         tile_mean_delta_aurc=a['mean'],pooled_delta_aurc=b['mean'],
                         tile_mean_sd=a.sd,pooled_sd=b.sd,
                         tile_mean_favoring=int(a.favoring),pooled_favoring=int(b.favoring),
                         tile_mean_delta_at_080=cell(block,arm,'mean_tile_'+target,'delta_risk_080')['mean'],
                         pooled_delta_at_080=cell(block,arm,'pooled_'+target,'delta_risk_080')['mean']))
table=pd.DataFrame(rows);table.to_csv(OUT/'key_comparison_table.csv',index=False)

frows=[]
for block,g in fidelity.groupby('block'):
    test=ordering[(ordering.block==block)&(ordering.split=='test')&(ordering.arm=='sdc_exact')]
    paired=metrics[(metrics.block==block)&(metrics.mask_basis=='stored_logits')&(metrics.arm.isin(['legacy_self_overlap','sdc_exact']))]
    pivot=paired.pivot(index=['partition','seed','endpoint'],columns='arm',values='aurc')
    delta=pivot.sdc_exact-pivot.legacy_self_overlap
    test_f=g[g.split=='test'];cal_f=g[g.split=='calibration']
    frows.append(dict(block=block,calibration_records=int(cal_f.records.sum()),test_records=int(test_f.records.sum()),
                      empty_prediction_calibration=int(cal_f.prediction_empty.sum()),empty_prediction_test=int(test_f.prediction_empty.sum()),
                      zero_denominators=int(g.zero_denominator.sum()),max_native_difference=g.exact_native_max_score_difference.max(),
                      max_float64_difference=g.exact_float64_max_score_difference.max(),
                      test_event_runs=len(test),changed_test_orders=int(test.order_changed.sum()),
                      changed_retained_sets_at_080=int((test.changed_members_at_080>0).sum()),
                      changed_members_at_080=int(test.changed_members_at_080.sum()),
                      max_absolute_aurc_change_all_endpoints=float(delta.abs().max())))
fidelity_summary=pd.DataFrame(frows);fidelity_summary.to_csv(OUT/'fidelity_summary.csv',index=False)

# Precision sensitivity has the same score vector in both evaluations.
mp=metrics.pivot(index=['block','partition','seed','arm','endpoint'],columns='mask_basis',values='aurc')
mp['delta_aurc_raw_minus_stored']=mp.original_audit-mp.stored_logits
mp.to_csv(OUT/'mask_precision_sensitivity.csv')
sp=summary.pivot(index=['block','arm','endpoint','metric'],columns='mask_basis',values='mean')
sp['mean_effect_change_raw_minus_stored']=sp.original_audit-sp.stored_logits
sp.to_csv(OUT/'paired_precision_sensitivity.csv')

# Exact SDC versus epsilon version, with native and float64 unsmoothed reductions.
fx=[]
for (block,basis),g in metrics.groupby(['block','mask_basis']):
    for arm in ['sdc_native','sdc_exact']:
        left=g[g.arm==arm].set_index(['partition','seed','endpoint'])
        right=g[g.arm=='legacy_self_overlap'].set_index(['partition','seed','endpoint'])
        for endpoint in left.index.get_level_values('endpoint').unique():
            for metric in ['aurc','risk_080','mean_risk_070_090']:
                d=left.xs(endpoint,level='endpoint')[metric]-right.xs(endpoint,level='endpoint')[metric]
                fx.append(dict(block=block,mask_basis=basis,arm=arm,endpoint=endpoint,metric=metric,
                               mean=d.mean(),min=d.min(),max=d.max(),max_absolute=d.abs().max()))
fx=pd.DataFrame(fx);fx.to_csv(OUT/'sdc_replacement_effects.csv',index=False)

rr=[]
for (block,basis),g in retention.groupby(['block','mask_basis']):
    base=g[g.arm=='sdc_exact'].set_index(['partition','seed'])
    for arm,h in g.groupby('arm'):
        if arm=='sdc_exact':continue
        h=h.set_index(['partition','seed'])
        for col in ['event_macro_tile_coverage','event_macro_area_coverage','event_macro_hazard_coverage']:
            d=h[col]-base[col]
            rr.append(dict(block=block,mask_basis=basis,arm=arm,coverage_measure=col,mean_delta=d.mean(),min_delta=d.min(),max_delta=d.max()))
rr=pd.DataFrame(rr);rr.to_csv(OUT/'retention_paired_summary.csv',index=False)

# Check native versus float64 equations do not lead to different test metrics.
npiv=metrics[metrics.arm.isin(['sdc_native','sdc_exact'])].pivot(index=['block','partition','seed','mask_basis','endpoint'],columns='arm',values='aurc')
native_exact_max=float((npiv.sdc_native-npiv.sdc_exact).abs().max())
extra={'native_vs_float64_max_aurc_difference':native_exact_max,
       'max_mask_precision_aurc_change':float(mp.delta_aurc_raw_minus_stored.abs().max()),
       'max_mask_precision_block_mean_effect_change':float(sp.mean_effect_change_raw_minus_stored.abs().max()),
       'test_event_runs_per_arm':int(len(ordering[(ordering.split=='test')&(ordering.arm=='sdc_exact')])),
       'distinct_test_tile_ids':int(pd.read_csv(OUT/'test_scores.csv').sample_id.nunique()),
       'note':'Records repeat shared tiles across seeds and model blocks; 2640 test records are not independent samples.'}
(OUT/'result_digest.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')


print('Numerical summary tables written.')
