from package_paths import SOURCE_ROOT, RUNS_ROOT, read_tile_scores
"""Build review tables from the complete, fixed temperature ablation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from package_paths import OUTPUT_ROOT as ROOT;E=ROOT/'evidence/temperature_analysis'
report=json.loads((E/'validation_report.json').read_text());assert report['runs']==18
summary=pd.read_csv(E/'paired_summary.csv',float_precision='round_trip')
rank=pd.read_csv(E/'ranking_changes.csv',float_precision='round_trip')
route=pd.read_csv(E/'routing_by_run.csv',float_precision='round_trip')
metric=pd.read_csv(E/'metrics_by_run.csv',float_precision='round_trip')
prob=pd.read_csv(E/'probability_by_event.csv',float_precision='round_trip')
ret=pd.read_csv(E/'retention_by_run.csv',float_precision='round_trip')

def get(block,arm,endpoint='pooled_miou',m='delta_aurc'):
    q=summary[(summary.block==block)&(summary.arm==arm)&(summary.endpoint==endpoint)&(summary.metric==m)]
    assert len(q)==1
    return q.iloc[0]

components=[]
for block in ['P1','P2']:
    for feature,label in [('uncertainty','Upper-tail entropy'),('top_tail_msp','Upper-tail MSP risk'),('spatial_inconsistency','Spatial inconsistency'),('soft_dice_risk_binary','Stored binary overlap'),('sdc_exact','Exact binary SDC')]:
        arm='t1_'+feature
        q=rank[(rank.block==block)&(rank.arm==arm)&(rank.split=='test')&(rank.event_id=='__whole_split__')]
        events=rank[(rank.block==block)&(rank.arm==arm)&(rank.split=='test')&(rank.event_id!='__whole_split__')]
        s=get(block,arm)
        components.append(dict(block=block,component=feature,label=label,median_spearman=q.spearman.median(),minimum_spearman=q.spearman.min(),
            mean_raw_score_change=q.mean_raw_score_change.mean(),mean_delta_pooled_miou_aurc=s['mean'],
            minimum_delta_aurc=s.minimum,maximum_delta_aurc=s.maximum,maximum_absolute_delta_aurc=max(abs(s.minimum),abs(s.maximum)),
            changed_event_sets_at_080=int((events.changed_members_at_080>0).sum()),event_runs=len(events)))
component=pd.DataFrame(components);component.to_csv(E/'component_sensitivity_table.csv',index=False)

composites=[];routing_pairs=[]
for block in ['P1','P2']:
    base=route[(route.block==block)&(route.arm=='cal_v2')].set_index(['partition','seed'])
    for arm,label in [('t1_v2_frozen_mapping','Frozen mapping and coefficients'),('t1_v2_new_ecdf_fixed_weights','New ECDF, fixed coefficients'),('t1_v2_refit','New ECDF and refitted coefficients')]:
        s=get(block,arm);tile=get(block,arm,'mean_tile_miou');point=get(block,arm,'pooled_miou','delta_risk_080')
        q=route[(route.block==block)&(route.arm==arm)].set_index(['partition','seed'])
        for key,r in q.iterrows():
            row=dict(block=block,partition=key[0],seed=key[1],arm=arm)
            for col in ['test_pooled_tile_coverage','test_event_macro_coverage','pooled_review_ratio','absolute_pooled_coverage_error','pooled_hazard_coverage']:
                row['delta_'+col]=r[col]-base.loc[key,col]
            routing_pairs.append(row)
        change=q.test_pooled_tile_coverage-base.test_pooled_tile_coverage
        composites.append(dict(block=block,arm=arm,label=label,mean_delta_tile_miou_aurc=tile['mean'],mean_delta_pooled_miou_aurc=s['mean'],
            min_delta_pooled_miou_aurc=s.minimum,max_delta_pooled_miou_aurc=s.maximum,mean_delta_pooled_risk_080=point['mean'],
            mean_delta_pooled_tile_retention=change.mean(),min_delta_pooled_tile_retention=change.min(),max_delta_pooled_tile_retention=change.max()))
composite=pd.DataFrame(composites);composite.to_csv(E/'composite_ablation_table.csv',index=False)
pd.DataFrame(routing_pairs).to_csv(E/'composite_routing_paired.csv',index=False)
route.groupby(['block','arm'])[['test_pooled_tile_coverage','test_event_macro_coverage','absolute_pooled_coverage_error','pooled_review_ratio']].agg(['mean','min','max']).to_csv(E/'routing_summary.csv')

pr=prob.groupby(['block','partition','seed','split','setting'])[['nll','brier']].mean().reset_index()
pr.to_csv(E/'probability_by_run.csv',index=False)
ps=[]
for (block,split),g in pr.groupby(['block','split']):
    for name in ['nll','brier']:
        pivot=g.pivot(index=['partition','seed'],columns='setting',values=name)
        d=pivot.t1-pivot.calibrated
        ps.append(dict(block=block,split=split,metric=name,calibrated_mean=pivot.calibrated.mean(),t1_mean=pivot.t1.mean(),
                       delta_t1_minus_calibrated=d.mean(),minimum_delta=d.min(),maximum_delta=d.max(),calibrated_lower_runs=int((d>0).sum())))
ps=pd.DataFrame(ps);ps.to_csv(E/'probability_summary.csv',index=False)

contrasts=[]
for block in ['P1','P2']:
    for condition,fit,base in [('calibrated','cal_v2','cal_sdc_exact'),('t1_refit','t1_v2_refit','t1_sdc_exact')]:
        for endpoint in ['mean_tile_miou','pooled_miou','mean_tile_dice','pooled_dice']:
            g=metric[(metric.block==block)&(metric.mask_basis=='stored_logits')&(metric.endpoint==endpoint)]
            a=g[g.arm==fit].set_index(['partition','seed']).aurc;b=g[g.arm==base].set_index(['partition','seed']).aurc
            d=a-b
            contrasts.append(dict(block=block,condition=condition,endpoint=endpoint,mean_delta_aurc=d.mean(),sd=d.std(ddof=1),minimum=d.min(),maximum=d.max(),negative=int((d<0).sum())))
contrasts=pd.DataFrame(contrasts);contrasts.to_csv(E/'pipeline_vs_exact_sdc.csv',index=False)

retpairs=[]
for (block,arm),g in ret[(ret.mask_basis=='stored_logits')&(ret.arm.str.startswith('t1_v2'))].groupby(['block','arm']):
    a=g.set_index(['partition','seed']);b=ret[(ret.block==block)&(ret.arm=='cal_v2')&(ret.mask_basis=='stored_logits')].set_index(['partition','seed'])
    for col in ['event_macro_area_coverage','event_macro_hazard_coverage']:
        d=a[col]-b[col];retpairs.append(dict(block=block,arm=arm,coverage_measure=col,mean_delta=d.mean(),minimum=d.min(),maximum=d.max()))
pd.DataFrame(retpairs).to_csv(E/'fixed_080_retention_summary.csv',index=False)

digest=dict(report=report,entropy=component[component.component=='uncertainty'].to_dict('records'),
            full_refit=composite[composite.arm=='t1_v2_refit'].to_dict('records'),
            matched_pipeline_contrasts=contrasts[contrasts.endpoint.isin(['mean_tile_miou','pooled_miou'])].to_dict('records'))
(E/'result_digest.json').write_text(json.dumps(digest,indent=2),encoding='utf-8')

print('Numerical summary tables written.')
