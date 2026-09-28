from package_paths import SOURCE_ROOT, RUNS_ROOT, read_tile_scores
"""Validate calibrated replay and quantify temperature sensitivity without test tuning."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from compare_primary_targets import COLUMNS,FEATURES,GRID,ecdf_design,fit_score,losses,evaluate

from package_paths import OUTPUT_ROOT as ROOT
OUT=ROOT/'evidence/temperature_analysis';OUT.mkdir(exist_ok=True)
REPLAY=ROOT/'evidence/temperature_replay'
ORIGINAL=RUNS_ROOT
PRIMITIVES=['uncertainty','top_tail_msp','spatial_inconsistency','soft_dice_risk_binary','sdc_exact']
PAIRS={**{f't1_{c}':f'cal_{c}' for c in PRIMITIVES},
       't1_v2_frozen_mapping':'cal_v2','t1_v2_new_ecdf_fixed_weights':'cal_v2','t1_v2_refit':'cal_v2'}
FIT_ATOL=1e-12

def quantile(values,events):
    weights=np.zeros(len(values),dtype=np.float64)
    unique=np.unique(events)
    for e in unique:
        mask=events==e;weights[mask]=1/(len(unique)*mask.sum())
    order=np.argsort(values)
    index=int(np.searchsorted(np.cumsum(weights[order]),.8,side='left'))
    return float(values[order[min(index,len(values)-1)]])

def weighted_score(values,reference,events,w):
    x=ecdf_design(values,reference,events)
    return np.asarray([r@w for r in x])

def spearman(first,second):
    # Same average-rank Pearson definition, avoiding a stalled NumPy cov/BLAS path.
    a=rankdata(first,method='average');b=rankdata(second,method='average')
    a=a-a.mean();b=b-b.mean()
    denominator=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return float(np.sum(a*b)/denominator) if denominator>0 else np.nan

def ranks(old,new,events,key,arm,split):
    rows=[]
    for event in ['__whole_split__',*np.unique(events)]:
        ix=np.arange(len(events)) if event=='__whole_split__' else np.flatnonzero(events==event)
        o=np.argsort(old[ix],kind='stable');n=np.argsort(new[ix],kind='stable')
        count=len(ix);ks=np.clip(np.ceil(GRID*count).astype(int),1,count)
        k=ks[np.argmin(np.abs(GRID-.8))]
        corr=spearman(old[ix],new[ix])
        rows.append(dict(**key,split=split,arm=arm,event_id=event,tiles=count,spearman=corr,
                         mean_raw_score_change=float(np.mean(new[ix]-old[ix])),max_absolute_score_change=float(np.max(np.abs(new[ix]-old[ix]))),
                         changed_order_positions=int(np.count_nonzero(o!=n)),order_changed=not np.array_equal(o,n),
                         changed_retained_sets_grid=int(sum(set(o[:j])!=set(n[:j]) for j in ks)),
                         changed_members_at_080=len(set(o[:k])^set(n[:k]))))
    return rows

def routing(cal,test,cal_events,test_events,threshold,key,arm,conf):
    keep=test<=threshold;ck=cal<=threshold
    event_rows=[]
    for e in np.unique(test_events):
        ix=test_events==e
        event_rows.append(dict(**key,arm=arm,event_id=e,tiles=int(ix.sum()),retained=int(keep[ix].sum()),coverage=float(keep[ix].mean())))
    row=dict(**key,arm=arm,threshold=threshold,
             calibration_event_macro_coverage=float(np.mean([ck[cal_events==e].mean() for e in np.unique(cal_events)])),
             test_event_macro_coverage=float(np.mean([r['coverage'] for r in event_rows])),
             test_pooled_tile_coverage=float(keep.mean()),test_tiles=len(keep),retained_tiles=int(keep.sum()),review_tiles=int((~keep).sum()),
             pooled_review_ratio=float((1-keep.mean())/.2),
             pooled_hazard_coverage=float(conf[keep,2:].sum()/conf[:,2:].sum()))
    row['absolute_pooled_coverage_error']=abs(row['test_pooled_tile_coverage']-.8)
    return row,event_rows,keep

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    files=sorted(p for p in REPLAY.glob('P[12]_F[123]_seed*.csv') if p.with_suffix('.json').exists())
    if not args.allow_partial:assert len(files)==18,len(files)
    checks=[];rank_rows=[];curve_rows=[];metric_rows=[];retention_rows=[];routing_rows=[];routing_events=[];weights=[];probability=[];score_rows=[];precision=[]
    for file in files:
        meta=json.loads(file.with_suffix('.json').read_text());key={k:meta[k] for k in ['block','partition','seed']}
        run=ORIGINAL/key['block']/key['partition'].lower()/f'seed_{key["seed"]}'/'risk_v2'
        saved=json.loads((run/'calibration/calibration.json').read_text())
        replay=pd.read_csv(file,float_precision='round_trip')
        comparator=pd.read_csv(ROOT/'evidence/comparator_replay'/file.name,float_precision='round_trip')
        frames={};raw={}
        for split in ['calibration','test']:
            f=read_tile_scores(run/f'predictions/{split}_tile_scores.parquet')
            ids=f.sample_id.astype(str).tolist();frames[split]=f
            old=comparator[comparator.split==split].set_index('sample_id').loc[ids].reset_index()
            for setting in ['calibrated','t1']:
                r=replay[(replay.split==split)&(replay.setting==setting)].sort_values('source_order').reset_index(drop=True)
                assert r.sample_id.astype(str).tolist()==ids and r.event_id.astype(str).tolist()==f.event_id.astype(str).tolist()
                assert np.isfinite(r[PRIMITIVES].to_numpy()).all()
                raw[split,setting]=r
                for event,g in r.groupby('event_id'):
                    probability.append(dict(**key,split=split,setting=setting,event_id=event,pixels=int(g.reference_valid_pixels.sum()),
                                            nll=float(g.nll_sum.sum()/g.reference_valid_pixels.sum()),brier=float(g.brier_sum.sum()/g.reference_valid_pixels.sum())))
            c=raw[split,'calibrated'];t=raw[split,'t1']
            row=dict(**key,split=split,records=len(f),temperature=saved['temperature'])
            for feature in PRIMITIVES:
                original=old.sdc_float64 if feature=='sdc_exact' else f[feature]
                row[feature+'_replay_error']=float(np.max(np.abs(c[feature].to_numpy()-original.to_numpy())))
                assert row[feature+'_replay_error']==0,row
            row['cal_confusion_mismatches']=int(np.count_nonzero(c[COLUMNS].to_numpy()-old[COLUMNS].to_numpy()))
            assert row['cal_confusion_mismatches']==0
            row['temperature_changed_hard_pixels']=int(c.hard_disagreement_all_pixels.sum())
            row['temperature_changed_reference_valid_hard_pixels']=int(c.hard_disagreement_reference_valid.sum())
            row['temperature_changed_tile_losses']=int(np.count_nonzero(losses(c[COLUMNS])[:,0]-losses(t[COLUMNS])[:,0]))
            checks.append(row)
            for col in COLUMNS:
                frames[split][col]=c[col].to_numpy()
                frames[split]['raw_'+col]=t[col].to_numpy()
        if args.validate_only:print('Validated',key,flush=True);continue
        cal=frames['calibration'];test=frames['test'];ce=cal.event_id.astype(str).to_numpy();te=test.event_id.astype(str).to_numpy()
        cf=cal[FEATURES].to_numpy(dtype=float,copy=True);tf=test[FEATURES].to_numpy(dtype=float,copy=True)
        fitted,w=fit_score(cf,tf,ce,cal.tile_loss.to_numpy())
        score_error=float(np.max(np.abs(fitted-test.georisk_v2.to_numpy())))
        weight_error=float(np.max(np.abs(w-np.asarray(saved['adaptive_fit']['weights']))))
        assert score_error<=FIT_ATOL and weight_error<=FIT_ATOL
        cal_fitted=weighted_score(cf,cf,ce,w)
        cal_error=float(np.max(np.abs(cal_fitted-cal.georisk_v2.to_numpy())))
        assert cal_error<=FIT_ATOL
        # Floating-point fits may vary across BLAS versions. Selection must not.
        for actual,original in [(fitted,test.georisk_v2.to_numpy()),(cal_fitted,cal.georisk_v2.to_numpy())]:
            assert np.array_equal(np.argsort(actual,kind='stable'),np.argsort(original,kind='stable'))
        reproduced_keep=fitted<=quantile(cal_fitted,ce)
        original_keep=test.georisk_v2.to_numpy()<=saved['thresholds']['georisk_v2']
        assert np.array_equal(reproduced_keep,original_keep)
        precision.append(dict(**key,test_score_error=score_error,calibration_score_error=cal_error,
                              coefficient_error=weight_error,order_exact=True,retained_set_exact=True))
        tc=cf.copy();tt=tf.copy()
        for j,feature in enumerate(FEATURES):
            if feature=='representation_shift':continue
            tc[:,j]=raw['calibration','t1'][feature];tt[:,j]=raw['test','t1'][feature]
        scores={'cal_v2':fitted};cscores={'cal_v2':cal_fitted}
        for setting,prefix in [('calibrated','cal'),('t1','t1')]:
            for feature in PRIMITIVES:
                scores[prefix+'_'+feature]=raw['test',setting][feature].to_numpy()
                cscores[prefix+'_'+feature]=raw['calibration',setting][feature].to_numpy()
        for arm,ref in [('t1_v2_frozen_mapping',cf),('t1_v2_new_ecdf_fixed_weights',tc)]:
            scores[arm]=weighted_score(tt,ref,ce,w);cscores[arm]=weighted_score(tc,ref,ce,w)
            weights.append(dict(**key,arm=arm,overlap=w[0],entropy=w[1],shift=w[2],spatial=w[3]))
        target=losses(raw['calibration','t1'][COLUMNS])[:,0]
        scores['t1_v2_refit'],nw=fit_score(tc,tt,ce,target)
        cscores['t1_v2_refit']=weighted_score(tc,tc,ce,nw)
        for arm,ww in [('cal_v2',w),('t1_v2_refit',nw)]:weights.append(dict(**key,arm=arm,overlap=ww[0],entropy=ww[1],shift=ww[2],spatial=ww[3]))
        for arm,base in PAIRS.items():
            rank_rows.extend(ranks(scores[base],scores[arm],te,key,arm,'test'))
            rank_rows.extend(ranks(cscores[base],cscores[arm],ce,key,arm,'calibration'))
        c,m,r=evaluate(test,scores,key,'stored_logits');curve_rows.extend(c);metric_rows.extend(m);retention_rows.extend(r)
        if np.any(raw['test','calibrated'].hard_disagreement_reference_valid.to_numpy()>0):
            own={k:v for k,v in scores.items() if k.startswith('t1_')}
            c,m,r=evaluate(test,own,key,'original_audit')
            for group in [c,m,r]:
                for item in group:item['mask_basis']='t1_prediction_sensitivity'
            curve_rows.extend(c);metric_rows.extend(m);retention_rows.extend(r)
        score_frame=test[['sample_id','event_id']].copy()
        for k,v in key.items():score_frame[k]=v
        for arm,values in scores.items():
            threshold=saved['thresholds']['georisk_v2'] if arm=='t1_v2_frozen_mapping' else quantile(cscores[arm],ce)
            if arm=='cal_v2':assert abs(threshold-saved['thresholds']['georisk_v2'])<=FIT_ATOL
            if arm.startswith('cal_') and arm[4:] in saved['thresholds']:assert threshold==saved['thresholds'][arm[4:]]
            r,es,keep=routing(cscores[arm],values,ce,te,threshold,key,arm,test[COLUMNS].to_numpy())
            routing_rows.append(r);routing_events.extend(es);score_frame[arm]=values;score_frame[arm+'_threshold_retained']=keep
        score_rows.append(score_frame)
        print('Validated and analyzed',key,flush=True)
    stem='partial_' if len(files)!=18 else ''
    pd.DataFrame(checks).to_csv(OUT/f'{stem}fidelity_by_run_split.csv',index=False)
    if args.validate_only:print(pd.DataFrame(checks).to_string(index=False));return
    pd.DataFrame(precision).to_csv(OUT/f'{stem}fit_precision.csv',index=False)
    metric=pd.DataFrame(metric_rows);route=pd.DataFrame(routing_rows)
    for name,frame in [('risk_curves',pd.DataFrame(curve_rows)),('metrics_by_run',metric),('retention_by_run',pd.DataFrame(retention_rows)),
                       ('ranking_changes',pd.DataFrame(rank_rows)),('routing_by_run',route),('routing_by_event',pd.DataFrame(routing_events)),
                       ('fitted_weights',pd.DataFrame(weights)),('probability_by_event',pd.DataFrame(probability)),('test_scores',pd.concat(score_rows,ignore_index=True))]:
        frame.to_csv(OUT/f'{stem}{name}.csv',index=False)
    paired=[]
    for arm,base in PAIRS.items():
        left=metric[(metric.arm==arm)&(metric.mask_basis=='stored_logits')].set_index(['block','partition','seed','endpoint'])
        right=metric[(metric.arm==base)&(metric.mask_basis=='stored_logits')].set_index(left.index.names)
        for index,row in left.iterrows():
            paired.append(dict(zip(left.index.names,index),arm=arm,baseline=base,**{'delta_'+k:float(row[k]-right.loc[index,k]) for k in ['aurc','risk_080','mean_risk_070_090']}))
    paired=pd.DataFrame(paired);paired.to_csv(OUT/f'{stem}paired_effects_by_run.csv',index=False)
    summary=[]
    for (block,arm,endpoint),g in paired.groupby(['block','arm','endpoint']):
        for m in ['delta_aurc','delta_risk_080','delta_mean_risk_070_090']:
            x=g[m]
            summary.append(dict(block=block,arm=arm,endpoint=endpoint,metric=m,n=len(x),mean=x.mean(),sd=x.std(ddof=1),median=x.median(),minimum=x.min(),maximum=x.max(),negative=int((x<0).sum())))
    pd.DataFrame(summary).to_csv(OUT/f'{stem}paired_summary.csv',index=False)
    # Cross-gates against completed prior analyses: calibrated SDC and v2 are identical.
    oldmetric=pd.read_csv(ROOT/'evidence/comparator_analysis/metrics_by_run.csv',float_precision='round_trip')
    errs=[]
    for newarm,oldarm in [('cal_v2','frozen_v2'),('cal_soft_dice_risk_binary','legacy_self_overlap'),('cal_sdc_exact','sdc_exact')]:
        left=metric[(metric.arm==newarm)&(metric.mask_basis=='stored_logits')].set_index(['block','partition','seed','endpoint'])
        right=oldmetric[(oldmetric.arm==oldarm)&(oldmetric.mask_basis=='stored_logits')].set_index(left.index.names)
        errs.extend(np.abs(left.aurc-right.loc[left.index].aurc).tolist())
    assert max(errs)==0,max(errs)
    oldroute=pd.read_csv(ROOT/'evidence/threshold_denominator_audit.csv',float_precision='round_trip')
    for row in route[route.arm=='cal_v2'].itertuples():
        g=oldroute[(oldroute.block==row.block)&(oldroute.partition==row.partition)&(oldroute.seed==row.seed)]
        assert len(g)==1 and row.retained_tiles==int(g.retained_tiles.iloc[0])
    check=pd.DataFrame(checks)
    report=dict(status='complete primary temperature ablation' if len(files)==18 else 'partial',runs=len(files),
        records_by_split=check.groupby('split').records.sum().to_dict(),temperature_min=float(check.temperature.min()),temperature_max=float(check.temperature.max()),
        max_calibrated_component_replay_error=float(check[[x for x in check if x.endswith('_replay_error')]].to_numpy().max()),
        hard_pixel_changes=int(check.temperature_changed_hard_pixels.sum()),reference_valid_hard_pixel_changes=int(check.temperature_changed_reference_valid_hard_pixels.sum()),
        calibrated_comparator_aurc_replay_error=max(errs),original_v2_fit_reproduced=True,original_v2_routing_counts_reproduced=True,
        fit_absolute_tolerance=FIT_ATOL,max_fit_error=max(max(r[k] for k in ['test_score_error','calibration_score_error','coefficient_error']) for r in precision),
        original_v2_rank_order_and_retained_set_exact=True,
        scope='18 primary binary runs; calibration-only ECDF and coefficient fitting; descriptive correlated-run summaries')
    (OUT/f'{stem}validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
