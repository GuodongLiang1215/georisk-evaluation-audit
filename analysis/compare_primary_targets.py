from package_paths import SOURCE_ROOT, RUNS_ROOT, read_tile_scores
"""Validate frozen replay and compare calibration-only target-aligned fits.

Run in the existing georisk environment in a process separate from CUDA replay.
All original sources are read-only. No test outcomes select features or settings.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from package_paths import OUTPUT_ROOT as ROOT
REPLAY=ROOT/'evidence/comparator_replay'
OUT=ROOT/'evidence/comparator_analysis'
OUT.mkdir(exist_ok=True)
ORIGINAL=RUNS_ROOT
SOURCE=SOURCE_ROOT/'r5_endpoint_audit'
GRID=np.arange(.10,1.001,.05)
COLUMNS=['confusion_0_0','confusion_0_1','confusion_1_0','confusion_1_1']
FEATURES=['soft_dice_risk_binary','uncertainty','representation_shift','spatial_inconsistency']
ENDPOINTS=['mean_tile_miou','mean_tile_dice','mean_tile_iou','pooled_miou','pooled_dice','pooled_iou']

def losses(conf):
    c=np.asarray(conf,dtype=float).reshape(-1,2,2)
    tp=np.diagonal(c,axis1=1,axis2=2)
    union=c.sum(axis=1)+c.sum(axis=2)-tp
    iou=np.divide(tp,union,out=np.zeros_like(tp),where=union>0)
    miou=(iou.sum(axis=1)/(union>0).sum(axis=1))
    denom=c[:,1,:].sum(axis=1)+c[:,:,1].sum(axis=1)
    dice=np.divide(2*tp[:,1],denom,out=np.zeros(len(c)),where=denom>0)
    return np.column_stack([1-miou,1-dice,1-iou[:,1]])

def ecdf_design(values,reference,events):
    # Equivalent to the original equal-event ECDF with inclusive <= ties.
    matrix=np.zeros((len(values),reference.shape[1]))
    unique=np.unique(events)
    for j in range(reference.shape[1]):
        columns=[]
        for e in unique:
            ref=reference[events==e,j]
            columns.append(np.mean(ref[:,None]<=values[:,j][None,:],axis=0))
        matrix[:,j]=np.mean(columns,axis=0)
    return matrix

def fit_score(calibration,test,events,target):
    x=ecdf_design(calibration,calibration,events)
    z=ecdf_design(test,calibration,events)
    sw=np.ones(len(events))
    for e in np.unique(events):sw[events==e]=1/np.count_nonzero(events==e)
    sw/=sw.mean()
    design=x*np.sqrt(sw)[:,None];y=np.asarray(target)*np.sqrt(sw)
    w=np.full(x.shape[1],1/x.shape[1],dtype=float)
    for _ in range(100):
        for j in range(len(w)):
            residual=y-design@w+design[:,j]*w[j]
            numerator=design[:,j]@residual
            denominator=design[:,j]@design[:,j]+.25
            w[j]=max(0.,numerator/denominator)
    assert np.isfinite(w).all() and w.sum()>0
    w/=w.sum()
    # The original scalar dot-product evaluation is retained for reproduction.
    return np.asarray([r@w for r in z]),w

def describe(v):
    x=np.asarray(v,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),
                minimum=float(x.min()),maximum=float(x.max()),favoring=int((x<0).sum()))

def evaluate(frame,scores,run_key,mask_basis):
    run_key=dict(**run_key,mask_basis=mask_basis)
    events=frame.event_id.astype(str).to_numpy()
    columns=COLUMNS if mask_basis=='stored_logits' else ['raw_'+c for c in COLUMNS]
    conf=frame[columns].to_numpy(dtype=np.int64)
    tl=losses(conf)
    points=[];integrals=[];retention=[]
    unique=np.unique(events)
    for arm,score in scores.items():
        event_curves=[];event_retention=[]
        for e in unique:
            ix=np.flatnonzero(events==e)
            ordered=ix[np.argsort(np.asarray(score)[ix],kind='stable')]
            cumulative_conf=conf[ordered].cumsum(axis=0)
            cumulative_loss=tl[ordered].cumsum(axis=0)
            n=len(ordered)
            k=np.clip(np.ceil(GRID*n).astype(int),1,n)
            risk=np.column_stack([cumulative_loss[k-1]/k[:,None],losses(cumulative_conf[k-1])])
            hazard_total=conf[ix,2:].sum()
            valid_total=conf[ix].sum()
            retained_hazard=cumulative_conf[k-1,2:].sum(axis=1)/hazard_total if hazard_total>0 else np.full(len(k),np.nan)
            retained_area=cumulative_conf[k-1].sum(axis=1)/valid_total
            event_curves.append(risk)
            event_retention.append(np.column_stack([k/n,retained_area,retained_hazard]))
            for j,endpoint in enumerate(ENDPOINTS):
                for c,r in zip(GRID,risk[:,j]):points.append(dict(**run_key,event_id=e,arm=arm,endpoint=endpoint,coverage=float(c),risk=float(r)))
        macro=np.mean(event_curves,axis=0)
        rt=np.nanmean(event_retention,axis=0)
        i80=int(np.argmin(np.abs(GRID-.8)))
        for j,endpoint in enumerate(ENDPOINTS):
            for c,r in zip(GRID,macro[:,j]):points.append(dict(**run_key,event_id='__event_macro__',arm=arm,endpoint=endpoint,coverage=float(c),risk=float(r)))
            window=(GRID>=.7-1e-10)&(GRID<=.9+1e-10)
            integrals.append(dict(**run_key,arm=arm,endpoint=endpoint,
                aurc=float(np.trapezoid(macro[:,j],GRID)),risk_080=float(macro[i80,j]),
                partial_aurc_070_090=float(np.trapezoid(macro[window,j],GRID[window])),
                mean_risk_070_090=float(np.trapezoid(macro[window,j],GRID[window])/.2)))
        retention.append(dict(**run_key,arm=arm,nominal_coverage=float(GRID[i80]),
            event_macro_tile_coverage=float(rt[i80,0]),event_macro_area_coverage=float(rt[i80,1]),
            event_macro_hazard_coverage=float(rt[i80,2])))
    return points,integrals,retention

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    files=sorted(REPLAY.glob('P[12]_F[123]_seed*.csv'))
    files=[p for p in files if p.with_suffix('.json').exists()]
    if not args.allow_partial:assert len(files)==18,f'Need all 18 runs, got {len(files)}'
    audits={split:pd.read_csv(SOURCE/f'{"calibration_" if split=="calibration" else ""}tile_audit_scores.csv',float_precision='round_trip') for split in ['calibration','test']}
    validation=[];ranks=[];curves=[];metrics=[];retentions=[];weights=[];score_tables=[];old_metrics=[]
    for file in files:
        meta=json.loads(file.with_suffix('.json').read_text())
        key={k:meta[k] for k in ['block','partition','seed']}
        run=ORIGINAL/key['block']/key['partition'].lower()/f'seed_{key["seed"]}'/'risk_v2'
        rep=pd.read_csv(file,float_precision='round_trip')
        frames={};checks=[]
        for split in ['calibration','test']:
            stored=read_tile_scores(run/f'predictions/{split}_tile_scores.parquet')
            ids=stored.sample_id.astype(str).tolist()
            r=rep[rep.split==split].sort_values('source_order')
            assert r.sample_id.astype(str).tolist()==ids,'Original sample order changed'
            af=audits[split]
            a=af[(af.block==key['block'])&(af.partition==key['partition'])&(af.seed==key['seed'])]
            assert set(a.sample_id)==set(ids)
            a=a.set_index('sample_id').loc[ids].reset_index()
            r=r.set_index('sample_id').loc[ids].reset_index()
            row=dict(**key,split=split,records=len(stored),
                score_max_error=float(np.max(np.abs(stored.soft_dice_risk_binary.to_numpy()-r.legacy_risk.to_numpy()))),
                entropy_max_error=float(np.max(np.abs(stored.uncertainty.to_numpy()-r.entropy.to_numpy()))),
                raw_confusion_mismatches=int(np.count_nonzero(a[COLUMNS].to_numpy()-r[['raw_'+c for c in COLUMNS]].to_numpy())),
                quantized_confusion_mismatches=int(np.count_nonzero(a[COLUMNS].to_numpy()-r[COLUMNS].to_numpy())),
                hard_mask_changed_tiles=int((r.hard_mask_disagreement>0).sum()),
                hard_mask_changed_pixels=int(r.hard_mask_disagreement.sum()),
                prediction_empty=int((r.predicted_foreground_input==0).sum()),zero_denominator=int((r.denominator_float64==0).sum()),
                exact_native_max_score_difference=float(np.max(np.abs(stored.soft_dice_risk_binary.to_numpy()-r.sdc_native.to_numpy()))),
                exact_float64_max_score_difference=float(np.max(np.abs(stored.soft_dice_risk_binary.to_numpy()-r.sdc_float64.to_numpy()))))
            assert row['score_max_error']<2e-5 and row['entropy_max_error']<2e-5,row
            assert row['raw_confusion_mismatches']==0,row
            checks.append(row)
            frame=stored.copy()
            for c in COLUMNS:
                frame[c]=r[c].to_numpy()
                frame['raw_'+c]=a[c].to_numpy()
            frame['sdc_exact']=r.sdc_float64.to_numpy();frame['sdc_native']=r.sdc_native.to_numpy()
            frame['audit_legacy_score']=a.soft_dice_risk_binary.to_numpy()
            frame['audit_v2_score']=a.georisk_v2.to_numpy()
            frames[split]=frame
            for e,g in frame.groupby('event_id',sort=True):
                old=g.soft_dice_risk_binary.to_numpy()
                oldorder=np.argsort(old,kind='stable')
                for arm in ['sdc_native','sdc_exact']:
                    new=g[arm].to_numpy();neworder=np.argsort(new,kind='stable')
                    k=int(np.ceil(GRID[np.argmin(np.abs(GRID-.8))]*len(g)))
                    changed=sum(set(oldorder[:k])!=set(neworder[:k]) for k in np.maximum(1,np.ceil(GRID*len(g)).astype(int)))
                    k=int(np.ceil(GRID[np.argmin(np.abs(GRID-.8))]*len(g)))
                    ranks.append(dict(**key,split=split,event_id=e,arm=arm,tiles=len(g),
                        order_changed=not np.array_equal(oldorder,neworder),positions_changed=int(np.count_nonzero(oldorder!=neworder)),
                        changed_retained_sets_across_grid=int(changed),
                        changed_members_at_080=int(len(set(oldorder[:k])^set(neworder[:k]))),
                        exact_risk_one_tiles=int(np.count_nonzero(new==1))))
        if args.validate_only:
            validation.extend(checks);continue
        cal=frames['calibration'];test=frames['test']
        events=cal.event_id.astype(str).to_numpy()
        cf=cal[FEATURES].to_numpy(dtype=float,copy=True);tf=test[FEATURES].to_numpy(dtype=float,copy=True)
        fitted,w=fit_score(cf,tf,events,cal.tile_loss.to_numpy())
        saved=json.loads((run/'calibration/calibration.json').read_text())['adaptive_fit']['weights']
        old_error=float(np.max(np.abs(fitted-test.georisk_v2.to_numpy())))
        weight_error=float(np.max(np.abs(w-np.asarray(saved))))
        assert old_error<1e-10 and weight_error<1e-10,dict(key=key,score=old_error,weights=weight_error)
        for check in checks:check.update(v2_refit_max_score_error=old_error,v2_refit_max_weight_error=weight_error)
        validation.extend(checks)
        scores={'legacy_self_overlap':test.soft_dice_risk_binary.to_numpy(),'frozen_v2':test.georisk_v2.to_numpy(),
                'sdc_native':test.sdc_native.to_numpy(),'sdc_exact':test.sdc_exact.to_numpy()}
        cf[:,0]=cal.sdc_exact;tf[:,0]=test.sdc_exact
        targets=losses(cal[COLUMNS].to_numpy())
        # All new targets use stored-logit hard masks, also used for the original target.
        # Verify that the mIoU target agrees with the saved original target.
        assert np.max(np.abs(targets[:,0]-cal.tile_loss.to_numpy()))<1e-12
        for j,target in enumerate(['miou','dice','iou']):
            score,w=fit_score(cf,tf,events,targets[:,j])
            arm='fit_'+target+'_exact';scores[arm]=score
            weights.append(dict(**key,arm=arm,overlap=w[0],entropy=w[1],shift=w[2],spatial=w[3]))
        s=test[['sample_id','event_id']].copy()
        for k,v in key.items():s[k]=v
        for k,v in scores.items():s[k]=v
        score_tables.append(s)
        for mask_basis in ['stored_logits','original_audit']:
            c,m,t=evaluate(test,scores,key,mask_basis);curves.extend(c);metrics.extend(m);retentions.extend(t)
        # Exact old-output gate retains the original CSV score precision as well.
        old_scores={'legacy_self_overlap':test.audit_legacy_score.to_numpy(),'frozen_v2':test.audit_v2_score.to_numpy()}
        _,m,_=evaluate(test,old_scores,key,'original_audit');old_metrics.extend(m)
        print('Validated and evaluated',key,flush=True)
    stem='partial_' if len(files)!=18 else ''
    pd.DataFrame(validation).to_csv(OUT/f'{stem}fidelity_by_run_split.csv',index=False)
    pd.DataFrame(ranks).to_csv(OUT/f'{stem}ordering_changes.csv',index=False)
    if args.validate_only:
        print(pd.DataFrame(validation).to_string(index=False));return
    metric=pd.DataFrame(metrics)
    pd.DataFrame(curves).to_csv(OUT/f'{stem}risk_curves.csv',index=False)
    metric.to_csv(OUT/f'{stem}metrics_by_run.csv',index=False)
    pd.DataFrame(retentions).to_csv(OUT/f'{stem}retention_by_run.csv',index=False)
    pd.DataFrame(weights).to_csv(OUT/f'{stem}fitted_weights.csv',index=False)
    pd.concat(score_tables,ignore_index=True).to_csv(OUT/f'{stem}test_scores.csv',index=False)
    index=['block','partition','seed','mask_basis','endpoint']
    exact=metric[metric.arm=='sdc_exact'].set_index(index)
    pairs=[]
    for arm in metric.arm.unique():
        if arm=='sdc_exact':continue
        f=metric[metric.arm==arm].set_index(index)
        for k,row in f.iterrows():
            base=exact.loc[k]
            pairs.append(dict(zip(index,k),arm=arm,**{'delta_'+m:float(row[m]-base[m]) for m in ['aurc','risk_080','partial_aurc_070_090','mean_risk_070_090']}))
    paired=pd.DataFrame(pairs);paired.to_csv(OUT/f'{stem}paired_effects_by_run.csv',index=False)
    sums=[]
    for (block,mask_basis,arm,endpoint),g in paired.groupby(['block','mask_basis','arm','endpoint']):
        for m in ['delta_aurc','delta_risk_080','delta_mean_risk_070_090']:
            sums.append(dict(block=block,mask_basis=mask_basis,arm=arm,endpoint=endpoint,metric=m,**describe(g[m])))
    summary=pd.DataFrame(sums);summary.to_csv(OUT/f'{stem}paired_summary.csv',index=False)
    # Independent agreement with frozen audit AURCs, including original coverage arithmetic.
    published=pd.read_csv(SOURCE/'endpoint_aurc_by_run.csv',float_precision='round_trip')
    checks=[]
    for r in pd.DataFrame(old_metrics).itertuples(index=False):
        emap={'pooled_miou':'present_class_miou_risk','pooled_dice':'foreground_dice_risk','pooled_iou':'foreground_iou_risk'}
        if r.endpoint not in emap:continue
        score='georisk_v2' if r.arm=='frozen_v2' else 'soft_dice_risk_binary'
        p=published[(published.block==r.block)&(published.partition==r.partition)&(published.seed==r.seed)&(published['subset']=='all_tiles')&(published.score==score)&(published.endpoint==emap[r.endpoint])]
        assert len(p)==1
        err=abs(r.aurc-float(p.event_macro_aurc.iloc[0]))
        checks.append(err)
    assert max(checks)<1e-10,max(checks)
    old_frame=pd.DataFrame(old_metrics)
    comparison=old_frame.merge(metric[metric.mask_basis=='original_audit'],on=['block','partition','seed','mask_basis','arm','endpoint'],suffixes=('_old_csv','_parquet'))
    comparison['delta_aurc_from_score_serialization']=comparison.aurc_parquet-comparison.aurc_old_csv
    comparison.to_csv(OUT/f'{stem}original_audit_reproduction.csv',index=False)
    validation_frame=pd.DataFrame(validation)
    report={'status':'complete primary comparator group' if len(files)==18 else 'partial',
        'runs':len(files),'records_by_split':validation_frame.groupby('split').records.sum().to_dict(),
        'max_legacy_score_replay_error':float(validation_frame.score_max_error.max()),
        'max_entropy_replay_error':float(validation_frame.entropy_max_error.max()),
        'max_exact_native_score_difference':float(validation_frame.exact_native_max_score_difference.max()),
        'max_exact_float64_score_difference':float(validation_frame.exact_float64_max_score_difference.max()),
        'raw_confusion_mismatches':int(validation_frame.raw_confusion_mismatches.sum()),
        'quantized_confusion_mismatches':int(validation_frame.quantized_confusion_mismatches.sum()),
        'hard_mask_changed_tiles':int(validation_frame.hard_mask_changed_tiles.sum()),
        'hard_mask_changed_pixels':int(validation_frame.hard_mask_changed_pixels.sum()),
        'max_original_aurc_reproduction_error':float(max(checks)),
        'max_published_endpoint_aurc_change_from_score_serialization':float(comparison[comparison.endpoint.str.startswith('pooled_')].delta_aurc_from_score_serialization.abs().max()),
        'max_all_six_endpoint_aurc_change_from_score_serialization':float(comparison.delta_aurc_from_score_serialization.abs().max()),
        'max_original_v2_refit_score_error':float(validation_frame.v2_refit_max_score_error.max()),
        'reference':'Borges et al. 2026 equation 22; https://link.springer.com/article/10.1007/s10994-026-07096-w',
        'scope':'18 primary frozen runs, calibration-only refits; no external multiclass reproduction or new independent validation'}
    (OUT/f'{stem}validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    chosen=summary[(summary.mask_basis=='stored_logits')&(summary.metric=='delta_aurc')&(summary.arm.isin(['frozen_v2','fit_miou_exact','fit_dice_exact','fit_iou_exact']))&(summary.endpoint.isin(['mean_tile_dice','mean_tile_miou','pooled_miou','pooled_dice']))]
    print(chosen.to_string(index=False))

if __name__=='__main__':main()
