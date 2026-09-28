"""Paired class-set audit from frozen tile confusions; no prediction or score fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

GRID=np.arange(.10,1.001,.05)
SCORES=['georisk_v2','self_dice_risk_multiclass']
PRESENT='present_class_miou_risk'
FIXED='fixed_full_event_class_miou_risk'
ATOL=1e-12

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.source is None or args.output is None:
        from package_paths import SOURCE_ROOT,OUTPUT_ROOT
    source=args.source if args.source is not None else SOURCE_ROOT/'r6_multiclass_audit'
    out=args.output if args.output is not None else OUTPUT_ROOT/'evidence/multiclass_scope'
    out.mkdir(parents=True,exist_ok=True)
    inputs=[source/'e1_tile_audit_scores.csv',source/'e1_endpoint_aurc.csv',source/'e1_endpoint_curves.csv']
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    frame=pd.read_csv(inputs[0],float_precision='round_trip',dtype={'sample_id':str,'event_id':str})
    reference=pd.read_csv(inputs[1],float_precision='round_trip')
    columns=[f'confusion_{a}_{b}' for a in range(3) for b in range(3)]
    assert len(frame)==6000 and set(frame.seed)=={13,37,71}
    assert not frame.duplicated(['seed','sample_id']).any()
    assert (frame[columns].sum(axis=1)==frame.valid_pixels).all()
    rows=[];events=[];curves=[]
    for seed,run in frame.groupby('seed',sort=True):
        assert len(run)==2000 and run.event_id.nunique()==10
        for score in SCORES:
            for event_id,event in run.groupby('event_id',sort=True):
                counts=event[columns].to_numpy(dtype=np.int64)
                full=counts.sum(axis=0).reshape(3,3)
                full_classes=(full.sum(axis=0)+full.sum(axis=1)-np.diag(full))>0
                order=np.argsort(event[score].to_numpy(),kind='stable')
                cumulative=counts[order].cumsum(axis=0)
                pr=[];fx=[]
                for coverage in GRID:
                    k=min(len(event),max(1,int(np.ceil(coverage*len(event)))))
                    c=cumulative[k-1].reshape(3,3)
                    union=c.sum(axis=0)+c.sum(axis=1)-np.diag(c);present=union>0
                    iou=np.divide(np.diag(c),union,out=np.zeros(3,dtype=float),where=present)
                    a=1-float(iou[present].mean());b=1-float(iou[full_classes].mean())
                    assert b>=a-ATOL
                    pr.append(a);fx.append(b)
                    curves.append(dict(seed=int(seed),score=score,event_id=event_id,coverage=float(coverage),
                        retained_tiles=k,total_tiles=len(event),present_risk=a,fixed_risk=b,class_set_shift=b-a,
                        permanent_water_omitted=int(full_classes[1] and not present[1]),
                        flood_omitted=int(full_classes[2] and not present[2])))
                events.append(dict(seed=int(seed),score=score,event_id=event_id,
                    present_aurc=float(np.trapezoid(pr,GRID)),fixed_aurc=float(np.trapezoid(fx,GRID)),
                    class_set_shift=float(np.trapezoid(np.array(fx)-pr,GRID))))
    curves=pd.DataFrame(curves);events=pd.DataFrame(events)
    macro=curves.groupby(['seed','score','coverage'],as_index=False)[['present_risk','fixed_risk','class_set_shift']].mean()
    reconstruction_errors=[]
    for (seed,score),g in macro.groupby(['seed','score'],sort=True):
        pa=float(np.trapezoid(g.present_risk,g.coverage));fa=float(np.trapezoid(g.fixed_risk,g.coverage));delta=fa-pa
        low=g[g.coverage<=.5+ATOL]
        at80=g.loc[np.isclose(g.coverage,.8,rtol=0,atol=ATOL)].iloc[0]
        for endpoint,value in [(PRESENT,pa),(FIXED,fa)]:
            ref=reference[(reference.seed==seed)&(reference.score==score)&(reference.endpoint==endpoint)]
            assert len(ref)==1
            error=abs(float(ref.event_macro_aurc.iloc[0])-value);reconstruction_errors.append(error)
            assert error<ATOL
        rows.append(dict(seed=int(seed),score=score,test_tiles=2000,test_events=10,present_aurc=pa,fixed_aurc=fa,
            class_set_shift=delta,relative_shift_percent=100*delta/pa,
            fraction_shift_010_050=float(np.trapezoid(low.class_set_shift,low.coverage)/delta),
            class_set_shift_at_080=float(at80.class_set_shift)))
    by_seed=pd.DataFrame(rows)
    original=pd.read_csv(inputs[2],float_precision='round_trip',dtype={'event_id':str})
    for name,key in [(PRESENT,'present_risk'),(FIXED,'fixed_risk')]:
        a=original[(original.score.isin(SCORES))&(original.event_id!='__event_macro__')&(original.endpoint==name)]
        merged=a.merge(curves,on=['seed','score','event_id','coverage'],validate='one_to_one')
        assert len(merged)==1140
        assert np.max(abs(merged.risk-merged[key]))<ATOL
    mean=macro.groupby('coverage',as_index=False)[['present_risk','fixed_risk','class_set_shift']].mean()
    total=float(np.trapezoid(mean.class_set_shift,mean.coverage))
    legacy=mean[mean.coverage<=.5];corrected=mean[mean.coverage<=.5+ATOL]
    at80=curves[np.isclose(curves.coverage,.8,rtol=0,atol=ATOL)]
    summary=dict(scope='One dataset, one U-Net architecture, three seeds, same 2000 test tiles and 10 test events.',
        test_records=len(frame),unique_test_tiles=frame.sample_id.nunique(),seeds=[13,37,71],
        seed_ordering_pairs=len(by_seed),all_pair_shifts_positive=bool((by_seed.class_set_shift>0).all()),
        mean_class_set_shift=total,relative_shift_of_means_percent=100*total/float(by_seed.present_aurc.mean()),
        relative_shift_pair_min_percent=float(by_seed.relative_shift_percent.min()),relative_shift_pair_max_percent=float(by_seed.relative_shift_percent.max()),
        shift_pair_min=float(by_seed.class_set_shift.min()),shift_pair_max=float(by_seed.class_set_shift.max()),
        corrected_fraction_shift_010_050=float(np.trapezoid(corrected.class_set_shift,corrected.coverage)/total),
        legacy_fraction_shift_010_045=float(np.trapezoid(legacy.class_set_shift,legacy.coverage)/total),
        mean_shift_at_080=float(at80.class_set_shift.mean()),
        permanent_water_omission_at_080=float(at80.permanent_water_omitted.mean()),
        flood_omission_at_080=float(at80.flood_omitted.mean()),
        max_aurc_reconstruction_error=max(reconstruction_errors),
        range_interpretation='Observed variation across dependent seed-ordering pairs, not a confidence interval.',
        prediction_scores_and_selection_changed=False,
        interval_fix='Include the nominal 0.50 grid endpoint with absolute tolerance 1e-12; original grid and selections unchanged.')
    by_seed.to_csv(out/'class_set_shift_by_seed.csv',index=False)
    events.to_csv(out/'class_set_shift_by_event_seed.csv',index=False)
    macro.to_csv(out/'class_set_macro_curves.csv',index=False)
    (out/'scope_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (out/'input_hashes.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
