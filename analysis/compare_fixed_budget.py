"""Compare saved calibration thresholds with a fixed 20% batch review count."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

KEY = ['block', 'partition', 'seed']
POLICIES = ['calibration_threshold', 'fixed_review_budget']

def read(path):
    return pd.read_csv(path, float_precision='round_trip', dtype={'sample_id': str, 'event_id': str})

def fixed_mask(scores):
    n = len(scores)
    order = np.argsort(np.asarray(scores), kind='stable')
    keep = np.zeros(n, dtype=bool)
    keep[order[:n-n//5]] = True
    return keep

def risks(counts, full_classes):
    pooled = counts.sum(axis=0)
    union = pooled.sum(axis=0) + pooled.sum(axis=1) - np.diag(pooled)
    iou = np.divide(np.diag(pooled), union, out=np.zeros(len(union)), where=union > 0)
    diag = np.diagonal(counts, axis1=1, axis2=2)
    tile_union = counts.sum(axis=1) + counts.sum(axis=2) - diag
    tile_iou = np.divide(diag, tile_union, out=np.zeros_like(diag, dtype=float), where=tile_union > 0)
    tile_loss = 1 - tile_iou.sum(axis=1) / (tile_union > 0).sum(axis=1)
    return dict(tile_mean_risk=float(tile_loss.mean()), pooled_fixed_class_risk=float(1-iou[full_classes].mean()))

def ratio(num, den):
    return float(num/den) if den else float('nan')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local = Path(__file__).resolve().parents[1]
    default_data = local/'data' if (local/'data').exists() else local/'reproducibility_package/georisk-geomatica/data'
    parser.add_argument('--data-root', type=Path, default=default_data)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        import os
        out = Path(os.environ.get('GEORISK_OUTPUT_DIR', local))/'evidence/fixed_budget'
    else: out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = args.data_root
    used = {}
    def track(path):
        used[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    binary = read(track(data/'source/r5_endpoint_audit/tile_audit_scores.csv'))
    multi = read(track(data/'source/r6_multiclass_audit/e1_tile_audit_scores.csv'))
    legacy = read(track(data/'source/legacy_r4/threshold_coverage_runs.csv'))
    assert len(legacy) == 27 and not legacy.duplicated(KEY).any()
    # Boundary ties preserve original low-to-high order, and the budget is a cap.
    assert fixed_mask([0., 1., 1., 1., 1.]).tolist() == [True, True, True, True, False]
    for n in range(1, 203):
        assert int((~fixed_mask(np.zeros(n))).sum()) == n//5
        assert n//5 <= n/5 < n//5+1
    rows, event_rows, selection_rows = [], [], []
    max_score_diff = max_loss_error = 0.
    for r in legacy.itertuples(index=False):
        key = dict(block=r.block, partition=r.partition, seed=int(r.seed))
        runpath = data/'runs'/r.block/r.partition.lower()/f'seed_{r.seed}'/'risk_v2'
        original = read(track(runpath/'predictions/test_tile_scores.csv'))
        config = json.loads(track(runpath/'calibration/calibration.json').read_text(encoding='utf-8'))
        threshold = float(config['thresholds']['georisk_v2'])
        source = multi if r.block == 'E1' else binary
        audit = source[(source.block == r.block)&(source.partition == r.partition)&(source.seed == r.seed)]
        assert not original.sample_id.duplicated().any()
        has_counts = r.block in ['P1','P2','E1']
        if has_counts:
            assert len(audit) == len(original) and not audit.sample_id.duplicated().any()
            audit = audit.set_index('sample_id').loc[original.sample_id].reset_index()
            assert audit.sample_id.tolist() == original.sample_id.tolist()
            assert audit.event_id.tolist() == original.event_id.tolist()
            diff = float(np.max(abs(audit.georisk_v2.to_numpy()-original.georisk_v2.to_numpy())))
            max_score_diff = max(max_score_diff, diff)
            assert diff < 1e-14
            assert np.max(abs(audit.tile_loss.to_numpy()-original.tile_loss.to_numpy())) < 1e-14
        else: assert len(audit)==0
        scores = original.georisk_v2.to_numpy(dtype=float)
        assert np.isfinite(scores).all()
        c = 3 if r.block == 'E1' else 2
        cols = [f'confusion_{a}_{b}' for a in range(c) for b in range(c)]
        if has_counts:
            counts = audit[cols].to_numpy(dtype=np.int64).reshape(-1,c,c)
            assert (counts >= 0).all()
            assert np.array_equal(counts.sum(axis=(1,2)), audit.valid_pixels.to_numpy())
            full = counts.sum(axis=0)
            classes = (full.sum(axis=0)+full.sum(axis=1)-np.diag(full)) > 0
            full_metrics = risks(counts, classes)
            tile_diag = np.diagonal(counts, axis1=1, axis2=2)
            union = counts.sum(axis=1)+counts.sum(axis=2)-tile_diag
            tiou = np.divide(tile_diag,union,out=np.zeros_like(tile_diag,dtype=float),where=union>0)
            losses = 1-tiou.sum(axis=1)/(union>0).sum(axis=1)
            error = float(np.max(abs(losses-original.tile_loss.to_numpy())))
            max_loss_error = max(max_loss_error,error)
        else:
            error = float('nan')
            full_metrics = {'pooled_fixed_class_risk':float('nan')}
        threshold_mask = scores <= threshold
        budget_mask = fixed_mask(scores)
        n = len(scores); b = n//5
        assert int((~budget_mask).sum()) == b
        assert scores[budget_mask].max() <= scores[~budget_mask].min()
        assert (np.all(~threshold_mask|budget_mask) or np.all(~budget_mask|threshold_mask))
        observed = float(pd.DataFrame({'event':original.event_id,'keep':threshold_mask}).groupby('event').keep.mean().mean())
        assert abs(observed-r.realized_coverage) < 1e-12
        hazard = 2 if r.block == 'E1' else 1
        if has_counts: ref = counts.sum(axis=2)
        ordered = np.argsort(scores,kind='stable')
        boundary = float(scores[ordered[n-b-1]])
        ties = int((scores==boundary).sum())
        tie_split = bool(np.any((scores==boundary)&budget_mask) and np.any((scores==boundary)&~budget_mask))
        for policy, keep in zip(POLICIES,[threshold_mask,budget_mask]):
            metrics = risks(counts[keep],classes) if has_counts else {'tile_mean_risk':float('nan'),'pooled_fixed_class_risk':float('nan')}
            row = dict(**key, policy=policy, tiles=n, review_budget_tiles=b,
                retained_tiles=int(keep.sum()), review_tiles=int((~keep).sum()),
                review_fraction=float((~keep).mean()), tile_coverage=float(keep.mean()),
                valid_pixel_coverage=ratio(int(counts[keep].sum()),int(counts.sum())) if has_counts else float('nan'),
                hazard_label='flood' if r.block=='E1' else ('water' if r.block.startswith('P') else 'burn_scar'),
                hazard_content_coverage=ratio(int(ref[keep,hazard].sum()),int(ref[:,hazard].sum())) if has_counts else float('nan'),
                permanent_water_coverage=ratio(int(ref[keep,1].sum()),int(ref[:,1].sum())) if r.block=='E1' else float('nan'),
                hazard_pixels_total=int(ref[:,hazard].sum()) if has_counts else float('nan'),
                hazard_pixels_retained=int(ref[keep,hazard].sum()) if has_counts else float('nan'),
                threshold=threshold, fixed_budget_boundary_score=boundary,
                boundary_tie_size=ties, boundary_tie_split=tie_split,
                tile_mean_risk=float(original.tile_loss.to_numpy()[keep].mean()),
                pooled_fixed_class_risk=metrics['pooled_fixed_class_risk'],
                audit_tile_mean_risk=metrics['tile_mean_risk'],audit_confusions_available=has_counts,
                max_audit_vs_stored_tile_loss_difference=error,
                full_batch_tile_mean_risk=float(original.tile_loss.mean()),
                full_batch_pooled_fixed_class_risk=full_metrics['pooled_fixed_class_risk'])
            rows.append(row)
            for event in original.event_id.unique():
                event_mask=original.event_id.to_numpy()==event
                event_rows.append(dict(**key,policy=policy,event_id=event,tiles=int(event_mask.sum()),
                    retained_tiles=int((event_mask&keep).sum()),review_tiles=int((event_mask&~keep).sum())))
            for i in range(n):
                selection_rows.append(dict(**key,policy=policy,input_position=i,sample_id=original.sample_id.iloc[i],
                    event_id=original.event_id.iloc[i],score=float(scores[i]),retained=bool(keep[i])))
    frame=pd.DataFrame(rows)
    assert len(frame)==54
    before=frame[frame.policy==POLICIES[0]].set_index(KEY)
    after=frame[frame.policy==POLICIES[1]].set_index(KEY)
    columns=['review_tiles','review_fraction','tile_coverage','tile_mean_risk','pooled_fixed_class_risk',
        'valid_pixel_coverage','hazard_content_coverage','permanent_water_coverage','audit_tile_mean_risk']
    paired=before[['tiles','review_budget_tiles','hazard_label']].copy()
    for col in columns:
        paired['threshold_'+col]=before[col]
        paired['fixed_'+col]=after[col]
        paired['delta_'+col]=after[col]-before[col]
    paired=paired.reset_index()
    summary=[]
    for block,g in paired.groupby('block',sort=True):
        row=dict(block=block,runs=len(g),threshold_over_budget_runs=int((g.threshold_review_tiles>g.review_budget_tiles).sum()),
            threshold_under_budget_runs=int((g.threshold_review_tiles<g.review_budget_tiles).sum()))
        for col in columns:
            for policy in ['threshold','fixed','delta']:
                vals=g[policy+'_'+col]
                for stat in ['mean','min','max']:row[f'{policy}_{col}_{stat}']=float(getattr(vals,stat)())
        for col in ['tile_mean_risk','pooled_fixed_class_risk','hazard_content_coverage']:
            vals=g['delta_'+col]
            row['delta_'+col+'_positive_runs']=int((vals>1e-12).sum())
            row['delta_'+col+'_negative_runs']=int((vals<-1e-12).sum())
        summary.append(row)
    outputs={'policy_by_run.csv':frame,'paired_by_run.csv':paired,'summary_by_block.csv':pd.DataFrame(summary),
             'policy_by_event.csv':pd.DataFrame(event_rows),'tile_selections.csv':pd.DataFrame(selection_rows)}
    for name,df in outputs.items():df.to_csv(out/name,index=False)
    assert used=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in used}
    checks=dict(runs=27,policies=2,prediction_records=len(selection_rows)//2,inputs_unchanged=True,
        original_score_order_preserved=True,score_source='Original per-run frozen scores; joined confusion counts are evaluation only.',
        max_audit_vs_original_score_difference=max_score_diff,max_audit_vs_stored_tile_loss_difference=max_loss_error,
        original_tile_mean_loss_used=True,pooled_risk_and_content_runs=21,
        unavailable_content_blocks=['E2','E3'],
        threshold_event_macro_reconstruction_passed=True,all_budgets_exact=True,nested_retained_sets=True,
        boundary_tie_split_runs=int(frame[frame.policy==POLICIES[1]].boundary_tie_split.sum()),
        integer_rounding_and_tie_fixture_passed=True,batch_scope='Complete test batch within one run',
        confidence_intervals=False,human_review_outcomes_measured=False,
        hashes={str(p.relative_to(data)).replace('\\','/'):v for p,v in used.items()})
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    compact=pd.DataFrame(summary)[['block','runs','threshold_over_budget_runs','threshold_under_budget_runs',
        'threshold_review_fraction_min','threshold_review_fraction_max','fixed_review_fraction_min','fixed_review_fraction_max',
        'delta_tile_mean_risk_mean','delta_pooled_fixed_class_risk_mean','delta_hazard_content_coverage_mean']]
    print(compact.to_string(index=False))
    print(json.dumps({k:v for k,v in checks.items() if k!='hashes'},indent=2))

if __name__=='__main__':main()
