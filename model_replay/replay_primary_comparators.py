"""Replay frozen primary probabilities; no model fitting and no original writes."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_code'))
import argparse
import csv
import gc
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from georisk.config import load_config,run_directory
from georisk.data import DATASET_INFO,build_datasets
from georisk.models import build_model
from georisk.risk import softmax,soft_dice_risk,uncertainty_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/model_replay/comparator_replay'
DATA=None

def main():
    global DATA,OUT
    args=argparse.ArgumentParser()
    args.add_argument('--block',choices=['P1','P2'])
    args.add_argument('--partition',choices=['F1','F2','F3'])
    args.add_argument('--seed',type=int,choices=[13,37,71])
    args.add_argument('--data-root',type=Path,required=True)
    args.add_argument('--output-dir',type=Path)
    a=args.parse_args()
    DATA=a.data_root.resolve()
    if a.output_dir is not None: OUT=a.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    config=load_config(DATA/'configs/georisk_experiment_v1.yaml')
    for block in ([a.block] if a.block else ['P1','P2']):
        for partition in ([a.partition] if a.partition else ['F1','F2','F3']):
            for seed in ([a.seed] if a.seed else [13,37,71]):
                name=f'{block}_{partition}_seed{seed}'
                target=OUT/(name+'.csv');done=OUT/(name+'.json')
                if target.exists() and done.exists():
                    print('Already completed',name,flush=True);continue
                started=time.time();run=run_directory(DATA,block,partition,seed)
                cal=json.loads((run/'risk_v2/calibration/calibration.json').read_text(encoding='utf-8'))
                temperature=float(cal['temperature'])
                datasets=build_datasets(DATA,'sen1floods11',partition,seed,config['training']['internal_validation_fraction_per_train_event'],config['training']['crop_size'][0])
                model_name=config['blocks'][block]['model']
                model=build_model(model_name,DATASET_INFO['sen1floods11'],DATA,config['models'][model_name]).to('cuda')
                checkpoint=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(checkpoint['model']);del checkpoint
                model.eval();rows=[]
                for split in ['calibration','test']:
                    with torch.inference_mode():
                        loader=DataLoader(datasets[split],batch_size=1,shuffle=False,num_workers=0)
                        for i,batch in enumerate(loader,1):
                            with torch.amp.autocast(device_type='cuda',enabled=bool(config['environment']['mixed_precision'])):
                                logits,_=model(batch['image'].to('cuda'))
                            z=logits[0].float().cpu().numpy().astype(np.float16)
                            mask=batch['input_valid_mask'][0].numpy().astype(bool)
                            assert mask.any()
                            p=softmax(z,temperature)
                            hard=p.argmax(axis=0)
                            y=hard==1;fg=p[1]
                            a_native=2.0*fg[y&mask].sum()
                            b_native=(y&mask).sum()+fg[mask].sum()
                            a64=2.0*fg[y&mask].sum(dtype=np.float64)
                            b64=float((y&mask).sum())+fg[mask].sum(dtype=np.float64)
                            target_mask=batch['target'][0].numpy()
                            valid=target_mask!=255
                            conf=np.bincount((target_mask[valid]*2+hard[valid]).astype(np.int64),minlength=4)
                            # Original confusion replay used the unquantized hard logits.
                            raw_hard=logits[0].argmax(dim=0).cpu().numpy()
                            rawconf=np.bincount((target_mask[valid]*2+raw_hard[valid]).astype(np.int64),minlength=4)
                            row=dict(block=block,partition=partition,seed=seed,split=split,source_order=i-1,
                                     sample_id=str(batch['sample_id'][0]),event_id=str(batch['event_id'][0]),temperature=temperature,
                                     legacy_risk=soft_dice_risk(p,mask),entropy=uncertainty_score(p,mask),
                                     sdc_native=1.-float(a_native/b_native if b_native>0 else 0.),
                                     sdc_float64=1.-float(a64/b64 if b64>0 else 0.),
                                     numerator_float64=float(a64),denominator_float64=float(b64),
                                     input_valid_pixels=int(mask.sum()),predicted_foreground_input=int((y&mask).sum()),
                                     hard_mask_disagreement=int(np.count_nonzero(raw_hard!=hard)))
                            for j,(old,new) in enumerate(zip(conf,rawconf)):
                                row[f'confusion_{j//2}_{j%2}']=int(old)
                                row[f'raw_confusion_{j//2}_{j%2}']=int(new)
                            rows.append(row)
                            if i%75==0:print(f'{name} {split} {i}/{len(datasets[split])}; {time.time()-started:.1f}s',flush=True)
                with target.open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                report=dict(run=name,block=block,partition=partition,seed=seed,temperature=temperature,
                            calibration_records=len(datasets['calibration']),test_records=len(datasets['test']),
                            total_records=len(rows),seconds=time.time()-started,
                            status='inference complete; fidelity validation required',
                            source_checkpoint=str(run/'checkpoints/best.pt'))
                done.write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps(report),flush=True)
                del model,datasets,loader,rows,logits,batch,p,z
                gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
