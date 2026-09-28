"""Frozen paired-temperature inference. Original research inputs are read-only."""
from pathlib import Path
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
from georisk.risk import softmax,soft_dice_risk,uncertainty_score,msp_score,spatial_inconsistency_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/model_replay/temperature_replay'
DATA=None

def main():
    global DATA,OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--block',choices=['P1','P2'])
    parser.add_argument('--partition',choices=['F1','F2','F3'])
    parser.add_argument('--seed',type=int,choices=[13,37,71])
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    DATA=args.data_root.resolve()
    if args.output_dir is not None: OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    config=load_config(DATA/'configs/georisk_experiment_v1.yaml')
    for block in ([args.block] if args.block else ['P1','P2']):
        for partition in ([args.partition] if args.partition else ['F1','F2','F3']):
            for seed in ([args.seed] if args.seed else [13,37,71]):
                name=f'{block}_{partition}_seed{seed}'
                target=OUT/(name+'.csv');done=OUT/(name+'.json')
                if target.exists() and done.exists():print('Already completed',name,flush=True);continue
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
                    loader=DataLoader(datasets[split],batch_size=1,shuffle=False,num_workers=0)
                    with torch.inference_mode():
                        for i,batch in enumerate(loader,1):
                            with torch.amp.autocast(device_type='cuda',enabled=bool(config['environment']['mixed_precision'])):
                                logits,_=model(batch['image'].to('cuda'))
                            z=logits[0].float().cpu().numpy().astype(np.float16)
                            mask=batch['input_valid_mask'][0].numpy().astype(bool)
                            target_mask=batch['target'][0].numpy()
                            valid=target_mask!=255
                            assert mask.any() and valid.any()
                            pair=[];hards=[]
                            for label,t in [('calibrated',temperature),('t1',1.0)]:
                                p=softmax(z,t);hard=p.argmax(axis=0);hards.append(hard)
                                y=hard==1;fg=p[1]
                                numerator=2.0*fg[y&mask].sum(dtype=np.float64)
                                denominator=float((y&mask).sum())+fg[mask].sum(dtype=np.float64)
                                conf=np.bincount((target_mask[valid]*2+hard[valid]).astype(np.int64),minlength=4)
                                # Stable log-softmax avoids clipping-induced NLL artifacts.
                                scaled=z[:,valid].astype(np.float64)/t
                                maxima=scaled.max(axis=0)
                                lognorm=maxima+np.log(np.exp(scaled-maxima).sum(axis=0))
                                labels=target_mask[valid].astype(np.int64)
                                nll=lognorm-scaled[labels,np.arange(len(labels))]
                                brier=(fg[valid].astype(np.float64)-(labels==1))**2
                                row=dict(block=block,partition=partition,seed=seed,split=split,setting=label,
                                    source_order=i-1,sample_id=str(batch['sample_id'][0]),event_id=str(batch['event_id'][0]),temperature=t,
                                    soft_dice_risk_binary=soft_dice_risk(p,mask),sdc_exact=1.-float(numerator/denominator if denominator>0 else 0.),
                                    uncertainty=uncertainty_score(p,mask),top_tail_msp=msp_score(p,mask),
                                    spatial_inconsistency=spatial_inconsistency_score(p,mask),
                                    input_valid_pixels=int(mask.sum()),reference_valid_pixels=int(valid.sum()),
                                    predicted_foreground_input=int((y&mask).sum()),nll_sum=float(nll.sum()),brier_sum=float(brier.sum()))
                                for j,v in enumerate(conf):row[f'confusion_{j//2}_{j%2}']=int(v)
                                pair.append(row)
                            changed=hards[0]!=hards[1]
                            for row in pair:
                                row['hard_disagreement_all_pixels']=int(changed.sum())
                                row['hard_disagreement_reference_valid']=int(changed[valid].sum())
                                rows.append(row)
                            if i%75==0:print(f'{name} {split} {i}/{len(datasets[split])}; {time.time()-started:.1f}s',flush=True)
                with target.open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                report=dict(run=name,block=block,partition=partition,seed=seed,temperature=temperature,
                    calibration_records=len(datasets['calibration']),test_records=len(datasets['test']),rows=len(rows),
                    seconds=time.time()-started,status='paired temperature inference complete; validation required')
                done.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
                del model,datasets,loader,rows,logits,batch,p,z
                gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
