"""Recompute the frozen-data analyses and main figures in a fresh output directory."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parent
STEPS=[
    ('binary_endpoint_audit','analyze_r5_endpoint_validity.py'),
    ('multiclass_endpoint_audit','analyze_r6_e1_endpoint_audit.py'),
    ('multiclass_scope','audit_multiclass_scope.py'),
    ('operating_points_external_effects','summarize_revision_evidence.py'),
    ('threshold_workload','audit_threshold_denominators.py'),
    ('fixed_batch_review_budget','compare_fixed_budget.py'),
    ('exact_sdc_matched_targets','compare_primary_targets.py'),
    ('comparator_summary_tables','summarize_comparator_results.py'),
    ('temperature_ablation','analyze_temperature_ablation.py'),
    ('temperature_summary_tables','summarize_temperature_results.py'),
    ('main_figures','polish_main_figures.py'),
    ('manuscript_table_exports','export_manuscript_tables.py'),
    ('validate_outputs','validate_outputs.py'),
]

def hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PACKAGE/'outputs')
    args=parser.parse_args();out=args.output.resolve()
    if out==PACKAGE or any(out==PACKAGE/x or (PACKAGE/x) in out.parents for x in ['data','expected','analysis','docs','model_code','model_materials','model_replay']):
        parser.error('Choose a separate output directory; input and source folders are protected.')
    if out.exists() and any(out.iterdir()):parser.error('Output directory must be empty. Choose a new path; existing files will not be deleted.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'evidence').mkdir();(out/'logs').mkdir();(out/'qa').mkdir()
    before=hashes(PACKAGE/'data')
    shutil.copytree(PACKAGE/'data/source',out/'source')
    for p in (PACKAGE/'data/replay').iterdir():shutil.copytree(p,out/'evidence'/p.name)
    env=os.environ.copy()
    env.update({'GEORISK_OUTPUT_DIR':str(out),'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1',
        'MPLBACKEND':'Agg','MPLCONFIGDIR':str(out/'.mplconfig'),
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    report={'status':'running','tier':'Frozen scores and replay sufficient statistics',
        'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','matplotlib','Pillow','pypdf']},'steps':[]}
    started=time.monotonic()
    for name,script in STEPS:
        print(f'Running {name}...',flush=True);step_time=time.monotonic()
        with (out/'logs'/f'{name}.log').open('w',encoding='utf-8') as log:
            process=subprocess.run([sys.executable,'-X','utf8',str(PACKAGE/'analysis'/script)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        report['steps'].append({'name':name,'returncode':process.returncode,'seconds':round(time.monotonic()-step_time,3)})
        if process.returncode:
            report['status']='failed';report['failed_step']=name
            (out/'reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print((out/'logs'/f'{name}.log').read_text(encoding='utf-8')[-8000:],file=sys.stderr)
            raise SystemExit(process.returncode)
    assert before==hashes(PACKAGE/'data'),'Input files changed during reproduction'
    report.update(status='passed',seconds=round(time.monotonic()-started,3),input_files_unchanged=True,
        inference_rerun=False,training_rerun=False)
    (out/'reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'PASS: analyses, figures and validation written to {out}',flush=True)

if __name__=='__main__':main()
