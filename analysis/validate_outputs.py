"""Compare recomputed CSVs with frozen reference results and check figure metadata."""
import hashlib
import json
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image
from pypdf import PdfReader,PdfWriter
from package_paths import PACKAGE_ROOT,OUTPUT_ROOT

# These earlier summary variants are not inputs or targets of this pipeline.
EXCLUDED={'source/r5_endpoint_audit/coverage_gain_decomposition.csv',
          'source/r5_endpoint_audit/external_comparator_summary.csv'}
ATOL=1e-11;RTOL=1e-9

def resolve(obj):return obj.get_object() if hasattr(obj,'get_object') else obj

def embedded(obj):
    font=resolve(obj);desc=resolve(font.get('/FontDescriptor'))
    if desc and any(k in desc and len(resolve(desc[k]).get_data())>0 for k in ['/FontFile','/FontFile2','/FontFile3']):return True
    children=resolve(font.get('/DescendantFonts',[]))
    return bool(children) and all(embedded(c) for c in children)

def main():
    checks=[];failures=[]
    for expected in sorted((PACKAGE_ROOT/'expected').rglob('*.csv')):
        relative=expected.relative_to(PACKAGE_ROOT/'expected').as_posix()
        if relative in EXCLUDED:continue
        actual=OUTPUT_ROOT/relative
        if not actual.exists():failures.append({'path':relative,'reason':'missing output'});continue
        ref=pd.read_csv(expected,float_precision='round_trip');new=pd.read_csv(actual,float_precision='round_trip')
        try:
            assert_frame_equal(new,ref,check_dtype=False,check_exact=False,rtol=RTOL,atol=ATOL)
            for column in ref:
                if pd.api.types.is_integer_dtype(ref[column]) or pd.api.types.is_bool_dtype(ref[column]):
                    assert np.array_equal(new[column].to_numpy(),ref[column].to_numpy()),column
            errors=[]
            for column in ref.select_dtypes(include='number'):
                delta=np.abs(new[column].to_numpy(dtype=float)-ref[column].to_numpy(dtype=float))
                errors.extend(delta[np.isfinite(delta)].tolist())
            checks.append({'path':relative,'rows':len(ref),'max_absolute_numeric_difference':max(errors,default=0.0)})
        except (AssertionError,ValueError) as exc:failures.append({'path':relative,'reason':str(exc)[:2500]})
    figure_checks=[];merged=PdfWriter()
    for n in range(1,6):
        stem=f'Figure_{n}_MR';reader=PdfReader(OUTPUT_ROOT/'figures'/f'{stem}.pdf')
        assert len(reader.pages)==1
        page=reader.pages[0];fonts=resolve(resolve(page.get('/Resources')).get('/Font',{}))
        assert fonts and all(embedded(f) for f in fonts.values()),stem
        with Image.open(OUTPUT_ROOT/'figures'/f'{stem}.tiff') as im:
            assert im.mode=='RGB' and all(abs(float(x)-600)<.01 for x in im.info['dpi'])
            raster_pixels=list(im.size)
        layout=json.loads((OUTPUT_ROOT/'qa/figure_layout'/f'{stem}_layout.json').read_text())
        assert not layout['text_outside_canvas'] and not layout['tick_label_overlaps'] and not layout['missing_glyph_warnings']
        figure_checks.append({'figure':n,'pdf_fonts_embedded':True,'raster_pixels':raster_pixels,'base_text_at_6p5_in_pt':layout['manuscript_font_min_pt']})
        merged.add_page(page)
    with (OUTPUT_ROOT/'figures/Main_Figures_MR.pdf').open('wb') as f:merged.write(f)
    report={'status':'failed' if failures else 'passed','absolute_tolerance':ATOL,'relative_tolerance':RTOL,
        'integer_and_boolean_columns':'exact equality required','csv_checks':checks,'failures':failures,
        'excluded_prior_summary_variants':sorted(EXCLUDED),'figures':figure_checks,
        'table_snapshots_exported':len(list((OUTPUT_ROOT/'tables').glob('Table_*.csv'))),
        'scope':'Frozen-data calculation reproduction, not independent data validation or model retraining.'}
    (OUTPUT_ROOT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'csvs_passed':len(checks),'csvs_failed':len(failures),'figures':len(figure_checks)},indent=2))
    if failures:
        print(json.dumps(failures,indent=2));raise SystemExit(1)

if __name__=='__main__':main()
