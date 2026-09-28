"""Export reviewed table cells; numerical recomputation lives in the analysis CSVs."""
import csv
import json
from package_paths import PACKAGE_ROOT, OUTPUT_ROOT

tables=json.loads((PACKAGE_ROOT/'data/table_layout_data.json').read_text(encoding='utf-8'))
out=OUTPUT_ROOT/'tables';out.mkdir(exist_ok=True)
index=[]
for table in tables:
    number=('S' if table['document']=='Supplementary' else '')+str(table['number'])
    with (out/f'Table_{number}.csv').open('w',newline='',encoding='utf-8') as stream:
        csv.writer(stream).writerows(table['cells'])
    index.append({'table':number,'caption':table['caption'],'notes':table['notes'],
        'origin':'Reviewed manuscript table-cell snapshot; export does not itself recompute a statistic.'})
(out/'table_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Exported {len(tables)} reviewed table snapshots. See docs/OUTPUT_MAP.md for recomputed sources.')
