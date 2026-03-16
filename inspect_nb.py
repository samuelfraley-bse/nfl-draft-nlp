import json
from pathlib import Path
path = Path('notebooks/grade_text_vs_grade.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type']=='code':
        print('CELL', i)
        print('\n'.join(cell['source']) if isinstance(cell['source'], list) else cell['source'])
        print('-----')
