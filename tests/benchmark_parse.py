import json
from pathlib import Path
import pdb
import pandas as pd
db = pdb.set_trace


def benchmarks(file=None):
    if file is None:
        path = Path('.benchmarks')
        for child in path.iterdir():
            sub = child

        list_of_paths = sub.glob('*')
        file = max(list_of_paths, key=lambda p: p.stat().st_ctime)
        print(f'using file: {file}')

    with open(file, 'r') as input:
        data = json.load(input)

    df = pd.DataFrame()
    for bm in data['benchmarks']:
        temp = pd.DataFrame(index=[bm['name']], data=bm['stats'])
        df = pd.concat([df, temp])

    return df


def compare(old, new):
    comp = pd.merge(old.reset_index(), new.reset_index(), on='index', how='left')
    comp['% reduction'] = 1 - comp['mean_y']/comp['mean_x']
    return comp[['index', 'mean_x', 'mean_y', '% reduction']].set_index('index')
