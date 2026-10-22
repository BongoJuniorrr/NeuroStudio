import json,shutil,zipfile
from pathlib import Path
from .util import atomic_json,file_hash


def export_exploration(run,destination):
    from .engine import exploration
    source=Path(run);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','result.json','pipeline.joblib','clusters.csv']:shutil.copy2(source/name,dest/name)
    shutil.copy2(exploration.__file__,dest/'runtime.py')
    (dest/'inference.py').write_text('''import json,argparse
from pathlib import Path
from runtime import predict_clusters
p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args()
print(json.dumps(predict_clusters(Path(__file__).resolve().parent,json.loads(Path(a.input).read_text(encoding='utf-8')))))
''', encoding='utf-8')
    (dest/'train.py').write_text('''import argparse,json
from pathlib import Path
from runtime import run_exploration
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,help='Dataset descriptor JSON with kind/path');a=p.parse_args()
here=Path(__file__).resolve().parent
run_exploration(json.loads((here/'config.json').read_text(encoding='utf-8')),json.loads(Path(a.dataset).read_text(encoding='utf-8')),here/'retrained')
''', encoding='utf-8')
    import importlib.metadata
    packages=['numpy','pandas','scikit-learn','joblib']+(['umap-learn'] if json.loads((source/'config.json').read_text(encoding='utf-8'))['projection']=='umap' else [])
    (dest/'requirements.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in packages)+'\n', encoding='utf-8')
    (dest/'README.md').write_text('# Clustering and projections\n\nInstall requirements.txt in Python 3.12. View clusters.csv/result.json. KMeans supports `python inference.py --input rows.json`. DBSCAN/agglomerative labels apply to fitted rows only. Projection coordinates are exploratory, not classification accuracy; t-SNE has no out-of-sample transform. Raw data excluded. Load only your own trusted joblib artifacts.\n', encoding='utf-8')
    atomic_json(dest/'manifest.json',dict(files={p.name:file_hash(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'}))
    archive=dest.with_suffix('.zip');pending=archive.with_suffix('.zip.pending')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for p in dest.iterdir():
            if p.is_file():z.write(p,p.name)
    pending.replace(archive);return archive
