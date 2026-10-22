"""Server-owned catalog and bounded task bundle ingestion."""
import json
import os
from pathlib import Path
import zipfile
from .util import file_hash,read_json


def benchmark_root(workspace):
    settings=read_json(Path(workspace)/'benchmark-settings.json',{})
    return Path(os.environ.get('NEUROSTUDIO_BENCHMARKS') or settings.get('directory') or Path(workspace)/'benchmarks').resolve()


def catalog(store):
    root=benchmark_root(store.root);result=[]
    if root.is_dir():
        for path in sorted(root.glob('*/*.json')):
            if path.stat().st_size>100*1024**2:continue
            try:data=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError):continue
            if data.get('format')!='neurostudio.task-dataset.v1':continue
            id='benchmark-'+file_hash(path)[:24]
            result.append(dict(id=id,kind='task_bundle',name=data.get('name',path.parent.name),task=data['task'],path=str(path.resolve()),hash=file_hash(path),counts={key:len(value) for key,value in data.get('splits',{}).items()},labels=data.get('labels',[]),source=data.get('source')))
    with store.connect() as db:
        for row in db.execute('SELECT body FROM datasets').fetchall():
            data=json.loads(row[0])
            if data.get('kind')=='task_bundle':result.append(data)
    return result


def extract_bundle(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        total=0
        if len(z.infolist())>100000:raise ValueError('DATASET_FILES: Quá nhiều file.')
        for item in z.infolist():
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination) or '\\' in item.filename or item.external_attr>>16&0o170000==0o120000:raise ValueError('DATASET_ARCHIVE: Đường dẫn không an toàn.')
            total+=item.file_size
            if total>1024**3:raise ValueError('DATASET_ARCHIVE: Dataset giải nén tối đa 1 GB.')
        z.extractall(destination)
    if not (destination/'manifest.json').is_file():raise ValueError('DATASET_MANIFEST: ZIP cần manifest.json ở thư mục gốc. Xem mẫu định dạng dữ liệu.')
    return destination/'manifest.json'


def create_text_csv(csv_path,destination,text_column='text',target_column='label',seed=42):
    import pandas as pd
    from .engine.tasks import save_json
    frame=pd.read_csv(csv_path)
    if len(frame)>250000 or len(frame.columns)>500:raise ValueError('CSV_LIMIT: Tối đa 250.000 dòng / 500 cột.')
    if text_column not in frame or target_column not in frame:raise ValueError(f'CSV_COLUMNS: Cần cột {text_column!r} và {target_column!r}. Các cột hiện có: '+', '.join(map(str,frame.columns[:30])))
    if frame[[text_column,target_column]].isna().any().any():raise ValueError('CSV_MISSING: Cột văn bản và nhãn cần được điền đầy đủ.')
    labels=sorted(frame[target_column].astype(str).unique())
    if not 2<=len(labels)<=1000:raise ValueError('CSV_LABELS: Cần 2–1000 lớp.')
    records=[dict(text=str(text),label=labels.index(str(label))) for text,label in zip(frame[text_column],frame[target_column])]
    splits,dropped=stratified_task_split(records,lambda row:' '.join(row['text'].split()).lower(),seed)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);path=destination/'manifest.json'
    save_json(path,dict(format='neurostudio.task-dataset.v1',name=Path(csv_path).stem,task='text_classification',labels=labels,splits=splits,source=dict(sha256=file_hash(csv_path),text_column=text_column,target_column=target_column),deduplication=dict(removed=dropped)))
    return path


def stratified_task_split(records,identity,seed):
    import random
    groups={};seen={};dropped=0
    for record in records:
        key=identity(record)
        if not key:raise ValueError('DATASET_EMPTY: Có mẫu trống.')
        if key in seen:
            if seen[key]!=record['label']:raise ValueError('LABEL_CONFLICT: Cùng dữ liệu có nhiều nhãn khác nhau.')
            dropped+=1;continue
        seen[key]=record['label'];groups.setdefault(record['label'],[]).append(record)
    splits=dict(train=[],validation=[],test=[])
    for rows in groups.values():
        if len(rows)<4:raise ValueError('DATASET_SMALL: Mỗi lớp cần ít nhất 4 mẫu khác nhau để chia ba tập.')
        random.Random(seed).shuffle(rows);n=max(1,round(len(rows)*.15))
        splits['test']+=rows[:n];splits['validation']+=rows[n:2*n];splits['train']+=rows[2*n:]
    return splits,dropped


def create_image_task(source):
    from .engine.tasks import save_json
    root=Path(source['path']);labels=sorted(path.name for path in root.iterdir() if path.is_dir())
    if not 2<=len(labels)<=1000:raise ValueError('IMAGE_LABELS: Cần 2–1000 thư mục lớp.')
    records=[]
    for label,name in enumerate(labels):
        for path in sorted((root/name).glob('*')):
            if path.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:records.append(dict(image=str(path.relative_to(root)),label=label))
    if source.get('splits'):
        splits={key:[records[i] for i in indices] for key,indices in source['splits'].items()};dropped=0
    else:splits,dropped=stratified_task_split(records,lambda row:file_hash(root/row['image']),42)
    path=root/'task-manifest.json';save_json(path,dict(format='neurostudio.task-dataset.v1',name=source['name'],task='image_classification',labels=labels,splits=splits,source=dict(dataset_id=source['id'],source_hash=source['hash']),deduplication=dict(removed=dropped)))
    return path


def adapt_dataset(source,destination,task,control=lambda:None):
    """Create an independent manifest without altering the source or its split boundaries."""
    import shutil
    from .engine.tasks import load_dataset,save_json,child
    data=load_dataset(source,control=control)
    expected={'image_reconstruction':'image_classification','text_generation':'text_classification'}
    if expected.get(task)!=data['task']:raise ValueError('DATASET_ADAPT: Unsupported conversion.')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if task=='image_reconstruction':
        for rows in data['splits'].values():
            for row in rows:
                control();file=child(Path(data['root']),row['image']);target=destination/row['image'];target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(file,target)
    result={k:data[k] for k in ['format','labels','splits']};result.update(name=data.get('name','Dataset')+' · '+task,task=task,source=dict(parent_manifest_sha256=data['manifest_hash'],conversion='Inputs reused with original split membership; labels ignored for reconstruction/generation.'))
    path=destination/'manifest.json';save_json(path,result);return path


def create_demo(destination,task,control=lambda:None):
    """Small offline teaching datasets, with honest provenance and explicit splits."""
    import numpy as np
    from PIL import Image
    from .engine.tasks import save_json
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    splits={key:[] for key in ['train','validation','test']}
    if task in ['image_classification','image_reconstruction']:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        data=load_digits();indices=np.arange(len(data.target));train,holdout=train_test_split(indices,test_size=.3,random_state=42,stratify=data.target);val,test=train_test_split(holdout,test_size=.5,random_state=42,stratify=data.target[holdout])
        for name,items in [('train',train[:160]),('validation',val[:40]),('test',test[:40])]:
            for i in items:
                control();filename=f'digit-{i}.png';Image.fromarray((data.images[i]*255/16).astype('uint8')).convert('RGB').resize((32,32)).save(destination/filename);splits[name].append(dict(image=filename,label=int(data.target[i])))
        labels=list(map(str,range(10)));source=dict(title='UCI Optical Recognition of Handwritten Digits · scikit-learn load_digits',url='https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset',subset=True,seed=42);name='Digits · Offline teaching subset'
    elif task in ['text_classification','text_generation']:
        # Original synthetic phrases. Not a public benchmark or an accuracy claim.
        words=['cat','dog','bird','fish','horse','rabbit','fox','bear','tiger','lion','duck','wolf','deer','frog','owl','mouse']
        for name,offset,count in [('train',0,48),('validation',48,12),('test',60,12)]:
            for i in range(offset,offset+count):
                animal=words[i%len(words)];text=f'the {animal} is {"small" if i%2 else "large"} near tree {i}'
                row=dict(text=text,label=i%2)
                if task=='text_generation':row['target_text']=f'{animal} size {"small" if i%2 else "large"}'
                splits[name].append(row)
        labels=['large','small'];source=dict(title='Original LAILA synthetic teaching phrases',synthetic=True,purpose='Tests configuration and execution; not a language quality benchmark.');name='LAILA · Synthetic text demo'
    else:raise ValueError('DEMO_TASK: No offline demo for this task.')
    path=destination/'manifest.json';save_json(path,dict(format='neurostudio.task-dataset.v1',name=name,task=task,labels=labels,splits=splits,source=source));return path


def create_series_csv(csv_path,destination,value_column='value',time_column='',window=16,horizon=1):
    import pandas as pd
    import numpy as np
    from .engine.tasks import save_json
    frame=pd.read_csv(csv_path)
    if not 50<=len(frame)<=250000 or value_column not in frame:raise ValueError('SERIES_DATA: Need 50-250000 ordered rows and the selected numeric value column.')
    values=pd.to_numeric(frame[value_column],errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all():raise ValueError('SERIES_VALUES: Values must be finite.')
    if time_column:
        if time_column not in frame:raise ValueError('SERIES_TIME: Time column is missing.')
        times=pd.to_datetime(frame[time_column],errors='raise')
        if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():raise ValueError('SERIES_TIME: Timestamps must be unique and strictly increasing.')
    bounds=[0,int(len(values)*.7),int(len(values)*.85),len(values)];splits={}
    for key,start,end in zip(['train','validation','test'],bounds[:-1],bounds[1:]):
        rows=[]
        for i in range(start+window,end-horizon+1):rows.append(dict(values=values[i-window:i].tolist(),target=float(values[i+horizon-1]),label=0,source_start=i-window,source_end=i+horizon-1))
        if len(rows)<2:raise ValueError('SERIES_SPLIT: More rows are needed for this window/horizon in each chronological split.')
        splits[key]=rows
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);path=destination/'manifest.json'
    save_json(path,dict(format='neurostudio.task-dataset.v1',name=Path(csv_path).stem,task='time_series',labels=['forecast'],splits=splits,source=dict(sha256=file_hash(csv_path),value_column=value_column,time_column=time_column,window=window,horizon=horizon,policy='Chronological 70/15/15 raw-row boundaries; windows never cross boundaries.')))
    return path
