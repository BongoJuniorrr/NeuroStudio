"""Build explicit train/validation/test datasets from separate user uploads."""
from pathlib import Path
import json,hashlib,shutil


def build(spec,control=lambda:None):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from .data import extract_images
    from .portable import audit_duplicate_splits
    destination=Path(spec['directory']);destination.mkdir(parents=True,exist_ok=True);uploads=spec['files'];frames={};records={};labels=[]
    image=Path(uploads['train']).suffix.lower()=='.zip';target=spec['target'];task=spec['task'];mode=spec['mode'];seed=spec.get('seed',42)
    for split,path in uploads.items():
        control()
        if image:
            folder=destination/('input-'+split);extract_images(path,folder)
            for directory in sorted(folder.iterdir()):
                if directory.is_dir() and directory.name not in labels:labels.append(directory.name)
            records[split]=[(file,parent.name) for parent in sorted(folder.iterdir()) if parent.is_dir() for file in sorted(parent.iterdir()) if file.is_file() and file.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp']]
        else:
            frame=pd.read_csv(path)
            if target not in frame:raise ValueError('SPLIT_TARGET: Every CSV must include the selected target column.')
            if frames and list(frame.columns)!=list(next(iter(frames.values())).columns):raise ValueError('SPLIT_COLUMNS: CSV columns must match in name and order.')
            if len(frame)>250000 or len(frame.columns)>500:raise ValueError('SPLIT_LIMIT: At most 250000 rows and 500 columns.')
            if frame[target].isna().any():raise ValueError('SPLIT_TARGET: Targets cannot be missing.')
            frames[split]=frame
    labels=sorted(labels)
    counts={key:len(value) for key,value in (records if image else frames).items()}
    if any(n<2 for n in counts.values()) or sum(counts.values())>250000:raise ValueError('SPLIT_SIZE: Each uploaded split needs at least two samples; total at most 250000.')
    partitions={};offset=0;all_labels=[];fingerprints=[];row_paths=[]
    for name in ['train','validation','test']:
        if name not in counts:continue
        partitions[name]=list(range(offset,offset+counts[name]));offset+=counts[name]
        if image:
            for source,label in records[name]:
                control();h=hashlib.sha256(source.read_bytes()).hexdigest();fingerprints.append(h);all_labels.append(labels.index(label));row_paths.append((source,label))
        else:
            frame=frames[name];all_labels.extend(frame[target].astype(str).tolist());fingerprints.extend(pd.util.hash_pandas_object(frame.drop(columns=target).astype(str),index=False).astype(str).tolist())
    def divide(indices,test_fraction):
        stratify=np.asarray(all_labels)[indices] if task=='classification' else None
        try:a,b=train_test_split(indices,test_size=test_fraction,random_state=seed,stratify=stratify)
        except ValueError as e:raise ValueError('SPLIT_CLASS_COUNTS: Add more samples per class or use regression for numeric targets. '+str(e)) from e
        return list(map(int,a)),list(map(int,b))
    if mode=='auto':
        partitions['train'],rest=divide(partitions['train'],.3);partitions['validation'],partitions['test']=divide(rest,.5)
    elif mode=='external_test':partitions['train'],partitions['validation']=divide(partitions['train'],.17647058823529413)
    elif mode=='validation_test':partitions['validation'],partitions['test']=divide(partitions['validation'],spec.get('holdout_test_fraction',.5))
    elif mode=='external_validation':partitions['train'],partitions['test']=divide(partitions['train'],.17647058823529413)
    elif mode!='manual':raise ValueError('SPLIT_MODE: Unsupported split mode.')
    if set(partitions)!=set(['train','validation','test']):raise ValueError('SPLIT_FILES: Upload files required by the selected split mode.')
    audit_duplicate_splits(fingerprints,partitions)
    if task=='classification':
        train_labels={all_labels[i] for i in partitions['train']}
        if any(all_labels[i] not in train_labels for key in ['validation','test'] for i in partitions[key]):raise ValueError('SPLIT_LABELS: Every validation/test class must occur in training data.')
    source_hashes={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in uploads.items()}
    if image:
        folder=destination/'images'
        for i,(source,label) in enumerate(row_paths):
            target_path=folder/label/(f'{i:09d}'+source.suffix.lower());target_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target_path)
        # ImageFolder sorts class first, then path. Rebind original row indices to its ordering.
        ordered=sorted(range(len(row_paths)),key=lambda i:(row_paths[i][1],f'{i:09d}'+row_paths[i][0].suffix.lower()));mapping={old:new for new,old in enumerate(ordered)}
        partitions={key:[mapping[i] for i in rows] for key,rows in partitions.items()}
        inventory={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*/*')};inventory_path=destination/'inventory.json';inventory_path.write_text(json.dumps(inventory), encoding='utf-8')
        desc=dict(kind='images',path=str(folder),inventory_path=str(inventory_path),inventory_hash=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),hash=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest())
    else:
        combined=pd.concat([frames[name] for name in ['train','validation','test'] if name in frames],ignore_index=True);path=destination/'data.csv';combined.to_csv(path,index=False)
        desc=dict(kind='csv',path=str(path),hash=hashlib.sha256(path.read_bytes()).hexdigest())
    desc.update(id=spec['dataset_id'],name=spec['name'],task=task,splits=partitions,split_policy=mode,split_counts={k:len(v) for k,v in partitions.items()},source_hashes=source_hashes)
    return desc
