"""Download small, traceable acceptance datasets outside the source tree.

Usage: python scripts/prepare_benchmarks.py --root /path/on/data/disk
No training or access to existing workspaces. Archives remain as source evidence.
"""
from __future__ import annotations
import csv,random
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


EXPECTED={'penn-fudan': {'PennFudanPed.zip': '9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2'}, 'spoken-digits': {'fsdd.zip': 'beaca4fb849cad13103b2cffba913b3d7fdb5f15222da4ddd07187b3c38a757a'}, 'ag-news': {'train.csv': '76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e', 'test.csv': '521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699', 'classes.txt': '612dbfa932f07a6ed837295a7d5746ce7412e7bb4bc32ba919fdb3cda3a1b9b5', 'readme.txt': '779c2e6bbf93dd810d9c7b318006daef7a4bfc34062002461d251dc10fb9f58c'}}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def fetch(url,path,limit=200*1024**2,control=lambda:None,emit=lambda *args:None):
    if path.is_file():return
    pending=path.with_suffix(path.suffix+'.pending')
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'NeuroStudio-Benchmark/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response,open(pending,'wb') as out:
            size=0
            while block:=response.read(1024**2):
                control();size+=len(block)
                emit('download',dict(bytes=size,total_bytes=int(response.headers.get('Content-Length',0)),percent=round(size/max(1,int(response.headers.get('Content-Length',size)))*100,1)))
                if size>limit:raise ValueError('Download exceeds configured size limit')
                out.write(block)
        pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def extract(archive,destination):
    with zipfile.ZipFile(archive) as z:
        total=0
        for item in z.infolist():
            path=destination/item.filename
            if not path.resolve().is_relative_to(destination.resolve()):raise ValueError('Unsafe archive path')
            if item.external_attr>>16&0o170000==0o120000:raise ValueError('Archive symlink')
            total+=item.file_size
            if total>1024**3:raise ValueError('Archive exceeds 1 GB')
        z.extractall(destination)


def prepare(root,name,control=lambda:None,emit=lambda *args:None):
    folder=root/name;folder.mkdir(parents=True,exist_ok=True)
    if name=='penn-fudan':
        files={'PennFudanPed.zip':'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'}
        metadata=dict(tasks=['object_detection','instance_segmentation'],source='https://www.cis.upenn.edu/~jshi/ped_html/',terms='Research dataset; retain original README and attribution. Review source terms before redistribution.',split='No official holdout; create a fixed image-level train/validation/test manifest before training.')
    elif name=='spoken-digits':
        revision='26eb9aaf76e81b692f806f9140c2d2777410d7a1'
        files={'fsdd.zip':f'https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/{revision}'}
        metadata=dict(tasks=['audio_classification'],source='https://github.com/Jakobovski/free-spoken-digit-dataset',revision=revision,terms='CC-BY-SA-4.0; retain license and attribution.',split='Official test: recording numbers 0–4. Train: 5–49. Validation must come from train only; optionally use speaker-disjoint evaluation.')
    elif name=='ag-news':
        revision='555590db4219b1243abb1918effd6a7425a2d75f'
        base=f'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/{revision}/data/ag_news_csv/'
        files={name:base+name for name in ['train.csv','test.csv','classes.txt','readme.txt']}
        metadata=dict(tasks=['text_classification'],source='https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv',revision=revision,terms='Original corpus states academic/non-commercial research use; see readme.txt. Download for local acceptance testing; do not bundle raw articles in releases.',split='Official train.csv and test.csv; validation sampled only from train.csv.')
    else:raise ValueError('DATASET_UNKNOWN: '+name)
    manifest=folder/'sources.json'
    if manifest.is_file():
        try:old=json.loads(manifest.read_text(encoding='utf-8'))
        except (ValueError,OSError):old={'files':[]}
        if all((folder/f['file']).is_file() and sha(folder/f['file'])==EXPECTED[name].get(f['file']) for f in old['files']) and {f['file'] for f in old['files']}==set(EXPECTED[name]):
            for record in old['files']:
                path=folder/record['file']
                if path.suffix=='.zip' and not any(p.is_dir() for p in folder.iterdir()):extract(path,folder)
            return old
    records=[]
    for filename,url in files.items():
        path=folder/filename
        if path.is_file() and sha(path)!=EXPECTED[name][filename]:path.unlink()
        fetch(url,path,control=control,emit=emit)
        if sha(path)!=EXPECTED[name][filename]:
            path.unlink(missing_ok=True);raise ValueError('DATASET_CHECKSUM: Nguồn dữ liệu không khớp checksum đã ghim.')
        records.append(dict(file=filename,url=url,bytes=path.stat().st_size,sha256=sha(path)))
        if path.suffix=='.zip':extract(path,folder)
    metadata.update(name=name,files=records)
    manifest.write_text(json.dumps(metadata,indent=2)+'\n', encoding='utf-8')
    print(f'{name}: ready ({sum(f["bytes"] for f in records)/1024**2:.1f} MB)',flush=True)
    return metadata



def write(root,name,task,labels,splits,source):
    seen=set();filtered={};removed={}
    for split in ['test','validation','train']:
        kept=[];removed[split]=0
        for row in splits[split]:
            identity=' '.join(row['text'].split()).lower() if 'text' in row else row.get('audio',row.get('image'))
            if identity in seen:removed[split]+=1;continue
            seen.add(identity);kept.append(row)
        filtered[split]=kept
    splits=filtered
    data=dict(deduplication=dict(policy='exact normalized text / file identity; holdout takes priority',removed=removed),format='neurostudio.task-dataset.v1',name=name,task=task,labels=labels,splits=splits,source=source)
    from .tasks import save_json
    save_json(root/f'{task}.json',data)
    print(name,{k:len(v) for k,v in splits.items()},flush=True)


def partition(items,key,seed=42):
    groups={}
    for row in items:groups.setdefault(key(row),[]).append(row)
    train=[];validation=[]
    for group in groups.values():
        random.Random(seed).shuffle(group);n=max(1,round(len(group)*.15));validation+=group[:n];train+=group[n:]
    return train,validation


def build_manifests(root,only=None):
    root=Path(root)
    penn=root/'penn-fudan';base=penn/'PennFudanPed'
    if base.is_dir() and only in [None,'penn-fudan']:
        import numpy as np
        from PIL import Image
        rows=[]
        for path in sorted((base/'PNGImages').glob('*.png')):
            mask=base/'PedMasks'/f'{path.stem}_mask.png';array=np.array(Image.open(mask));boxes=[]
            for identity in np.unique(array):
                if identity==0:continue
                y,x=np.where(array==identity)
                boxes.append([int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1])
            rows.append(dict(image=str(path.relative_to(penn)),mask=str(mask.relative_to(penn)),boxes=boxes,label=1))
        random.Random(42).shuffle(rows);test=rows[:26];validation=rows[26:52];train=rows[52:]
        for task in ['segmentation','detection']:
            write(penn,'Penn-Fudan pedestrians',task,['background','pedestrian'],dict(train=train,validation=validation,test=test),'sources.json')
    audio=root/'spoken-digits';folder=next(audio.glob('free-spoken-digit-dataset-*/recordings'),None)
    if folder and only in [None,'spoken-digits']:
        train=[];test=[]
        for path in sorted(folder.glob('*.wav')):
            digit,speaker,index=path.stem.split('_');row=dict(audio=str(path.relative_to(audio)),label=int(digit),speaker=speaker,recording=int(index))
            (test if int(index)<5 else train).append(row)
        train,validation=partition(train,lambda r:(r['label'],r['speaker']))
        write(audio,'Free Spoken Digit Dataset','audio_classification',list(map(str,range(10))),dict(train=train,validation=validation,test=test),'sources.json')
    news=root/'ag-news'
    if (news/'test.csv').is_file() and only in [None,'ag-news']:
        def read(path):
            with open(path,newline='',encoding='utf-8') as f:return [dict(text=title+' '+description,label=int(label)-1) for label,title,description in csv.reader(f)]
        train,validation=partition(read(news/'train.csv'),lambda r:r['label'])
        write(news,'AG News','text_classification',['World','Sports','Business','Sci/Tech'],dict(train=train,validation=validation,test=read(news/'test.csv')),'sources.json')

