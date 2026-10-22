from __future__ import annotations
import io
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def extract_images(archive, destination):
    from PIL import Image
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        files=z.infolist()
        if len(files)>30000 or sum(f.file_size for f in files)>2*1024**3:raise ValueError("ARCHIVE_LIMIT: ZIP vượt giới hạn 30.000 file / 2 GB giải nén.")
        for f in files:
            path=PurePosixPath(f.filename.replace("\\","/"))
            if path.is_absolute() or any(p in ("..","") or ":" in p for p in path.parts) or stat.S_ISLNK(f.external_attr>>16):raise ValueError("UNSAFE_ARCHIVE: Đường dẫn ZIP không an toàn.")
            if f.is_dir():continue
            if f.file_size>20*1024**2 or f.file_size/max(f.compress_size,1)>250:raise ValueError("ARCHIVE_LIMIT: File ảnh hoặc tỷ lệ nén quá lớn.")
            if path.suffix.lower() not in (".png",".jpg",".jpeg",".bmp",".webp"):continue
            if len(path.parts)!=2:raise ValueError("IMAGE_LAYOUT: ZIP cần class_name/image.jpg, không thêm folder gốc.")
            raw=z.read(f)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width*image.height>16000000:raise ValueError("IMAGE_SIZE: Ảnh quá lớn.")
                image.verify()
            target=dest.joinpath(*path.parts);target.parent.mkdir(exist_ok=True);target.write_bytes(raw)
    if not list(dest.glob('*/*')):raise ValueError("EMPTY_DATASET: Không có ảnh trong ZIP.")


def profile(descriptor, target="target"):
    import numpy as np
    import pandas as pd
    if descriptor["kind"]=="iris":
        from sklearn.datasets import load_iris
        ds=load_iris(as_frame=True);frame=ds.data.copy();frame["target"]=[ds.target_names[i] for i in ds.target]
    elif descriptor['kind']=='diabetes':
        from sklearn.datasets import load_diabetes
        frame=load_diabetes(as_frame=True).frame
    elif descriptor["kind"]=="csv":frame=pd.read_csv(descriptor["path"],nrows=250001)
    else:
        from PIL import Image
        if descriptor["kind"]=="cifar10":
            return dict(kind="image",rows=60000,classes=10,class_counts={k:6000 for k in ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']},
                official_split={"train_pool":50000,"test":10000},preview=[],warnings=["Giữ nguyên 10.000 ảnh test chính thức. train/validation chia trong 50.000 ảnh train."])
        paths=sorted(Path(descriptor["path"]).glob('*/*'));counts={};resolutions={};previews=[]
        for p in paths:
            counts[p.parent.name]=counts.get(p.parent.name,0)+1
            with Image.open(p) as im:
                key=f"{im.width}×{im.height}";resolutions[key]=resolutions.get(key,0)+1
            if len(previews)<24:
                relative=str(p.relative_to(descriptor['path']));previews.append(relative)
                thumbnail=Path(descriptor['path']).parent/'thumbnails'/(relative+'.png');thumbnail.parent.mkdir(parents=True,exist_ok=True)
                if not thumbnail.exists():
                    with Image.open(p) as source:
                        image=source.convert('RGB');image.thumbnail((180,180));image.save(thumbnail,format='PNG')
        return dict(kind="image",rows=len(paths),classes=len(counts),class_counts=counts,resolutions=resolutions,preview=previews,warnings=[])
    if len(frame)==0:raise ValueError("EMPTY_DATASET: File không có dòng dữ liệu.")
    if len(frame)>250000 or len(frame.columns)>500:raise ValueError("DATASET_LIMIT: Alpha tối đa 250.000 dòng và 500 cột CSV.")
    counts=frame[target].fillna("(missing)").astype(str).value_counts().to_dict() if target in frame else {}
    numeric=frame.select_dtypes(include="number");summaries={};histograms={}
    for c in numeric.columns:
        vals=numeric[c].dropna().to_numpy();vals=vals[np.isfinite(vals)]
        if len(vals):
            hist,bins=np.histogram(vals,bins=12);histograms[c]=dict(counts=hist.tolist(),bins=bins.tolist())
            summaries[c]=dict(mean=float(vals.mean()),min=float(vals.min()),max=float(vals.max()),std=float(vals.std()))
    warnings=[]
    duplicates=int(frame.duplicated().sum())
    if duplicates:warnings.append(f"Có {duplicates} dòng trùng; cần xử lý trước train để tránh overlap.")
    if counts and min(counts.values())/max(counts.values())<0.2:warnings.append("Dữ liệu mất cân bằng lớp; xem macro-F1 và per-class report.")
    return dict(kind="tabular",rows=len(frame),columns=[dict(name=c,dtype=str(frame[c].dtype),missing=int(frame[c].isna().sum())) for c in frame],
        class_counts=counts,classes=len(counts),duplicates=duplicates,summary=summaries,histograms=histograms,
        correlation=json.loads(numeric.iloc[:,:12].sample(n=min(len(numeric),10000),random_state=42).corr().to_json()),correlation_policy='First 12 numeric columns; deterministic sample up to 10,000 rows',preview=json.loads(frame.head(30).to_json(orient="records")),warnings=warnings)
