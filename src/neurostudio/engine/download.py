"""Bounded ranged downloads with resume and final official checksum verification."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import os
from pathlib import Path
import time
import urllib.request

CIFAR_URL='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_MD5='c58f30108f718f92721af3b95e74349a'


def download_cifar(root,emit=lambda *args:None,control=lambda:None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);target=root/'cifar-10-python.tar.gz'
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest()==CIFAR_MD5:return
    req=urllib.request.Request(CIFAR_URL,headers={'Range':'bytes=0-0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        if r.status!=206:raise RuntimeError('DOWNLOAD_RANGE: Nguồn không hỗ trợ tải tiếp; thử lại sau.')
        size=int(r.headers['Content-Range'].split('/')[-1]);r.read()
    chunks=16;parts=root/'.cifar-download';parts.mkdir(exist_ok=True);width=(size+chunks-1)//chunks
    def fetch(i):
        lo=i*width;hi=min(size-1,(i+1)*width-1);part=parts/f'{i:02}.part';expected=hi-lo+1
        if part.exists() and part.stat().st_size==expected:return i
        for attempt in range(3):
            try:
                control()
                offset=part.stat().st_size if part.exists() else 0
                if offset>expected:part.unlink();offset=0
                req=urllib.request.Request(CIFAR_URL,headers={'Range':f'bytes={lo+offset}-{hi}'})
                with urllib.request.urlopen(req,timeout=30) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {lo+offset}-{hi}/{size}':raise RuntimeError('DOWNLOAD_RANGE_MISMATCH')
                    with open(part,'ab') as out:
                        while block:=r.read(256*1024):control();out.write(block)
                if part.stat().st_size!=expected:raise RuntimeError('DOWNLOAD_INCOMPLETE')
                return i
            except Exception:
                if attempt==2:raise
                time.sleep(.5)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,i) for i in range(chunks)]
        for future in as_completed(futures):
            future.result();control();done=sum(p.stat().st_size for p in parts.glob('*.part'))
            emit('download',dict(bytes=done,total_bytes=size,percent=round(done/size*100,1)))
    tmp=root/'cifar-complete.pending';h=hashlib.md5()
    with open(tmp,'wb') as out:
        for i in range(chunks):
            with open(parts/f'{i:02}.part','rb') as src:
                while block:=src.read(1024**2):out.write(block);h.update(block)
    if h.hexdigest()!=CIFAR_MD5:
        tmp.unlink()
        for part in parts.glob('*.part'):part.unlink()
        raise RuntimeError('DOWNLOAD_CHECKSUM: File tải không khớp checksum. Đã xóa dữ liệu tạm; nhấn Tải lại.')
    os.replace(tmp,target)
    for part in parts.iterdir():part.unlink()
    parts.rmdir()
