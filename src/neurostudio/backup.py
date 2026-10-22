"""Workspace snapshots exclude runtime environments/cache and never overwrite a workspace."""
from pathlib import Path
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import tempfile
import zipfile
from .util import atomic_json,file_hash,safe_child


def backup_workspace(root,destination,control=lambda:None):
    root=Path(root).resolve();destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        db_path=Path(tmp)/'studio.db'
        with closing(sqlite3.connect(root/'studio.db')) as source,closing(sqlite3.connect(db_path)) as dest:
            with dest:source.backup(dest)
        files=[(db_path,'studio.db')]
        for folder in ['datasets','runs','exports']:
            for p in (root/folder).rglob('*'):
                if p.is_file() and not p.is_symlink() and root in p.resolve().parents:files.append((p,p.relative_to(root).as_posix()))
        if (root/'settings.json').is_file():files.append((root/'settings.json','settings.json'))
        hashes={};pending=destination.with_suffix('.pending')
        try:
            with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as archive:
                for source,name in files:
                    control();h=hashlib.sha256()
                    with source.open('rb') as src,archive.open(name,'w') as dest:
                        while block:=src.read(1024**2):control();dest.write(block);h.update(block)
                    hashes[name]=h.hexdigest()
                archive.writestr('BACKUP_MANIFEST.json',json.dumps(dict(format='neurostudio-workspace-v1',source_root=str(root),files=hashes)))
            os.replace(pending,destination)
        finally:
            if pending.exists():pending.unlink()
    return dict(path=str(destination),bytes=destination.stat().st_size,sha256=file_hash(destination))


def restore_workspace(archive,destination,expected_hash,control=lambda:None):
    archive=Path(archive);destination=Path(destination)
    if file_hash(archive)!=expected_hash:raise ValueError('BACKUP_CHECKSUM_MISMATCH')
    if destination.exists():raise ValueError('RESTORE_DESTINATION_EXISTS: Không ghi đè workspace đang có.')
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('BACKUP_MANIFEST.json'))
            if manifest['format']!='neurostudio-workspace-v1':raise ValueError('BACKUP_VERSION_UNSUPPORTED')
            for name,expected in manifest['files'].items():
                control();target=safe_child(destination,name);target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
                with z.open(name) as src,target.open('wb') as out:
                    while block:=src.read(1024**2):control();out.write(block);h.update(block)
                if h.hexdigest()!=expected:raise ValueError('BACKUP_FILE_CHECKSUM_MISMATCH')
        old=manifest['source_root'];new=str(destination.resolve())
        def remap(value):
            if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
            if isinstance(value,list):return [remap(v) for v in value]
            if isinstance(value,str) and (value==old or value.startswith(old+'/') or value.startswith(old+'\\')):return new+value[len(old):].replace('\\',os.sep).replace('/',os.sep)
            return value
        with closing(sqlite3.connect(destination/'studio.db')) as db,db:
            for id,body in db.execute('SELECT id,body FROM datasets').fetchall():db.execute('UPDATE datasets SET body=? WHERE id=?',(json.dumps(remap(json.loads(body))),id))
            for id,body in db.execute('SELECT id,spec FROM jobs').fetchall():db.execute('UPDATE jobs SET spec=? WHERE id=?',(json.dumps(remap(json.loads(body))),id))
            db.execute("UPDATE jobs SET state='FAILED',error='Backup archives are not nested in workspace backups. Create a new backup.' WHERE kind='backup' AND state='COMPLETED'")
            db.execute("UPDATE jobs SET state='INTERRUPTED',error='Restored from workspace backup' WHERE state NOT IN ('COMPLETED','FAILED','CANCELLED','INTERRUPTED')")
        for name in ('task.json','result.json'):
            for task in (destination/'runs').glob('*/'+name):atomic_json(task,remap(json.loads(task.read_text(encoding='utf-8'))))
        return dict(directory=str(destination.resolve()),message='Workspace mới đã khôi phục. Dừng ứng dụng, mở launcher với --data-dir thư mục này; cài engine nếu cần.')
    except Exception:
        import shutil
        shutil.rmtree(destination,ignore_errors=True);raise
