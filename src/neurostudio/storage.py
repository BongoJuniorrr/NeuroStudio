from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time
import uuid
from .util import canonical


class Conflict(Exception):pass


class Store:
    def __init__(self, root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT);
            INSERT OR IGNORE INTO meta VALUES('schema','1');
            CREATE TABLE IF NOT EXISTS archived_projects(id TEXT PRIMARY KEY,body TEXT,deleted REAL);
            CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,revision INTEGER,body TEXT,updated REAL);
            CREATE TABLE IF NOT EXISTS versions(project_id TEXT,revision INTEGER,body TEXT,created REAL,PRIMARY KEY(project_id,revision));
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,state TEXT,spec TEXT,project_id TEXT,parent_id TEXT,created REAL,updated REAL,idempotency TEXT UNIQUE,error TEXT);
            CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,body TEXT);
            CREATE TABLE IF NOT EXISTS progress(id TEXT PRIMARY KEY,body TEXT);
            ''')
            if db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0]!='1':raise RuntimeError("DATABASE_VERSION_UNSUPPORTED")
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.root/"studio.db",timeout=15);db.row_factory=sqlite3.Row
        try:yield db;db.commit()
        except Exception:db.rollback();raise
        finally:db.close()
    def create(self, body):
        id=uuid.uuid4().hex;now=time.time()
        with self.connect() as db:
            db.execute("INSERT INTO projects VALUES(?,?,?,?)",(id,1,canonical(body),now))
            db.execute("INSERT INTO versions VALUES(?,?,?,?)",(id,1,canonical(body),now))
        return self.project(id)
    def project(self,id):
        with self.connect() as db:row=db.execute("SELECT * FROM projects WHERE id=?",(id,)).fetchone()
        if not row:raise KeyError(id)
        return dict(id=id,revision=row['revision'],project=json.loads(row['body']),updated=row['updated'])
    def projects(self):
        with self.connect() as db:rows=db.execute("SELECT id FROM projects ORDER BY updated DESC").fetchall()
        return [self.project(r['id']) for r in rows]
    def archive_project(self,id,revision):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM projects WHERE id=?',(id,)).fetchone()
            if not row:raise KeyError(id)
            if row['revision']!=revision:raise Conflict('Dự án đã thay đổi; tải lại trước khi xóa.')
            active=db.execute("SELECT 1 FROM jobs WHERE project_id=? AND state NOT IN ('COMPLETED','FAILED','CANCELLED','INTERRUPTED')",(id,)).fetchone()
            if active:raise Conflict('Dừng hoặc chờ tác vụ của dự án hoàn tất trước khi xóa.')
            db.execute('INSERT INTO archived_projects VALUES(?,?,?)',(id,canonical(dict(row)),time.time()))
            db.execute('DELETE FROM projects WHERE id=?',(id,))
        return dict(id=id,archived=True)
    def archived_projects(self):
        with self.connect() as db:rows=db.execute('SELECT * FROM archived_projects ORDER BY deleted DESC').fetchall()
        return [dict(id=r['id'],name=json.loads(json.loads(r['body'])['body'])['name'],deleted=r['deleted']) for r in rows]
    def restore_project(self,id):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT body FROM archived_projects WHERE id=?',(id,)).fetchone()
            if not row:raise KeyError(id)
            project=json.loads(row['body'])
            db.execute('INSERT INTO projects VALUES(?,?,?,?)',(id,project['revision'],project['body'],time.time()))
            db.execute('DELETE FROM archived_projects WHERE id=?',(id,))
        return self.project(id)
    def update(self,id,revision,body):
        now=time.time()
        with self.connect() as db:
            result=db.execute("UPDATE projects SET revision=revision+1,body=?,updated=? WHERE id=? AND revision=?",(canonical(body),now,id,revision))
            if result.rowcount!=1:raise Conflict("Project đã thay đổi ở tab khác. Tải lại trước khi lưu.")
            db.execute("INSERT INTO versions VALUES(?,?,?,?)",(id,revision+1,canonical(body),now))
        return self.project(id)
    def job(self,id):
        with self.connect() as db:row=db.execute("SELECT * FROM jobs WHERE id=?",(id,)).fetchone()
        if not row:raise KeyError(id)
        d=dict(row);d['spec']=json.loads(d['spec']);return d
    def jobs(self):
        with self.connect() as db:rows=db.execute("SELECT id FROM jobs ORDER BY created DESC LIMIT 200").fetchall()
        return [self.job(r['id']) for r in rows]
    def pending_jobs(self,state=None):
        with self.connect() as db:
            if state:rows=db.execute('SELECT id FROM jobs WHERE state=? ORDER BY created,id',(state,)).fetchall()
            else:rows=db.execute("SELECT id FROM jobs WHERE state NOT IN ('COMPLETED','FAILED','CANCELLED','INTERRUPTED') ORDER BY created,id").fetchall()
        return [self.job(row['id']) for row in rows]
    def enqueue(self,kind,spec,project_id=None,parent=None,key=None):
        id=uuid.uuid4().hex;now=time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if project_id and not db.execute('SELECT 1 FROM projects WHERE id=?',(project_id,)).fetchone():raise Conflict('Dự án đã bị xóa; khôi phục trước khi chạy.')
            if key:
                found=db.execute("SELECT id FROM jobs WHERE idempotency=?",(key,)).fetchone()
                if found:
                    existing=self.job(found['id'])
                    if existing['spec']!=spec or existing['kind']!=kind or existing['project_id']!=project_id or existing['parent_id']!=parent:raise Conflict("IDEMPOTENCY_CONFLICT")
                    return existing
            inserted=db.execute("INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)",(id,kind,"QUEUED",canonical(spec),project_id,parent,now,now,key,None))
            if inserted.rowcount==0:
                found=db.execute('SELECT id FROM jobs WHERE idempotency=?',(key,)).fetchone()
                existing=self.job(found['id'])
                if existing['spec']!=spec or existing['kind']!=kind or existing['project_id']!=project_id or existing['parent_id']!=parent:raise Conflict('IDEMPOTENCY_CONFLICT')
                return existing
        return self.job(id)
    def cancel_queued(self,id):
        with self.connect() as db:
            changed=db.execute("UPDATE jobs SET state='CANCELLED',updated=? WHERE id=? AND state='QUEUED'",(time.time(),id))
            return changed.rowcount==1
    def claim(self,id):
        with self.connect() as db:
            changed=db.execute("UPDATE jobs SET state='PREPARING',updated=? WHERE id=? AND state='QUEUED'",(time.time(),id))
            return changed.rowcount==1
    def state(self,id,state,error=None):
        with self.connect() as db:db.execute("UPDATE jobs SET state=?,updated=?,error=? WHERE id=?",(state,time.time(),error,id))
    def save_dataset(self,id,body):
        with self.connect() as db:db.execute("INSERT OR REPLACE INTO datasets VALUES(?,?)",(id,canonical(body)))
    def dataset(self,id):
        if id=='diabetes':return dict(id='diabetes',kind='diabetes',name='Diabetes · Hồi quy',hash='sklearn-diabetes-v1',task='regression')
        if id=='iris':return dict(id='iris',kind='iris',name='Iris',hash='sklearn-iris-v1',task='classification')
        with self.connect() as db:row=db.execute("SELECT body FROM datasets WHERE id=?",(id,)).fetchone()
        if not row:raise ValueError("DATASET_NOT_READY: Nhập hoặc tải dataset trước khi train.")
        return json.loads(row['body'])
