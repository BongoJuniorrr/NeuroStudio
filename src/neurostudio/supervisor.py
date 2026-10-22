from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from .util import atomic_json,read_json

TERMINAL={'COMPLETED','FAILED','CANCELLED','INTERRUPTED'}


def kill_tree(process):
    import psutil
    try:
        parent=psutil.Process(process.pid);children=parent.children(recursive=True)
        for p in children:
            try:p.terminate()
            except psutil.NoSuchProcess:pass
        process.terminate()
        _,alive=psutil.wait_procs(children,timeout=2)
        for p in alive:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    except psutil.NoSuchProcess:pass


class Supervisor:
    def __init__(self,store,engine_python=None):
        self.store=store;self.root=store.root;self.engine_python=engine_python or os.environ.get('NEUROSTUDIO_ENGINE_PYTHON',sys.executable)
        self.operation_lock=threading.RLock()
        self.halt=threading.Event();self.process=None;self.active=None;self.thread=None
    def start(self):
        # Workspace lock in launcher prevents multiple owners. A previous supervisor owns no surviving processes:
        # recovery records include PID + creation time to avoid PID-reuse kills.
        import psutil
        for job in self.store.pending_jobs():
            if job['state']=='QUEUED' and not (self.directory(job['id'])/'ready').exists():
                self.store.state(job['id'],'FAILED','Job staging bị gián đoạn. Hãy tạo lần chạy mới.');continue
            if job['state'] not in TERMINAL|{'QUEUED'}:
                info=read_json(self.directory(job['id'])/'process.json',{})
                try:
                    p=psutil.Process(info['pid'])
                    if abs(p.create_time()-info['created'])<0.01:
                        for child in p.children(recursive=True):child.kill()
                        p.kill()
                except (psutil.Error,KeyError):pass
                self.store.state(job['id'],'INTERRUPTED','Ứng dụng bị dừng. Có thể tiếp tục từ checkpoint hợp lệ.')
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def directory(self,id):return self.root/'runs'/id
    def submit(self,kind,spec,project_id=None,parent=None,key=None):
        from .storage import Conflict
        with self.operation_lock:
            if read_json(self.root/'setup.json',{}).get('state')=='RUNNING':raise Conflict('Đang cài môi trường, vui lòng chờ.')
            job=self.store.enqueue(kind,spec,project_id,parent,key)
            out=self.directory(job['id']);out.mkdir(parents=True,exist_ok=True)
            task=dict(spec,kind=kind,run_id=job['id'],project_id=project_id,parent_run=parent)
            try:atomic_json(out/'task.json',task)
            except OSError as e:
                self.store.state(job['id'],'FAILED','Không thể ghi task: '+str(e));raise
            return job
    def cancel(self,id,stop=False):
        job=self.store.job(id)
        if job['state'] in TERMINAL:return job
        if job['state']=='QUEUED':
            if self.store.cancel_queued(id):return self.store.job(id)
            job=self.store.job(id)
            if job['state'] in TERMINAL:return job
        flag='stop' if stop else 'cancel';(self.directory(id)/flag).touch()
        self.store.state(id,'STOP_REQUESTED' if stop else 'CANCEL_REQUESTED')
        return self.store.job(id)
    def loop(self):
        while not self.halt.wait(0.2):
            jobs=self.store.pending_jobs(state='QUEUED')
            if not jobs:continue
            job=jobs[0];self.active=job['id'];out=self.directory(job['id'])
            # Submit stages all files before exposing a runnable marker.
            if not (out/'ready').exists():continue
            if not self.store.claim(job['id']):continue
            env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
            for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='2'
            env['PYTHONUNBUFFERED']='1';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            try:
                with open(out/'worker.log','ab') as log:
                    self.process=subprocess.Popen([self.engine_python,'-m','neurostudio.worker',str(out)],env=env,stdout=log,stderr=log,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
                    import psutil
                    atomic_json(out/'process.json',dict(pid=self.process.pid,created=psutil.Process(self.process.pid).create_time()))
                    cancel_started=None;began=time.monotonic()
                    timeout=job['spec'].get('timeout_seconds',job['spec'].get('project',{}).get('training',{}).get('timeout_seconds',3600))
                    while self.process.poll() is None:
                        state=read_json(out/'status.json',{}).get('state','PREPARING')
                        current=self.store.job(job['id'])['state']
                        if current not in {'CANCEL_REQUESTED','STOP_REQUESTED'} and state not in TERMINAL:self.store.state(job['id'],state)
                        if self.halt.is_set() or (out/'cancel').exists() or time.monotonic()-began>timeout:
                            if cancel_started is None:cancel_started=time.monotonic()
                            if time.monotonic()-cancel_started>3:kill_tree(self.process);break
                        time.sleep(0.15)
                    status=read_json(out/'status.json',{})
                    retry=read_json(out/'retry-device.json')
                    task=read_json(out/'task.json',{})
                    if retry and job['kind']=='train' and status.get('state')=='PREPARING' and not task.get('force_cpu_reason') and not (out/'cancel').exists() and not self.halt.is_set():
                        task['force_cpu_reason']=retry['reason'];atomic_json(out/'task.json',task)
                        self.store.state(job['id'],'QUEUED');continue
                    state=status.get('state','FAILED');message=status.get('message')
                    if (out/'cancel').exists():state='CANCELLED'
                    elif self.halt.is_set():state='INTERRUPTED'
                    elif self.process.returncode!=0 or state not in TERMINAL:state='FAILED';message=message or 'Worker dừng bất thường. Xem worker.log trong gói chẩn đoán.'
                    if state=='COMPLETED' and job['kind'] in ('import','download','clean','task_import'):
                        result=read_json(out/'result.json');self.store.save_dataset(result['dataset']['id'],result['dataset'])
                    if state!='COMPLETED':
                        manifest=read_json(out/'manifest.json')
                        if manifest:
                            manifest.update(completed=False,terminal_state=state);atomic_json(out/'manifest.json',manifest)
                    self.store.state(job['id'],state,message)
            except Exception as e:self.store.state(job['id'],'FAILED',str(e))
            finally:self.process=None;self.active=None
    def close(self):
        self.halt.set()
        if self.thread:self.thread.join(timeout=8)
        if self.process and self.process.poll() is None:kill_tree(self.process)
    def ready(self,id):(self.directory(id)/'ready').touch()
