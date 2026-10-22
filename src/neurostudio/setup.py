"""Side-by-side engine installation; the API never imports torch."""
from __future__ import annotations
from collections import deque
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from .util import atomic_json,read_json

_lock=threading.Lock()
_sessions={}
_sessions_lock=threading.Lock()


class SetupSession:
    def __init__(self,record=None):
        self.record=record
        self.cancelled=threading.Event()
        self.guard=threading.Lock()
        self.process=None
        self.thread=None

    def command(self,command,on_line,timeout=None):
        from .supervisor import kill_tree
        with self.guard:
            if self.cancelled.is_set():raise RuntimeError('Cài đặt bị gián đoạn khi ứng dụng dừng. Có thể thử lại.')
            self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            process=self.process
            if self.record:
                import psutil
                try:atomic_json(self.record,dict(pid=process.pid,created=psutil.Process(process.pid).create_time()))
                except Exception:
                    kill_tree(process);self.process=None;raise
        try:
            if timeout is None:
                for line in process.stdout:
                    if self.cancelled.is_set():raise RuntimeError('Cài đặt bị gián đoạn khi ứng dụng dừng. Có thể thử lại.')
                    on_line(line.rstrip())
                code=process.wait()
            else:
                try:
                    output,_=process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    kill_tree(process)
                    raise RuntimeError('Kiểm tra môi trường quá thời gian; thử lại hoặc chọn CPU.')
                for line in output.splitlines():on_line(line)
                code=process.returncode
            if self.cancelled.is_set():raise RuntimeError('Cài đặt bị gián đoạn khi ứng dụng dừng. Có thể thử lại.')
            if code:raise RuntimeError('Cài đặt không thành công; xem log và thử lại hoặc chọn CPU.')
        finally:
            if process.poll() is None:kill_tree(process)
            process.stdout.close()
            with self.guard:
                self.process=None
                if self.record:self.record.unlink(missing_ok=True)

    def stop(self):
        from .supervisor import kill_tree
        self.cancelled.set()
        with self.guard:
            if self.process is not None and self.process.poll() is None:kill_tree(self.process)
        if self.thread:self.thread.join(timeout=8)


def recover_setup(root):
    """Only terminate a recorded installer when PID and creation time still match."""
    import psutil
    record=Path(root)/'setup-process.json'
    info=read_json(record,{})
    try:
        process=psutil.Process(info['pid'])
        if abs(process.create_time()-info['created'])<.01:
            children=process.children(recursive=True)
            for child in children:
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            process.terminate()
            _,alive=psutil.wait_procs([process,*children],timeout=2)
            for child in alive:
                try:child.kill()
                except psutil.NoSuchProcess:pass
    except (KeyError,psutil.Error):pass
    finally:record.unlink(missing_ok=True)


def stop_setup(root):
    with _sessions_lock:session=_sessions.get(str(Path(root).resolve()))
    if session:session.stop()



def recommend_profile(lines):
    cards=[line.split(',') for line in lines.strip().splitlines() if line.strip()]
    if not cards:return 'cpu'
    return 'cuda' if all(float(c[0].strip())>=7.5 and int(c[1].strip().split('.')[0])>=570 for c in cards) else 'cuda-legacy'



def start_setup(root,profile,supervisor):
    analysis=profile=='analysis' or profile.startswith('full-')
    full=profile.startswith('full-')
    if profile=='analysis':profile='tabular'
    if full:profile='nlp-'+profile[5:]
    nlp=profile.startswith('nlp-')
    if nlp:profile=profile[4:]
    if profile=='auto':
        try:
            hardware=subprocess.run(['nvidia-smi','--query-gpu=compute_cap,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
            profile=recommend_profile(hardware.stdout)
        except (OSError,subprocess.SubprocessError,ValueError):profile='cpu'
    if profile not in {'tabular','cpu','cuda','cuda-legacy'}:raise ValueError('PROFILE_UNSUPPORTED')
    if nlp and profile=='tabular':raise ValueError('NLP cần PyTorch.')
    label=('full-' if full else 'nlp-' if nlp else 'analysis-' if analysis else '')+profile
    if not _lock.acquire(blocking=False):raise ValueError('SETUP_RUNNING')
    session=SetupSession(Path(root)/'setup-process.json');session_key=str(Path(root).resolve())
    with _sessions_lock:_sessions[session_key]=session
    def run():
        root_path=Path(root);env_dir=root_path/'environments'/f'{label}-{time.time_ns()}'
        status=root_path/'setup.json';logs=deque(maxlen=80);last_progress=0.0
        def update(state,message):atomic_json(status,dict(state=state,profile=label,message=message,log=list(logs)))
        try:
            update('RUNNING','Chuẩn bị Python environment riêng…')
            uv=os.environ.get('NEUROSTUDIO_UV') or shutil.which('uv') or str(Path.home()/'.local'/'bin'/('uv.exe' if os.name=='nt' else 'uv'))
            commands=[[uv,'venv',str(env_dir),'--python',sys.executable]]
            python=env_dir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            lock=Path(__file__).with_name('engine-base.lock')
            commands.append([uv,'pip','install','--python',str(python),'-r',str(lock),'--require-hashes'])
            if profile!='tabular':
                torch_lock=Path(__file__).with_name(f'torch-{profile}.lock')
                commands.append([uv,'pip','install','--python',str(python),'-r',str(torch_lock),'--require-hashes'])
            if nlp:
                commands.append([uv,'pip','install','--python',str(python),'-r',str(Path(__file__).with_name('engine-nlp.lock')),'--require-hashes','--no-deps'])
            if analysis:
                commands.append([uv,'pip','install','--python',str(python),'-r',str(Path(__file__).with_name('engine-analysis.lock')),'--require-hashes','--no-deps'])
            for command in commands:
                update('RUNNING','Đang tải/cài dependencies. Lần đầu có thể mất vài phút…')
                def progress(line):
                    nonlocal last_progress
                    if line.startswith('WARN Skipping file for '):return
                    logs.append(line)
                    now=time.monotonic()
                    if now-last_progress>=.25:
                        update('RUNNING',line);last_progress=now
                session.command(command,progress)
            update('RUNNING','Kiểm tra mô hình scikit-learn…' if profile=='tabular' else 'Kiểm tra import và phép tính PyTorch…')
            probe='import sklearn; from sklearn.datasets import load_iris; from sklearn.linear_model import LogisticRegression; d=load_iris(); LogisticRegression(max_iter=500).fit(d.data,d.target)' if profile=='tabular' else 'import torch, torchvision, sklearn; x=torch.ones(2,requires_grad=True); x.sum().backward(); print(torch.__version__)'
            if analysis:probe+='; import umap; print(umap.__version__)'
            if nlp:probe+='; import transformers, peft; print(transformers.__version__,peft.__version__)'
            session.command([str(python),'-c',probe],lambda line:logs.append(line),timeout=90)
            if session.cancelled.is_set():raise RuntimeError('Cài đặt bị gián đoạn khi ứng dụng dừng. Có thể thử lại.')
            previous=read_json(root_path/'engine.json')
            history=read_json(root_path/'engine-history.json',[])
            if previous and previous not in history:history.append(previous)
            atomic_json(root_path/'engine-history.json',history)
            atomic_json(root_path/'engine.json',dict(python=str(python),profile=label))
            supervisor.engine_python=str(python);update('COMPLETED','ML nhẹ sẵn sàng cho Iris/CSV; có thể bổ sung PyTorch khi dùng CNN.' if profile=='tabular' else 'Môi trường sẵn sàng. Nhấn Kiểm tra máy để xác minh GPU.')
        except Exception as e:update('FAILED',str(e))
        finally:
            with _sessions_lock:_sessions.pop(session_key,None)
            _lock.release()
    try:
        atomic_json(Path(root)/'setup.json',dict(state='RUNNING',message='Bắt đầu cài đặt…',profile=profile,log=[]))
        session.thread=threading.Thread(target=run,daemon=True);session.thread.start()
    except Exception:
        with _sessions_lock:_sessions.pop(session_key,None)
        _lock.release();raise
    return dict(state='RUNNING')
