from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
import uuid
from urllib.parse import urlparse
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import __version__
from .core.schema import Project
from .core.templates import template
from .core.registry import REGISTRY
from .core.validation import validate, resolve, GraphError
from .storage import Store,Conflict
from .supervisor import Supervisor,TERMINAL
from .util import atomic_json,read_json,data_home,file_hash,safe_child
from .compiler import compile_project,export_run


class ResourceCache:
    def __init__(self):
        import threading
        self.lock=threading.Lock();self.value=None;self.timestamp=0
    def get(self,loader):
        import time
        with self.lock:
            if self.value is None or time.monotonic()-self.timestamp>2:
                self.value=loader();self.timestamp=time.monotonic()
            return self.value


def create_app(root=None, engine_python=None, start_worker=True):
    resources_cache=ResourceCache()
    store=Store(root or data_home());supervisor=Supervisor(store,engine_python)
    token=secrets.token_urlsafe(32)
    @asynccontextmanager
    async def lifespan(app):
        from .setup import recover_setup
        recover_setup(store.root)
        setup_state=read_json(store.root/'setup.json',{})
        if setup_state.get('state')=='RUNNING':
            setup_state.update(state='FAILED',message='Cài đặt bị gián đoạn khi ứng dụng dừng. Có thể thử lại; môi trường cũ vẫn được giữ.')
            atomic_json(store.root/'setup.json',setup_state)
        if start_worker:supervisor.start()
        yield
        from .setup import stop_setup
        stop_setup(store.root)
        if start_worker:supervisor.close()
    app=FastAPI(title='Neuro Studio',version=__version__,lifespan=lifespan)
    app.state.store=store;app.state.supervisor=supervisor;app.state.token=token
    @app.middleware('http')
    async def security(request,call_next):
        host=request.url.hostname
        if host not in {'127.0.0.1','localhost','::1'}:return JSONResponse({'detail':'LOCAL_ONLY'},status_code=403)
        origin=request.headers.get('origin')
        if origin and origin!=str(request.base_url).rstrip('/'):
            return JSONResponse({'detail':'ORIGIN_REJECTED'},status_code=403)
        if request.url.path.startswith('/api/') and request.url.path!='/api/v1/session':
            if not secrets.compare_digest(request.cookies.get('neuro_session',''),token):return JSONResponse({'detail':'SESSION_REQUIRED'},status_code=401)
            if request.method not in {'GET','HEAD'} and not secrets.compare_digest(request.headers.get('x-neurostudio',''),token):return JSONResponse({'detail':'CSRF_REJECTED'},status_code=403)
        size=request.headers.get('content-length')
        if size and (not size.isdigit() or int(size)>110*1024**2):return JSONResponse({'detail':'REQUEST_TOO_LARGE'},status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='no-referrer';response.headers['X-Frame-Options']='DENY'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'"
        return response
    @app.exception_handler(Conflict)
    async def conflict(request,e):return JSONResponse({'detail':str(e),'code':'REVISION_CONFLICT'},status_code=409)
    @app.exception_handler(KeyError)
    async def missing(request,e):return JSONResponse({'detail':'Không tìm thấy dữ liệu.','code':'NOT_FOUND'},status_code=404)
    @app.exception_handler(ValueError)
    async def invalid(request,e):return JSONResponse({'detail':str(e),'code':'INVALID_INPUT'},status_code=422)
    @app.get('/api/v1/session')
    def session():
        response=JSONResponse(dict(token=token,version=__version__))
        response.set_cookie('neuro_session',token,httponly=True,samesite='strict');response.headers['Cache-Control']='no-store';return response
    @app.get('/api/v1/system/status')
    def system_status():
        import importlib.util
        progress=None
        active=supervisor.active
        if active and store.job(active)['kind']=='download':
            events=supervisor.directory(active)/'events.jsonl'
            if events.is_file():
                for line in reversed(events.read_text(encoding='utf-8').splitlines()):
                    try:
                        event=json.loads(line)
                        if event['type']=='download':progress=event['data'];break
                    except ValueError:pass
        import importlib.util
        executable=Path(supervisor.engine_python)
        site=executable.parent.parent/('Lib/site-packages' if os.name=='nt' else 'lib/python3.12/site-packages')
        sites=[site]
        if site.is_dir():
            for pth in site.glob('*.pth'):
                for line in pth.read_text(encoding='utf-8',errors='replace').splitlines():
                    line=line.strip()
                    if line and not line.startswith(('import ','#')):
                        extra=(site/line).resolve()
                        if extra.is_dir():sites.append(extra)
        def installed(name):
            if any((directory/name).exists() for directory in sites):return True
            return str(executable)==sys.executable and importlib.util.find_spec(name) is not None
        engine_ready=executable.is_file() and installed('numpy') and installed('sklearn')
        engine_torch=installed('torch')
        from importlib import metadata
        torch_version=next((metadata.Distribution.at(directory).version for site_directory in sites for directory in site_directory.glob('torch-*.dist-info')),None)
        if torch_version is None and str(executable)==sys.executable:
            try:torch_version=metadata.version('torch')
            except metadata.PackageNotFoundError:pass
        return dict(version=__version__,progress=progress,torch_version=torch_version,engine_profile=read_json(store.root/'engine.json',{}).get('profile'),engine_ready=engine_ready,engine_torch=engine_torch,engine_nlp=installed('transformers') and installed('peft'),data_directory=str(store.root),engine_python=supervisor.engine_python,
                    active_job=active,disk_free_gb=round(shutil.disk_usage(store.root).free/1024**3,2),setup=read_json(store.root/'setup.json',{'state':'IDLE'}))
    @app.post('/api/v1/system/offline-bundle')
    def offline_bundle(body:dict):
        profile=body.get('profile','tabular')
        if profile not in ['tabular','analysis','cpu']:raise ValueError('OFFLINE_PROFILE: Choose tabular, analysis or cpu.')
        if shutil.disk_usage(store.root).free<3*1024**3:raise ValueError('DISK_FULL: At least 3 GB of free space is required.')
        return enqueue('offline_bundle',dict(profile=profile,output_directory=str(store.root/'exports'),timeout_seconds=7200))
    @app.get('/api/v1/system/resources')
    def resources():
        from .hardware import system_resources
        return resources_cache.get(system_resources)
    @app.get('/api/v1/system/hardware')
    def hardware():
        from .hardware import hardware_inventory
        return hardware_inventory()
    @app.post('/api/v1/system/doctor')
    def doctor():return enqueue('doctor',{})
    @app.get('/api/v1/templates')
    def templates():return {name:template(name).model_dump() for name in ['iris','iris_rf','iris_svm','iris_pca','regression','diabetes_forest','cifar10']}
    @app.get('/api/v1/catalog/blocks')
    def blocks():return list(REGISTRY.values())
    @app.get('/api/v1/projects')
    def projects():return store.projects()
    @app.post('/api/v1/projects')
    def create(project:Project):return store.create(project.model_dump())
    @app.get('/api/v1/project-trash')
    def project_trash():return store.archived_projects()
    @app.post('/api/v1/project-trash/{id}/restore')
    def restore_project(id:str):return store.restore_project(id)
    @app.delete('/api/v1/projects/{id}')
    async def delete_project(id:str,request:Request):
        body=await request.json()
        with supervisor.operation_lock:return store.archive_project(id,int(body['revision']))
    @app.get('/api/v1/projects/{id}')
    def get_project(id:str):return store.project(id)
    @app.put('/api/v1/projects/{id}')
    async def save_project(id:str,request:Request):
        body=await request.json();return store.update(id,int(body['revision']),Project.model_validate(body['project']).model_dump())
    @app.post('/api/v1/projects/{id}/clone')
    def clone(id:str):
        p=store.project(id)['project'];p['name']+=' (bản sao)';return store.create(p)
    @app.get('/api/v1/projects/{id}/backup')
    def backup(id:str):return store.project(id)['project']
    @app.get('/api/v1/projects/{id}/versions')
    def versions(id:str):
        with store.connect() as db:return [dict(r) for r in db.execute('SELECT revision,created FROM versions WHERE project_id=? ORDER BY revision DESC',(id,)).fetchall()]
    @app.post('/api/v1/graphs/validate')
    def graph_validate(project:Project):return validate(project)
    @app.post('/api/v1/graphs/code')
    def code(project:Project):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p=compile_project(project,tmp)
            source=(Path(tmp)/'model.py').read_text(encoding='utf-8')
        return dict(code=source,resolved=p)
    def project_dataset(project):
        desc=store.dataset(project.data.dataset_id)
        if desc.get('task') and desc['task']!=project.task:raise ValueError('DATASET_TASK: Dataset mẫu không phù hợp với loại bài toán; chọn Diabetes cho hồi quy hoặc Iris cho phân loại.')
        if (desc['kind'] in {'iris','csv','diabetes'}) != (project.backend=='sklearn'):
            expected='ảnh (CIFAR-10, ZIP hoặc thư mục ảnh)' if project.backend=='torch' else 'bảng (Iris hoặc CSV)'
            raise ValueError(f"MODALITY: Dự án {project.backend} cần dữ liệu {expected}. Dataset đang chọn: {desc.get('name',desc['kind'])}. Mở Dữ liệu để chọn lại; kiến trúc mô hình được giữ nguyên.")
        return desc
    def enqueue(kind,spec,project_id=None,parent=None,key=None):
        if read_json(store.root/'setup.json',{}).get('state')=='RUNNING':raise Conflict('Đang cài môi trường, vui lòng chờ.')
        job=supervisor.submit(kind,spec,project_id,parent,key);supervisor.ready(job['id']);return job
    @app.post('/api/v1/exploration')
    def exploration(body:dict):
        from .exploration_schema import Exploration
        config=Exploration.model_validate(body).model_dump();dataset=store.dataset(config['dataset_id'])
        if dataset['kind'] not in {'iris','diabetes','csv'}:raise ValueError('EXPLORATION_DATA: Chọn dữ liệu bảng.')
        return enqueue('exploration',dict(config=config,dataset=dataset,timeout_seconds=600))
    @app.post('/api/v1/exploration/{id}/export')
    def exploration_export(id:str):
        job=store.job(id)
        if job['kind']!='exploration' or job['state']!='COMPLETED':raise Conflict('Chọn lần khám phá đã hoàn tất.')
        return enqueue('exploration_export',dict(run_directory=str(supervisor.directory(id)),output_directory=str(store.root/'exports'/id)),parent=id)
    @app.get('/api/v1/tasks/catalog')
    def tasks_catalog():
        from .advanced_schema import TASK_MODELS
        from .task_datasets import catalog,benchmark_root
        capabilities=system_status()
        with store.connect() as db:images=[json.loads(row[0]) for row in db.execute('SELECT body FROM datasets').fetchall() if json.loads(row[0]).get('kind')=='images']
        return dict(models=TASK_MODELS,datasets=catalog(store),images=images,benchmark_directory=str(benchmark_root(store.root)),capabilities={key:capabilities[key] for key in ['engine_torch','engine_nlp']},benchmarks=[dict(id='penn-fudan',name='Penn-Fudan · ảnh + mask + bbox',size_mb=51,tasks=['segmentation','detection'],terms='Dữ liệu nghiên cứu; giữ attribution gốc.'),dict(id='spoken-digits',name='Spoken Digits · WAV',size_mb=16,tasks=['audio_classification'],terms='CC-BY-SA-4.0'),dict(id='ag-news',name='AG News · văn bản',size_mb=30,tasks=['text_classification'],terms='Nguồn gốc giới hạn nghiên cứu / phi thương mại; xem README đi kèm.')])
    @app.post('/api/v1/tasks/benchmarks/{name}/download')
    def benchmark_download(name:str):
        from .task_datasets import benchmark_root
        if name not in {'penn-fudan','spoken-digits','ag-news'}:raise ValueError('DATASET_UNKNOWN')
        return enqueue('benchmark_download',dict(name=name,root=str(benchmark_root(store.root)),timeout_seconds=1800))
    @app.post('/api/v1/tasks/series/import')
    async def import_series(file:UploadFile=File(...),value_column:str=Form('value'),time_column:str=Form(''),window:int=Form(16),horizon:int=Form(1)):
        if Path(file.filename or '').suffix.lower()!='.csv' or not 2<=window<=256 or not 1<=horizon<=256:raise ValueError('SERIES_IMPORT: CSV required; window/horizon must be within limits.')
        identifier=uuid.uuid4().hex;directory=store.root/'datasets'/identifier;directory.mkdir(parents=True);path=directory/'source.csv';total=0
        try:
            with path.open('wb') as out:
                while chunk:=await file.read(1024**2):
                    total+=len(chunk)
                    if total>100*1024**2:raise HTTPException(413,'Maximum 100 MB.')
                    out.write(chunk)
            return enqueue('task_import',dict(series_csv=str(path),value_column=value_column,time_column=time_column,window=window,horizon=horizon,directory=str(directory/'data'),dataset_id=identifier,name=file.filename))
        except Exception:
            shutil.rmtree(directory,ignore_errors=True);raise
    @app.post('/api/v1/tasks/datasets/import')
    async def task_import(file:UploadFile=File(...),text_column:str=Form('text'),target_column:str=Form('label')):
        ext=Path(file.filename or '').suffix.lower()
        if ext not in {'.zip','.csv'}:raise ValueError('DATASET_FILE: Chọn ZIP chứa manifest.json hoặc CSV văn bản.')
        if not text_column.strip() or not target_column.strip() or max(len(text_column),len(target_column))>500:raise ValueError('CSV_COLUMNS: Tên cột không hợp lệ.')
        id=uuid.uuid4().hex;folder=store.root/'datasets'/id;folder.mkdir(parents=True)
        archive=folder/('source'+ext);total=0
        try:
            with open(archive,'wb') as out:
                while chunk:=await file.read(1024**2):
                    total+=len(chunk)
                    if total>100*1024**2:raise HTTPException(413,'Dataset ZIP tối đa 100 MB.')
                    out.write(chunk)
            return enqueue('task_import',dict(archive=str(archive),directory=str(folder/'data'),dataset_id=id,name=file.filename,text_column=text_column,target_column=target_column))
        except Exception:
            shutil.rmtree(folder,ignore_errors=True);raise
    @app.post('/api/v1/tasks/datasets/from-images/{id}')
    def task_from_images(id:str):
        source=store.dataset(id)
        if source['kind']!='images':raise ValueError('IMAGE_DATASET: Chọn bộ ảnh đã nhập.')
        return enqueue('task_import',dict(source_images=source,dataset_id=uuid.uuid4().hex,name=source['name']))
    @app.post('/api/v1/tasks/datasets/adapt/{id}')
    def task_adapt(id:str,body:dict):
        from .task_datasets import catalog
        source=next((d for d in catalog(store) if d['id']==id),None)
        target=body.get('task')
        if not source or {'image_reconstruction':'image_classification','text_generation':'text_classification'}.get(target)!=source['task']:raise ValueError('DATASET_ADAPT: Chọn dữ liệu nguồn tương thích.')
        identifier=uuid.uuid4().hex
        return enqueue('task_import',dict(adapt_source=source,adapt_task=target,directory=str(store.root/'datasets'/identifier/'data'),dataset_id=identifier,name=source['name']+' · '+target))
    @app.post('/api/v1/tasks/demo/{task}')
    def task_demo(task:str):
        if task not in {'image_classification','image_reconstruction','text_classification','text_generation'}:raise ValueError('DEMO_TASK: Chọn demo ảnh hoặc văn bản.')
        identifier=uuid.uuid4().hex
        return enqueue('task_import',dict(demo_task=task,directory=str(store.root/'datasets'/identifier/'data'),dataset_id=identifier,name='LAILA demo · '+task))
    @app.post('/api/v1/tasks/train')
    def task_train(body:dict):
        from .advanced_schema import TaskTraining
        from .task_datasets import catalog
        config=TaskTraining.model_validate(body).model_dump()
        capabilities=system_status()
        if not capabilities['engine_torch']:raise ValueError('PYTORCH_REQUIRED: Mở Máy & cài đặt để cài PyTorch tự động.')
        if config['model'] in ['bert_tiny','bert_tiny_lora','bert_base','distilbert','deberta','hf_classifier','gpt2','distilgpt2','hf_causal'] and not capabilities['engine_nlp']:raise ValueError('NLP_REQUIRED: Mở Máy & cài đặt và chọn Cài môi trường NLP + GPU / CPU.')
        dataset=next((d for d in catalog(store) if d['id']==config['dataset_id']),None)
        if not dataset:raise ValueError('DATASET_MISSING: Chọn dataset trước khi train.')
        if dataset['task']!=config['task']:raise ValueError('DATASET_TASK: Dataset và bài toán không tương thích.')
        if shutil.disk_usage(store.root).free<1024**3:raise ValueError('DISK_FULL: Cần ít nhất 1 GB trống.')
        return enqueue('task_train',dict(config=config,dataset=dataset,timeout_seconds=config['timeout_seconds']))
    @app.post('/api/v1/tasks/study')
    def task_study(body:dict):
        from .advanced_schema import TaskStudy
        from .task_datasets import catalog
        study=TaskStudy.model_validate(body);capabilities=system_status()
        if not capabilities['engine_torch']:raise ValueError('PYTORCH_REQUIRED')
        if any(c.model in ['bert_tiny','bert_tiny_lora','bert_base','distilbert','deberta','hf_classifier','gpt2','distilgpt2','hf_causal'] for c in study.candidates) and not capabilities['engine_nlp']:raise ValueError('NLP_REQUIRED')
        first=study.candidates[0];dataset=next((d for d in catalog(store) if d['id']==first.dataset_id),None)
        if not dataset or dataset['task']!=first.task:raise ValueError('DATASET_TASK: Chọn dữ liệu phù hợp.')
        return enqueue('task_study',dict(candidates=[c.model_dump() for c in study.candidates],dataset=dataset,timeout_seconds=min(86400,sum(c.timeout_seconds for c in study.candidates))))
    @app.post('/api/v1/tasks/{id}/evaluate')
    def task_evaluate(id:str):
        job=store.job(id)
        if job['kind']!='task_train' or job['state']!='COMPLETED':raise Conflict('Chọn lần train đã hoàn tất.')
        return enqueue('task_evaluate',dict(run_directory=str(supervisor.directory(id)),dataset=job['spec']['dataset']),parent=id)
    @app.post('/api/v1/tasks/{id}/predict')
    def task_predict(id:str,body:dict):
        job=store.job(id)
        if job['kind']!='task_train' or job['state']!='COMPLETED':raise Conflict('Chọn lần train đã hoàn tất.')
        text=body.get('text')
        if job['spec']['config']['task'] not in ['text_classification','text_generation','time_series'] or not isinstance(text,str) or not text.strip() or len(text)>100000:raise ValueError('PREDICT_TEXT: Cần văn bản hợp lệ cho model văn bản.')
        return enqueue('task_predict',dict(run_directory=str(supervisor.directory(id)),sample=text),parent=id)
    @app.post('/api/v1/tasks/{id}/predict-file')
    async def task_predict_file(id:str,file:UploadFile=File(...)):
        job=store.job(id)
        if job['kind']!='task_train' or job['state']!='COMPLETED':raise Conflict('Chọn lần train đã hoàn tất.')
        if job['spec']['config']['task'] in ['text_classification','text_generation']:raise ValueError('PREDICT_TEXT: Dùng ô văn bản để dự đoán.')
        ext=Path(file.filename or '').suffix.lower();audio=job['spec']['config']['task']=='audio_classification'
        if ext not in ({'.wav'} if audio else {'.png','.jpg','.jpeg'}):raise ValueError('PREDICT_FILE: Cần WAV cho audio hoặc PNG/JPG cho ảnh.')
        folder=store.root/'uploads';folder.mkdir(exist_ok=True);path=folder/(uuid.uuid4().hex+ext);size=0
        try:
            with open(path,'wb') as out:
                while chunk:=await file.read(1024**2):
                    size+=len(chunk)
                    if size>10*1024**2:raise HTTPException(413,'Mẫu dự đoán tối đa 10 MB.')
                    out.write(chunk)
            return enqueue('task_predict',dict(run_directory=str(supervisor.directory(id)),sample=str(path),uploaded_sample=True),parent=id)
        except Exception:
            path.unlink(missing_ok=True);raise
    @app.post('/api/v1/tasks/{id}/export')
    def task_export(id:str):
        job=store.job(id)
        if job['kind']!='task_train' or job['state']!='COMPLETED':raise Conflict('Chọn lần train đã hoàn tất.')
        return enqueue('task_export',dict(run_directory=str(supervisor.directory(id)),output_directory=str(store.root/'exports'/id)),parent=id)
    @app.get('/api/v1/datasets')
    def datasets():
        with store.connect() as db:rows=db.execute('SELECT body FROM datasets').fetchall()
        return [store.dataset('iris'),store.dataset('diabetes')]+[json.loads(r[0]) for r in rows]
    @app.post('/api/v1/datasets/{id}/profile')
    async def dataset_profile(id:str,request:Request):
        body=await request.json();return enqueue('profile',dict(dataset=store.dataset(id),target=body.get('target','target')))
    @app.post('/api/v1/datasets/import-splits')
    async def import_splits(request:Request):
        async with request.form(max_files=3,max_fields=8) as form:
            mode=str(form.get('mode','auto'));target=str(form.get('target','target'));task=str(form.get('task','classification'))
            required={'auto':{'train'},'manual':{'train','validation','test'},'external_test':{'train','test'},'external_validation':{'train','validation'},'validation_test':{'train','validation'}}
            files={key:form[key] for key in ['train','validation','test'] if form.get(key) and getattr(form[key],'filename',None)}
            if mode not in required or set(files)!=required[mode] or task not in ['classification','regression'] or not 1<=len(target)<=500:raise ValueError('SPLIT_FILES: Upload the files required by the selected mode.')
            extensions={Path(file.filename).suffix.lower() for file in files.values()}
            if len(extensions)!=1 or not extensions<={'.csv','.zip'} or extensions=={'.zip'} and task!='classification':raise ValueError('SPLIT_FORMAT: Use matching CSV files or image ZIPs. Images support classification.')
            fraction=float(form.get('holdout_test_fraction',.5))
            if not 0<fraction<1:raise ValueError('SPLIT_RATIO: Holdout test fraction must be between 0 and 1.')
            identifier=uuid.uuid4().hex;folder=store.root/'datasets'/identifier;folder.mkdir(parents=True);paths={}
            try:
                for key,file in files.items():
                    path=folder/(key+next(iter(extensions)));total=0
                    with path.open('wb') as out:
                        while chunk:=await file.read(1024**2):
                            total+=len(chunk)
                            if total>100*1024**2:raise HTTPException(413,'Maximum 100 MB per split.')
                            out.write(chunk)
                    if not total:raise ValueError('SPLIT_EMPTY: Empty upload.')
                    paths[key]=str(path)
                spec=dict(directory=str(folder/'prepared'),files=paths,dataset_id=identifier,name=Path(files['train'].filename).stem+' - explicit splits',mode=mode,target=target,task=task,holdout_test_fraction=fraction)
                return enqueue('import',dict(dataset={'kind':'csv'},split_import=spec,target=target))
            except Exception:
                shutil.rmtree(folder,ignore_errors=True);raise
    @app.post('/api/v1/datasets/import')
    async def import_dataset(file:UploadFile=File(...)):
        ext=Path(file.filename or '').suffix.lower()
        if ext not in {'.csv','.zip'}:raise ValueError('FILE_TYPE: Chỉ nhận CSV hoặc ZIP ảnh.')
        id=uuid.uuid4().hex;folder=store.root/'datasets'/id;folder.mkdir(parents=True)
        path=folder/('source'+ext);size=0
        try:
            with open(path,'wb') as f:
                while chunk:=await file.read(1024**2):
                    size+=len(chunk)
                    if size>100*1024**2:raise HTTPException(413,'Tối đa 100 MB mỗi lần nhập.')
                    f.write(chunk)
            if not size:raise ValueError('EMPTY_DATASET: File trống.')
            desc=dict(id=id,kind='csv' if ext=='.csv' else 'images',name=Path(file.filename).name,hash=file_hash(path),path=str(path if ext=='.csv' else folder/'images'))
            if ext=='.zip':desc['archive']=str(path)
            return enqueue('import',dict(dataset=desc))
        except Exception:
            shutil.rmtree(folder,ignore_errors=True);raise
    @app.post('/api/v1/datasets/folder')
    async def import_folder(request:Request):
        import zipfile
        from pathlib import PurePosixPath
        async with request.form(max_files=5000,max_fields=10) as form:
            uploads=form.getlist('files')
            if not uploads:raise ValueError('EMPTY_DATASET: Thư mục không có ảnh.')
            paths=[PurePosixPath(f.filename.replace('\\','/')) for f in uploads]
            strip_root=all(len(p.parts)>=3 for p in paths) and len({p.parts[0] for p in paths})==1
            id=uuid.uuid4().hex;folder=store.root/'datasets'/id;folder.mkdir(parents=True)
            archive=folder/'source.zip';total=0
            try:
                with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                    for upload,path in zip(uploads,paths):
                        parts=path.parts[1:] if strip_root else path.parts
                        if path.is_absolute() or len(parts)!=2 or any(part in ('.','..') or ':' in part for part in parts):raise ValueError('IMAGE_LAYOUT: Chọn thư mục chứa class_name/image.jpg.')
                        with z.open('/'.join(parts),'w') as dest:
                            while chunk:=await upload.read(1024**2):
                                total+=len(chunk)
                                if total>100*1024**2:raise HTTPException(413,'Tối đa 100 MB cho folder import; dùng ZIP cho cấu trúc lớn hơn.')
                                dest.write(chunk)
                desc=dict(id=id,kind='images',name='Image folder',path=str(folder/'images'),archive=str(archive),hash=file_hash(archive))
                return enqueue('import',dict(dataset=desc))
            except Exception:
                shutil.rmtree(folder,ignore_errors=True);raise
    @app.post('/api/v1/datasets/{id}/deduplicate')
    def deduplicate(id:str):
        source=store.dataset(id)
        if source['kind'] not in {'csv','iris'}:raise ValueError('Chỉ hỗ trợ xóa dòng trùng CSV/Iris ở Alpha.')
        new_id=uuid.uuid4().hex;folder=store.root/'datasets'/new_id;folder.mkdir(parents=True)
        descriptor=dict(id=new_id,kind='csv',name=source.get('name','Dataset')+' · deduplicated',path=str(folder/'data.csv'),parent_id=id)
        return enqueue('clean',dict(dataset=source,output_dataset=descriptor))
    @app.post('/api/v1/preview')
    def transform_preview(project:Project):
        return enqueue('preview',dict(project=resolve(project),dataset=project_dataset(project)))
    @app.post('/api/v1/explain')
    def explain():return enqueue('explain',{})
    @app.post('/api/v1/datasets/cifar10/download')
    def download_cifar():
        if read_json(store.root/'engine.json',{}).get('profile')=='tabular':raise ValueError('PYTORCH_REQUIRED: Cài PyTorch trước khi tải CIFAR-10.')
        return enqueue('download',dict(dataset=dict(id='cifar10',kind='cifar10',name='CIFAR-10',hash='cifar10-official-torchvision-md5',path=str(store.root/'datasets'/'cifar10'))))
    @app.get('/api/v1/datasets/{id}/image')
    def dataset_image(id:str,name:str):
        desc=store.dataset(id)
        if desc['kind']!='images':raise HTTPException(404)
        path=safe_child(desc['path'],name)
        if not path.is_file():raise HTTPException(404)
        return FileResponse(path)
    @app.post('/api/v1/runs')
    async def submit_run(request:Request):
        body=await request.json()
        if read_json(store.root/'setup.json',{}).get('state')=='RUNNING':raise Conflict('Đang cài môi trường, vui lòng chờ.')
        saved=store.project(body['project_id'])
        if saved['revision']!=body['revision']:raise Conflict('Phiên bản project đã thay đổi. Lưu/tải lại trước khi train.')
        project=Project.model_validate(saved['project']);resolved=resolve(project);desc=project_dataset(project)
        if project.backend=='torch' and read_json(store.root/'engine.json',{}).get('profile')=='tabular':raise ValueError('PYTORCH_REQUIRED: Mở Máy & cài đặt để bổ sung PyTorch (Auto GPU/CPU).')
        if shutil.disk_usage(store.root).free<250*1024**2:raise ValueError('DISK_FULL: Cần tối thiểu 250 MB trống để chạy.')
        spec=dict(project=resolved,dataset=desc,project_revision=saved['revision'])
        parent=body.get('resume_from')
        if parent:
            previous=store.job(parent)
            if previous['state'] not in {'FAILED','INTERRUPTED','CANCELLED','COMPLETED'}:raise Conflict('Run trước chưa dừng.')
            checkpoint=supervisor.directory(parent)/'last.pt'
            if not checkpoint.is_file():raise ValueError('NO_CHECKPOINT: Chưa có checkpoint epoch để resume.')
            spec.update(resume=str(checkpoint),resume_hash=file_hash(checkpoint))
        job=supervisor.submit('train',spec,saved['id'],parent,body.get('idempotency_key'))
        out=supervisor.directory(job['id'])
        if not (out/'ready').exists():
            try:
                compile_project(project,out/'compiled-project');supervisor.ready(job['id'])
            except Exception as e:
                store.state(job['id'],'FAILED','Không thể chuẩn bị file thực thi.')
                raise ValueError('STAGING_FAILED: Không thể ghi file thực thi. Kiểm tra dung lượng và quyền ghi workspace.') from e
        return job
    @app.get('/api/v1/runs')
    def runs():return [{**{k:v for k,v in j.items() if k!='spec'},**({'name':j['spec']['config']['name']} if j['kind']=='task_train' else {})} for j in store.jobs()]
    @app.get('/api/v1/runs/{id}')
    def run(id:str):
        job=store.job(id);out=supervisor.directory(id)
        return dict(**job,manifest=read_json(out/'manifest.json'),evaluation=read_json(out/'evaluation.json'),test_evaluation=read_json(out/'test-evaluation.json'),result=read_json(out/'result.json'),study=read_json(out/'study.json'),history=read_json(out/'history.json'),predictions=read_json(out/'predictions.json'))
    @app.post('/api/v1/runs/{id}/study')
    async def submit_study(id:str,request:Request):
        import copy,math,random
        previous=store.job(id)
        if previous['kind']!='train' or previous['state']!='COMPLETED' or previous['spec']['project']['backend']!='sklearn':raise Conflict('Chọn run scikit-learn đã hoàn tất để kiểm tra chéo.')
        body=await request.json();folds=body.get('folds',5);budget=body.get('budget_seconds',120);trials=body.get('trials',10)
        if type(folds) is not int or not 2<=folds<=10 or type(trials) is not int or not 1<=trials<=20 or type(budget) is not int or not 10<=budget<=1800:raise ValueError('STUDY_LIMIT: folds 2–10, trials 1–20, budget 10–1800 giây.')
        mode=body.get('mode','cv');metric=body.get('metric','r2_mean' if previous['spec']['project'].get('task')=='regression' else 'accuracy_mean')
        if mode not in {'cv','search'} or metric not in {'accuracy_mean','macro_f1_mean','r2_mean','neg_rmse_mean'}:raise ValueError('STUDY_OPTIONS_INVALID')
        base=Project.model_validate({k:v for k,v in previous['spec']['project'].items() if k in Project.model_fields})
        if (base.task=='regression')!=(metric in {'r2_mean','neg_rmse_mean'}):raise ValueError('STUDY_METRIC: Tiêu chí không phù hợp với loại bài toán.')
        node=next(g for g in base.graphs if g.kind=='model').nodes[0]
        space=body.get('space',{}) if mode=='search' else {}
        if not isinstance(space,dict) or len(space)>5:raise ValueError('SEARCH_SPACE: Tối đa 5 tham số.')
        for key,values in space.items():
            if key not in REGISTRY[node.type]['params'] or not isinstance(values,list) or not 1<=len(values)<=10:raise ValueError('SEARCH_SPACE: Mỗi tham số hợp lệ cần 1–10 lựa chọn.')
        keys=sorted(space);count=math.prod(len(space[k]) for k in keys)
        indices=random.Random(base.training.seed).sample(range(count),min(trials,count))
        candidates=[]
        for index in indices:
            candidate=copy.deepcopy(base);params=next(g for g in candidate.graphs if g.kind=='model').nodes[0].params
            for key in reversed(keys):
                choices=space[key];params[key]=choices[index%len(choices)];index//=len(choices)
            candidates.append(resolve(candidate))
        return enqueue('study',dict(project=previous['spec']['project'],dataset=previous['spec']['dataset'],estimator=node.type,mode=mode,metric=metric,folds=folds,budget_seconds=budget,timeout_seconds=budget+10,candidates=candidates,run_directory=str(supervisor.directory(id)),source_run=id),previous['project_id'],id)
    @app.post('/api/v1/runs/{id}/cancel')
    def cancel(id:str):return supervisor.cancel(id)
    @app.post('/api/v1/runs/{id}/stop')
    def stop(id:str):
        j=store.job(id)
        if j['kind']!='train' or j['spec']['project']['backend']!='torch':raise ValueError('STOP_UNSUPPORTED: Tác vụ này dùng Hủy.')
        return supervisor.cancel(id,True)
    @app.get('/api/v1/runs/{id}/events')
    def events(id:str,after_seq:int=0,stream:bool=False):
        store.job(id);path=supervisor.directory(id)/'events.jsonl'
        def read(after):
            if not path.exists():return []
            lines=[]
            with open(path,encoding='utf-8') as f:
                for line in f:
                    try:
                        e=json.loads(line)
                        if e['seq']>after:lines.append(e)
                    except json.JSONDecodeError:break
            return lines
        if not stream:return read(after_seq)
        async def generate():
            seq=after_seq
            while True:
                for event in read(seq):
                    seq=event['seq'];yield f"id: {seq}\ndata: {json.dumps(event,ensure_ascii=False)}\n\n"
                if store.job(id)['state'] in TERMINAL:break
                yield ': heartbeat\n\n';await asyncio.sleep(0.7)
        return StreamingResponse(generate(),media_type='text/event-stream',headers={'Cache-Control':'no-cache'})
    @app.post('/api/v1/runs/{id}/evaluate-test')
    def test_evaluate(id:str):
        j=store.job(id)
        if j['kind']!='train' or j['state']!='COMPLETED':raise Conflict('Chờ training hoàn tất trước khi đánh giá test.')
        count=read_json(supervisor.directory(id)/'test-history.json',[]);count.append(time.time());atomic_json(supervisor.directory(id)/'test-history.json',count)
        return enqueue('evaluate',dict(**j['spec'],run_directory=str(supervisor.directory(id))),j['project_id'],id)
    @app.post('/api/v1/runs/{id}/inference')
    async def predict(id:str,request:Request):
        j=store.job(id)
        if j['kind']!='train' or j['state']!='COMPLETED':raise Conflict('Chờ training hoàn tất.')
        body=await request.json();return enqueue('inference',dict(run_directory=str(supervisor.directory(id)),sample=body.get('sample')),j['project_id'],id)
    @app.post('/api/v1/runs/{id}/feature-maps')
    async def feature_maps(id:str,request:Request):
        j=store.job(id)
        if j['kind']!='train' or j['state']!='COMPLETED':raise Conflict('Chọn run PyTorch hoàn tất.')
        body=await request.json()
        return enqueue('feature_maps',dict(run_directory=str(supervisor.directory(id)),layer=body.get('layer')),j['project_id'],id)
    @app.post('/api/v1/runs/{id}/export')
    async def export(id:str):
        j=store.job(id)
        if j['kind']!='train' or j['state']!='COMPLETED':raise Conflict('Chỉ xuất lần train hoàn tất.')
        return enqueue('export',dict(run_directory=str(supervisor.directory(id)),output_directory=str(store.root/'exports'/id)),j['project_id'],id)
    @app.get('/api/v1/runs/{id}/download')
    def download(id:str):
        store.job(id);p=store.root/'exports'/f'{id}.zip'
        if not p.exists():raise HTTPException(404)
        return FileResponse(p,filename=f'NeuroStudio-{id[:8]}.zip')
    @app.get('/api/v1/runs/{id}/metrics.csv')
    def metrics_csv(id:str):
        import csv,io
        store.job(id);path=supervisor.directory(id)/'events.jsonl';rows=[]
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                try:
                    e=json.loads(line)
                    if e['type']=='metric':rows.append(e['data'])
                except json.JSONDecodeError:pass
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=sorted({k for row in rows for k in row}));writer.writeheader();writer.writerows(rows)
        return StreamingResponse(iter([buffer.getvalue()]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=metrics.csv'})
    @app.get('/api/v1/curriculum')
    def curriculum():
        return read_json(Path(__file__).with_name('curriculum.json'))
    @app.put('/api/v1/learning/{id}')
    async def learning(id:str,request:Request):
        body=await request.json()
        lessons=read_json(Path(__file__).with_name('curriculum.json'))['lessons'];lesson=next((l for l in lessons if l['id']==id),None)
        if not lesson or not isinstance(body,dict) or set(body)-{'completed','quiz_answer'} or type(body.get('completed')) is not bool:raise ValueError('LEARNING_PROGRESS: Invalid lesson or progress.')
        if body['completed'] and (type(body.get('quiz_answer')) is not int or body['quiz_answer']!=lesson['correct']):raise ValueError('LEARNING_QUIZ: Answer the lesson quiz correctly before completing it.')
        with store.connect() as db:db.execute('INSERT OR REPLACE INTO progress VALUES(?,?)',(id,json.dumps(body)))
        return body
    @app.get('/api/v1/learning')
    def learning_progress():
        with store.connect() as db:return {r['id']:json.loads(r['body']) for r in db.execute('SELECT * FROM progress').fetchall()}
    @app.post('/api/v1/system/setup')
    async def setup(request:Request):
        from .setup import start_setup
        body=await request.json()
        with supervisor.operation_lock:
            if bool(store.pending_jobs()):raise Conflict('Dừng hoặc chờ các tác vụ trước khi sửa môi trường.')
            return start_setup(store.root,body.get('profile','cpu'),supervisor)
    @app.post('/api/v1/system/backup')
    def workspace_backup():
        id=uuid.uuid4().hex
        return enqueue('backup',dict(root=str(store.root),destination=str(store.root/'backups'/f'{id}.zip')))
    @app.get('/api/v1/system/backups/{id}/download')
    def backup_download(id:str):
        j=store.job(id)
        if j['kind']!='backup' or j['state']!='COMPLETED':raise Conflict('Backup chưa sẵn sàng.')
        result=read_json(supervisor.directory(id)/'result.json')
        return FileResponse(result['path'],filename=f'NeuroStudio-workspace-{id[:8]}.zip')
    @app.post('/api/v1/system/backups/{id}/restore')
    def restore_backup(id:str):
        j=store.job(id)
        if j['kind']!='backup' or j['state']!='COMPLETED':raise Conflict('Backup chưa sẵn sàng.')
        result=read_json(supervisor.directory(id)/'result.json')
        dest=store.root.parent/('NeuroStudio-restored-'+uuid.uuid4().hex[:8])
        return enqueue('restore',dict(archive=result['path'],sha256=result['sha256'],destination=str(dest)))
    @app.get('/api/v1/system/environments')
    def environments():
        return dict(active=read_json(store.root/'engine.json'),previous=read_json(store.root/'engine-history.json',[]))
    @app.post('/api/v1/system/environments/activate')
    async def activate_environment(request:Request):
        body=await request.json()
        with supervisor.operation_lock:
            if bool(store.pending_jobs()) or read_json(store.root/'setup.json',{}).get('state')=='RUNNING':raise Conflict('Chờ các tác vụ hoàn tất trước khi đổi môi trường.')
            history=read_json(store.root/'engine-history.json',[])
            index=body.get('index')
            if type(index) is not int or not 0<=index<len(history):raise ValueError('ENVIRONMENT_NOT_FOUND')
            selected=history[index]
            if not Path(selected['python']).is_file():raise ValueError('ENVIRONMENT_MISSING: Cài lại profile này.')
            active=read_json(store.root/'engine.json')
            if active and active not in history:history.append(active)
            atomic_json(store.root/'engine-history.json',history);atomic_json(store.root/'engine.json',selected)
            supervisor.engine_python=selected['python']
            return selected
    @app.get('/api/v1/system/diagnostics')
    def diagnostics():
        # Excludes datasets, serialized artifacts, task paths and session credentials.
        return dict(version=__version__,jobs=[dict(id=j['id'],kind=j['kind'],state=j['state'],created=j['created']) for j in store.jobs()],setup=read_json(store.root/'setup.json',{}))
    static=Path(__file__).with_name('static')
    if static.exists():app.mount('/',StaticFiles(directory=static,html=True),name='web')
    return app
