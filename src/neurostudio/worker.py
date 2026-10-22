from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
import time
import traceback
from neurostudio.util import atomic_json,read_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args()
    out=Path(args.directory);task=read_json(out/'task.json');seq=0
    if (out/'events.jsonl').exists():
        for line in (out/'events.jsonl').read_text(encoding='utf-8').splitlines():
            try:seq=max(seq,json.loads(line)['seq'])
            except (ValueError,KeyError):pass
    def emit(kind,data):
        nonlocal seq
        seq+=1;event=dict(seq=seq,run_id=out.name,type=kind,time=time.time(),data=data)
        with open(out/'events.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False,allow_nan=False,default=str)+'\n');f.flush()
        if kind=='status':atomic_json(out/'status.json',data)
    began=time.monotonic()
    def control():
        if (out/'cancel').exists():raise Cancelled('Người dùng đã hủy lần chạy.')
        timeout=task.get('timeout_seconds',task.get('project',{}).get('training',{}).get('timeout_seconds',3600))
        if time.monotonic()-began>timeout:raise RuntimeError('TIMEOUT: Đã vượt thời gian cho phép.')
    from neurostudio.engine.portable import Cancelled,Stopped,GPUPreflightError
    try:
        from neurostudio.winjob import bind_worker
        bind_worker()
        emit('status',dict(state='PREPARING'))
        kind=task['kind']
        if kind=='train':
            sys.path.insert(0,str(out/'compiled-project'))
            module_spec=importlib.util.spec_from_file_location('compiled_runtime',out/'compiled-project'/'runtime.py')
            runtime=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(runtime)
            runtime.Cancelled=Cancelled;runtime.Stopped=Stopped;runtime.GPUPreflightError=GPUPreflightError
            resume=task.get('resume')
            if resume and file_hash(resume)!=task['resume_hash']:raise ValueError('CHECKPOINT_HASH_MISMATCH')
            runtime.train(task['project'],task['dataset'],out,emit,control,lambda:(out/'stop').exists(),resume,task.get('force_cpu_reason'))
            manifest=read_json(out/'manifest.json',{})
            manifest.update(run_id=out.name,project_id=task.get('project_id'),project_revision=task.get('project_revision'),parent_run=task.get('parent_run'),
                compiled_sources={p.name:file_hash(p) for p in (out/'compiled-project').iterdir() if p.is_file()})
            atomic_json(out/'manifest.json',manifest)
        elif kind=='benchmark_download':
            from neurostudio.engine.benchmarks import prepare,build_manifests
            root=Path(task['root']);root.mkdir(parents=True,exist_ok=True)
            source=prepare(root,task['name'],control,emit);control();build_manifests(root,task['name'])
            atomic_json(out/'result.json',dict(name=task['name'],sources=source))
        elif kind=='task_import':
            from neurostudio.task_datasets import extract_bundle,create_text_csv,create_image_task,adapt_dataset,create_demo,create_series_csv
            from neurostudio.engine.tasks import load_dataset
            if task.get('series_csv'):path=create_series_csv(task['series_csv'],task['directory'],task['value_column'],task.get('time_column',''),task['window'],task['horizon'])
            elif task.get('demo_task'):path=create_demo(task['directory'],task['demo_task'],control)
            elif task.get('adapt_source'):path=adapt_dataset(task['adapt_source']['path'],task['directory'],task['adapt_task'],control)
            elif task.get('source_images'):path=create_image_task(task['source_images'])
            elif Path(task['archive']).suffix=='.csv':path=create_text_csv(task['archive'],task['directory'],task.get('text_column','text'),task.get('target_column','label'))
            else:path=extract_bundle(task['archive'],task['directory'])
            data=load_dataset(path)
            desc=dict(id=task['dataset_id'],kind='task_bundle',name=Path(task['name']).stem,task=data['task'],path=str(path),hash=file_hash(path),counts=data['counts'],labels=data['labels'],source=data.get('source'))
            atomic_json(out/'result.json',dict(dataset=desc))
        elif kind=='task_study':
            from neurostudio.engine.tasks import study
            emit('status',dict(state='RUNNING'));study(task['candidates'],task['dataset']['path'],out,emit,control)
        elif kind=='task_train':
            from neurostudio.engine.tasks import train
            if file_hash(task['dataset']['path'])!=task['dataset']['hash']:raise ValueError('DATASET_CHANGED: Manifest dataset đã thay đổi.')
            train(task['config'],task['dataset']['path'],out,emit,control)
        elif kind=='task_evaluate':
            from neurostudio.engine.tasks import holdout
            atomic_json(out/'result.json',holdout(task['run_directory'],task['dataset']['path'],control))
        elif kind=='task_predict':
            from neurostudio.engine.tasks import predict
            atomic_json(out/'result.json',predict(task['run_directory'],task['sample']))
        elif kind=='task_export':
            from neurostudio.task_export import export_task
            archive=export_task(task['run_directory'],task['output_directory'])
            atomic_json(out/'result.json',dict(url=f"/api/v1/runs/{Path(task['run_directory']).name}/download",bytes=archive.stat().st_size))
        elif kind=='exploration':
            from neurostudio.engine.exploration import run_exploration
            emit('status',dict(state='RUNNING'));run_exploration(task['config'],task['dataset'],out,control)
        elif kind=='exploration_export':
            from neurostudio.exploration_export import export_exploration
            archive=export_exploration(task['run_directory'],task['output_directory'])
            atomic_json(out/'result.json',dict(url=f"/api/v1/runs/{Path(task['run_directory']).name}/download",bytes=archive.stat().st_size))
        elif kind=='study':
            emit('status',dict(state='RUNNING'))
            from neurostudio.engine.study import study
            study(task,out,emit,control)
        elif kind=='profile':
            from neurostudio.engine.data import profile
            atomic_json(out/'result.json',profile(task['dataset'],task.get('target','target')))
        elif kind=='import':
            from neurostudio.engine.data import profile,extract_images
            desc=task['dataset']
            if task.get('split_import'):
                from neurostudio.engine.split_import import build
                desc=build(task['split_import'],control)
            if desc['kind']=='images' and not desc.get('splits'):extract_images(desc['archive'],desc['path'])
            if desc['kind']=='images':
                inventory={str(p.relative_to(desc['path'])):file_hash(p) for p in Path(desc['path']).glob('*/*') if p.is_file()}
                inventory_path=Path(desc['path']).parent/'inventory.json';atomic_json(inventory_path,inventory)
                desc['inventory_path']=str(inventory_path);desc['inventory_hash']=file_hash(inventory_path)
            result=profile(desc,task.get('target','target'));atomic_json(out/'result.json',dict(dataset=desc,profile=result))
        elif kind=='download':
            from torchvision.datasets import CIFAR10
            emit('log',dict(message='Tải CIFAR-10 (~170 MB) từ nguồn torchvision; có thể mất vài phút.'))
            from neurostudio.engine.download import download_cifar
            download_cifar(task['dataset']['path'],emit,control)
            CIFAR10(task['dataset']['path'],train=True,download=True);CIFAR10(task['dataset']['path'],train=False,download=True)
            atomic_json(out/'result.json',dict(dataset=task['dataset']))
        elif kind=='evaluate':
            from neurostudio.engine.portable import evaluate_holdout
            atomic_json(out/'result.json',evaluate_holdout(task['project'],task['dataset'],task['run_directory']))
        elif kind=='inference':
            from neurostudio.engine.portable import inference
            atomic_json(out/'result.json',inference(task['run_directory'],task.get('sample')))
        elif kind=='export':
            from neurostudio.compiler import export_run
            archive=export_run(task['run_directory'],task['output_directory'])
            atomic_json(out/'result.json',dict(url=f"/api/v1/runs/{Path(task['run_directory']).name}/download",bytes=archive.stat().st_size))
        elif kind=='clean':
            import pandas as pd
            source=task['dataset'];desc=task['output_dataset']
            if source['kind']=='iris':
                from sklearn.datasets import load_iris
                ds=load_iris(as_frame=True);frame=ds.data.copy();frame['target']=[ds.target_names[i] for i in ds.target]
            else:frame=pd.read_csv(source['path'])
            clean=frame.drop_duplicates();clean.to_csv(desc['path'],index=False)
            desc.update(hash=file_hash(desc['path']),cleaning_policy='exact_rows_keep_first',removed_rows=len(frame)-len(clean))
            from neurostudio.engine.data import profile
            atomic_json(out/'result.json',dict(dataset=desc,profile=profile(desc)))
        elif kind=='offline_bundle':
            from neurostudio.offline import build_offline
            emit('status',dict(state='RUNNING'));result=build_offline(Path(task['output_directory'])/out.name,task['profile'],control,emit);atomic_json(out/'result.json',result)
        elif kind=='backup':
            from neurostudio.backup import backup_workspace
            atomic_json(out/'result.json',backup_workspace(task['root'],task['destination'],control))
        elif kind=='restore':
            from neurostudio.backup import restore_workspace
            atomic_json(out/'result.json',restore_workspace(task['archive'],task['destination'],task['sha256'],control))
        elif kind=='feature_maps':
            from neurostudio.engine.preview import feature_maps
            atomic_json(out/'result.json',feature_maps(task['run_directory'],task.get('layer')))
        elif kind=='preview':
            from neurostudio.engine.preview import preview
            atomic_json(out/'result.json',preview(task['project'],task['dataset']))
        elif kind=='explain':
            from neurostudio.engine.preview import explain
            atomic_json(out/'result.json',explain())
        elif kind=='doctor':
            import platform
            import importlib.metadata as package_metadata
            import psutil
            result=dict(engine_python=sys.executable,python=platform.python_version(),platform=platform.platform(),cpu=os.cpu_count(),ram_gb=round(psutil.virtual_memory().total/1024**3,1),packages={})
            for name in ['numpy','pandas','scikit-learn','torch','torchvision']:
                try:result['packages'][name]=package_metadata.version(name)
                except package_metadata.PackageNotFoundError:result['packages'][name]=None
            from neurostudio.hardware import hardware_inventory
            result.update(hardware_inventory())
            try:
                import torch
                result.update(cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),cuda_device_count=torch.cuda.device_count())
                from neurostudio.engine.portable import choose_device
                d,reason=choose_device('auto');result.update(device=str(d),reason=reason)
                if d.type=='cuda':
                    import torch
                    properties=torch.cuda.get_device_properties(d)
                    result.update(device_name=properties.name,vram_total_mb=round(properties.total_memory/1024**2),cuda_runtime=torch.version.cuda)
            except Exception as e:result.update(device='cpu',reason=str(e))
            atomic_json(out/'result.json',result)
        else:raise ValueError('UNKNOWN_TASK')
        control();emit('status',dict(state='COMPLETED'))
    except GPUPreflightError as e:
        atomic_json(out/'retry-device.json',dict(reason=str(e)))
        emit('retry_device',dict(reason=str(e),device='cpu',stage='preflight'))
        emit('status',dict(state='PREPARING',message='GPU preflight thất bại; chuyển worker mới sang CPU.'))
    except Cancelled as e:emit('status',dict(state='CANCELLED',message=str(e)))
    except Stopped as e:emit('status',dict(state='INTERRUPTED',message=str(e)))
    except Exception as e:
        (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        emit('status',dict(state='FAILED',message=f'{type(e).__name__}: {e}'))

if __name__=='__main__':main()
