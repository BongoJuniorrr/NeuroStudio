"""Export task adapters as standalone Python plus fitted state, without raw data."""
import json
from pathlib import Path
import shutil
import zipfile
from .util import atomic_json,file_hash


def export_task(run,destination):
    from .engine import tasks
    run=Path(run);dest=Path(destination);manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('completed'):raise ValueError('RUN_NOT_COMPLETE')
    dest.mkdir(parents=True,exist_ok=True);artifacts=dest/'artifacts';artifacts.mkdir(exist_ok=True)
    for name in ['config.json','labels.json','manifest.json','model.safetensors','vocabulary.json','evaluation.json','test-evaluation.json','history.json','pretrained-source.json','data-audit.json']:
        if (run/name).is_file():shutil.copy2(run/name,artifacts/name)
    for folder in ['tokenizer','bert-config']:
        if (run/folder).is_dir():shutil.copytree(run/folder,artifacts/folder,dirs_exist_ok=True)
    shutil.copy2(tasks.__file__,dest/'runtime.py')
    from .engine import objectives
    shutil.copy2(objectives.__file__,dest/'objectives.py')
    (dest/'inference.py').write_text('''import argparse,json
from pathlib import Path
from runtime import predict
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='Text string or path to image/WAV');args=p.parse_args()
    print(json.dumps(predict(Path(__file__).resolve().parent/'artifacts',args.input),ensure_ascii=False))
''', encoding='utf-8')
    (dest/'train.py').write_text('''import argparse,json
from pathlib import Path
from runtime import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,help='Path to task dataset manifest.json');p.add_argument('--output',default='retrained');p.add_argument('--device',default='auto');args=p.parse_args()
    config=json.loads((Path(__file__).resolve().parent/'artifacts/config.json').read_text(encoding='utf-8'));config['device']=args.device
    train(config,args.dataset,args.output,emit=lambda kind,data:print(kind,json.dumps(data),flush=True))
''', encoding='utf-8')
    (dest/'evaluate.py').write_text('''import argparse,json
from pathlib import Path
from runtime import holdout
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);args=p.parse_args()
    print(json.dumps(holdout(Path(__file__).resolve().parent/'artifacts',args.dataset)))
''', encoding='utf-8')
    requirements={name:version for name,version in manifest['versions'].items()}
    if manifest['model'] in ['bert_tiny','bert_tiny_lora']:
        import importlib.metadata
        for name in ['transformers','tokenizers','huggingface-hub','accelerate','peft']:
            if name not in requirements:requirements[name]=importlib.metadata.version(name)
    # Pin the CPU local-version wheels explicitly: PyPI's plain torch version may
    # otherwise pull multi-GB CUDA dependencies even for inference on CPU.
    common={name:version for name,version in requirements.items() if name not in {'torch','torchvision'}}
    common.update(pillow='11.2.1',numpy='2.2.6')
    cpu={**common,**{name:requirements[name].split('+')[0]+'+cpu' for name in ['torch','torchvision']}}
    (dest/'requirements.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n'+'\n'.join(f'{name}=={version}' for name,version in cpu.items())+'\n', encoding='utf-8')
    if '+cu' in requirements['torch']:
        suffix=requirements['torch'].split('+')[1]
        (dest/'requirements-gpu.txt').write_text('--extra-index-url https://download.pytorch.org/whl/'+suffix+'\n'+'\n'.join(f'{name}=={version}' for name,version in {**common,**{name:requirements[name] for name in ['torch','torchvision']}}.items())+'\n', encoding='utf-8')
    (dest/'README.md').write_text('''# Neuro Studio — exported task

Python 3.12. Create a virtual environment and install requirements.txt. Install the
requirements-gpu.txt profile if included and your NVIDIA driver supports it. Otherwise choose a compatible CUDA wheel from pytorch.org.

- Predict: `python inference.py --input "your text or image/WAV path"`
- Retrain: `python train.py --dataset /path/to/manifest.json --device auto`
- Original test: `python evaluate.py --dataset /original/dataset/manifest.json`

Input images use RGB, square resizing and the stored preprocessing settings.
Detection boxes use coordinates in the resized image. Binary masks use foreground 1.
Audio uses the first second, resampled to 8 kHz; PCM 16-bit WAV required.
Text vocabulary is fitted on training only; BERT tokenizer is included for offline inference.
All model weights are included; retraining pretrained backbones may need internet.
Raw datasets are excluded. Keep their original license and attribution separately.
A subset run verifies execution; it does not establish benchmark quality.
`artifacts/config.json` and `history.json` record the settings and training history.
No Neuro Studio installation is required.
''', encoding='utf-8')
    hashes={p.relative_to(dest).as_posix():file_hash(p) for p in sorted(dest.rglob('*')) if p.is_file() and p!=dest/'manifest.json'}
    atomic_json(dest/'manifest.json',dict(format='neurostudio.task-export.v1',files=hashes))
    archive=dest.with_suffix('.zip');pending=archive.with_suffix('.zip.pending')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(dest.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(dest).as_posix())
    pending.replace(archive);return archive
