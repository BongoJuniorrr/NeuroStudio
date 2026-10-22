from __future__ import annotations
import importlib.metadata
import json
import os
import inspect
from pathlib import Path
import shutil
import zipfile
from .util import atomic_json, file_hash


def compile_project(project, destination):
    from .core.validation import resolve
    from .engine import portable
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    spec=resolve(project);atomic_json(dest/"config.json",spec)
    shutil.copy2(portable.__file__,dest/"runtime.py")
    from .engine import objectives
    shutil.copy2(objectives.__file__,dest/'objectives.py')
    from .engine import layers
    shutil.copy2(layers.__file__,dest/'layers.py')
    if project.backend == "torch":
        graph=next(g for g in spec["graphs"] if g["kind"]=="model")
        nodes={n["id"]:n for n in graph["nodes"]}
        names=layers.NN_NAMES
        lines=["import torch", "from torch import nn", "from layers import make_layer", "", "class Model(nn.Module):", "    def __init__(self):", "        super().__init__()", "        self.layers = nn.ModuleDict()"]
        for key in spec["execution_order"]["model"]:
            n=nodes[key];t=n["type"]
            if t in ("Input","Output","Add","Multiply","Concatenate"):continue
            args=", ".join(f"{k}={v!r}" for k,v in n["params"].items())
            expr=f"nn.{names[t]}({args})" if t in names else f"make_layer({t!r}, {n['params']!r})"
            lines.append(f"        self.layers[{key!r}] = {expr}")
        lines.extend(["", "    def forward(self, x):", "        values = {}"])
        for key in spec["execution_order"]["model"]:
            n=nodes[key];t=n["type"]
            parents=[e["source"] for e in sorted(graph["edges"],key=lambda e:(e["order"],e["source"])) if e["target"]==key]
            refs=[f"values[{parent!r}]" for parent in parents]
            if t=="Input":expr="x"
            elif t=="Output":lines.append(f"        return {refs[0]}");continue
            elif t=="Add":expr=f"{refs[0]} + {refs[1]}"
            elif t=="Multiply":expr=f"{refs[0]} * {refs[1]}"
            elif t=="Concatenate":expr=f"torch.cat([{', '.join(refs)}], dim={n['params']['dim']})"
            else:expr=f"self.layers[{key!r}]({refs[0]}"+(".clone()" if n["params"].get("inplace") else "")+")"
            lines.append(f"        values[{key!r}] = {expr}  # {t}")
        (dest/"model.py").write_text('\n'.join(lines)+'\n',encoding="utf-8")
        with open(dest/"runtime.py","a",encoding="utf-8") as f:
            f.write('\n# This generated Model is used for both UI execution and standalone export.\ndef build_model(spec):\n    from model import Model\n    return Model()\n')
    else:
        source=inspect.getsource(portable.sklearn_model).replace('def sklearn_model(spec, frame):','def build_pipeline(spec, frame):',1)
        (dest/"model.py").write_text('"""Generated preprocessing and estimator pipeline, fit on training data only."""\nfrom runtime import make_estimator\n\n'+source,encoding="utf-8")
        with open(dest/"runtime.py","a",encoding="utf-8") as f:
            f.write('\n# UI execution and export share this generated pipeline.\ndef sklearn_model(spec, frame):\n    from model import build_pipeline\n    return build_pipeline(spec, frame)\n')
    (dest/"dataset.py").write_text('from runtime import tabular_data, image_data, split_indices\n',encoding="utf-8")
    (dest/"train.py").write_text('''import json
from pathlib import Path
from runtime import train
if __name__ == "__main__":
    here=Path(__file__).resolve().parent
    spec=json.loads((here/"config.json").read_text(encoding='utf-8'))
    descriptor=json.loads((here/"dataset-source.json").read_text(encoding='utf-8'))
    train(spec,descriptor,here/"artifacts",emit=lambda kind,data:print(kind,data,flush=True))
''',encoding="utf-8")
    (dest/"inference.py").write_text('''import argparse, json
from pathlib import Path
from runtime import inference
if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",help="JSON rows for tabular or normalized CHW float tensor for image")
    args=p.parse_args();sample=json.loads(Path(args.input).read_text(encoding='utf-8')) if args.input else None
    print(json.dumps(inference(Path(__file__).resolve().parent/"artifacts",sample),ensure_ascii=False))
''',encoding="utf-8")
    (dest/"evaluate.py").write_text('''import json
from pathlib import Path
from runtime import evaluate_holdout
if __name__ == "__main__":
    p=Path(__file__).resolve().parent
    print(json.dumps(evaluate_holdout(json.loads((p/"config.json").read_text(encoding='utf-8')),json.loads((p/"dataset-source.json").read_text(encoding='utf-8')),p/"artifacts")))
''',encoding="utf-8")
    return spec


def export_run(run_dir,destination):
    run_dir=Path(run_dir);dest=Path(destination)
    original_manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8')) if (run_dir/'manifest.json').is_file() else {}
    if not original_manifest.get('completed'):raise ValueError('RUN_NOT_COMPLETE: Chỉ xuất run đã hoàn tất.')
    shutil.copytree(run_dir/"compiled-project",dest,dirs_exist_ok=True)
    artifacts=dest/"artifacts";artifacts.mkdir(exist_ok=True)
    names=["config.json","manifest.json","splits.json","pipeline.joblib","model.safetensors","sample.json","sample.npy","evaluation.json","test-evaluation.json","study.json"]
    for name in names:
        if (run_dir/name).is_file():shutil.copy2(run_dir/name,artifacts/name)
    if not (artifacts/"manifest.json").exists():raise ValueError("RUN_NOT_COMPLETE")
    manifest=json.loads((artifacts/"manifest.json").read_text(encoding='utf-8'));versions=manifest["versions"]
    required={"numpy","pandas","scikit-learn","scipy","joblib","threadpoolctl"}
    if manifest["backend"]=="torch":required.update({"torch","torchvision","safetensors","pillow"})
    requirements=[f"{name}=={version}" for name,version in versions.items() if name in required]
    torch_version=versions.get("torch", "") if manifest["backend"]=="torch" else ""
    if "+cpu" in torch_version: requirements.insert(0,"--extra-index-url https://download.pytorch.org/whl/cpu")
    elif "+cu" in torch_version: requirements.insert(0,"--extra-index-url https://download.pytorch.org/whl/"+torch_version.split("+")[1])
    (dest/"requirements.txt").write_text('\n'.join(requirements)+'\n', encoding='utf-8')
    # Raw user data is intentionally not exported. Built-in Iris/Diabetes can be reconstructed.
    spec=json.loads((dest/"config.json").read_text(encoding='utf-8'))
    descriptor={"kind":"iris","hash":manifest["dataset_hash"]} if spec["data"]["dataset_id"]=="iris" else {"kind":"diabetes","hash":manifest["dataset_hash"]} if spec["data"]["dataset_id"]=="diabetes" else {"kind":"cifar10" if spec["data"]["dataset_id"]=="cifar10" else "csv" if spec["backend"]=="sklearn" else "images","path":"REPLACE_WITH_LOCAL_DATASET_PATH","hash":manifest["dataset_hash"]}
    if manifest.get('explicit_splits'):descriptor['splits']=json.loads((run_dir/'splits.json').read_text(encoding='utf-8'))
    atomic_json(dest/"dataset-source.json",descriptor)
    (dest/"README.md").write_text('''# Standalone Neuro Studio export
Python 3.12. Install `python -m pip install -r requirements.txt`, then `python inference.py`.
No Neuro Studio installation is required. Inference uses the included sample, fitted preprocessing and labels.
For another input: `python inference.py --input sample.json` (tabular rows or normalized CHW floats).
For images, resize RGB to config.data.image_size, divide pixels by 255, then normalize (x-0.5)/0.5 if enabled; use CHW layout.
`python train.py` retrains; `python evaluate.py` evaluates holdout. Set dataset-source.json path to your data first (except built-in Iris/Diabetes).
The raw dataset is not included. Image folders must have class_name/image.jpg layout. CIFAR-10 must already be downloaded to the indicated path.
A CUDA torch version may require its corresponding official PyTorch wheel index. CPU inference is supported.
Only load pipeline.joblib from your own trusted export: pickle/joblib can execute code. Never load an untrusted model artifact.
Training is not guaranteed bitwise reproducible across hardware/library versions. Check manifest.json and splits.json.
''',encoding="utf-8")
    tests=dest/"tests";tests.mkdir(exist_ok=True)
    (tests/"test_inference.py").write_text('''import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import inference
def test_sample():
    result=inference(Path(__file__).resolve().parents[1]/"artifacts")
    assert result
    if "value" in result[0]:assert isinstance(result[0]["value"],float)
    else:assert result[0]["label"] and abs(sum(result[0]["scores"])-1)<1e-4
''', encoding='utf-8')
    hashes={p.relative_to(dest).as_posix():file_hash(p) for p in dest.rglob('*') if p.is_file() and p!=dest/'manifest.json' and '__pycache__' not in p.parts}
    atomic_json(dest/"manifest.json",dict(format="neurostudio-export-v1",files=hashes,run=manifest))
    archive=dest.with_suffix('.zip')
    pending=archive.with_suffix('.zip.pending')
    with zipfile.ZipFile(pending,"w",zipfile.ZIP_DEFLATED) as z:
        for p in dest.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(dest))
    os.replace(pending,archive)
    return archive
