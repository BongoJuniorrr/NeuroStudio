"""Standalone execution source. Copied verbatim into exported projects; no app imports."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time


try:
    from .objectives import classification_metrics,add_custom,primary_score,loss_value,text_scores
except ImportError:
    from objectives import classification_metrics,add_custom,primary_score,loss_value,text_scores


try:
    from .layers import make_layer,LAYER_SPECS
except ImportError:
    from layers import make_layer,LAYER_SPECS


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False, default=str)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def execution_digest(spec):
    semantic=copy.deepcopy(spec)
    semantic.pop('name',None);semantic.pop('description',None)
    for graph in semantic.get('graphs',[]):
        for node in graph['nodes']:node.pop('position',None)
    return digest(semantic)



def compatible_execution(previous,current):
    """Normalize only defaults added after early Alpha; reject semantic changes."""
    training_defaults=dict(drop_last=False,metrics=[],primary_metric=None,primary_mode='auto',metric_formulas={},metric_code='',loss_code='',focal_gamma=2,adam_beta1=.9,adam_beta2=.999,optimizer_eps=1e-8,amsgrad=False,dampening=0,nesterov=False,rmsprop_alpha=.99,centered=False,label_smoothing=0,min_lr=0,scheduler_patience=3)
    node_defaults={name:{k:v['default'] for k,v in rule['params'].items()} for name,rule in LAYER_SPECS.items()}
    node_defaults.update({'LogisticRegression':dict(solver='lbfgs',tol=1e-4,fit_intercept=True,class_weight='none'),'KNN':dict(algorithm='auto',leaf_size=30,p=2,weights='uniform'),'Imputer':dict(strategy='median'),'StandardScaler':dict(with_mean=True,with_std=True),'MinMaxScaler':dict(clip=False),'OneHotEncoder':dict(handle_unknown='ignore',min_frequency=1)})
    def normalize(spec):
        value=copy.deepcopy(spec);value.setdefault('task','classification')
        value['training']={**training_defaults,**value['training']};value['training'].pop('gpu_memory_fraction',None)
        for graph in value['graphs']:
            for node in graph['nodes']:node['params']={**node_defaults.get(node['type'],{}),**node['params']}
        return execution_digest(value)
    return normalize(previous)==normalize(current)


def seed_all(seed, deterministic=True, include_torch=True):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    if not include_torch:return
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        torch.backends.cudnn.benchmark = not deterministic
    except ImportError:
        pass


def split_indices(y, data, seed):
    import numpy as np
    from sklearn.model_selection import train_test_split
    ix = np.arange(len(y))
    stratify = data["stratify"]
    train, rest = train_test_split(ix, train_size=data["train"], random_state=seed, stratify=y if stratify else None)
    val, test = train_test_split(rest, test_size=data["test"]/(data["validation"]+data["test"]), random_state=seed,
                                 stratify=y[rest] if stratify else None)
    # Exact duplicate samples must not cross splits; callers audit content fingerprints.
    return {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}


def resolved_splits(y,data,seed,descriptor):
    import numpy as np
    if not descriptor.get('splits'):return split_indices(y,data,seed)
    splits=descriptor['splits'];seen=set()
    if set(splits)!=set(['train','validation','test']):raise ValueError('SPLIT_KEYS: Need three explicit splits.')
    for rows in splits.values():
        if not rows or any(type(i) is not int or i<0 or i>=len(y) or i in seen for i in rows) or len(set(rows))!=len(rows):raise ValueError('SPLIT_INDICES: Split indices must be unique, disjoint and in range.')
        seen.update(rows)
    if len(seen)!=len(y):raise ValueError('SPLIT_COVERAGE: Every sample must belong to exactly one split.')
    return splits


def tabular_data(spec, descriptor):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import LabelEncoder
    if descriptor["kind"] == "iris":
        ds = load_iris(as_frame=True)
        frame = ds.data; target = pd.Series([ds.target_names[i] for i in ds.target], name="target")
    elif descriptor['kind']=='diabetes':
        from sklearn.datasets import load_diabetes
        ds=load_diabetes(as_frame=True);frame=ds.data;target=ds.target
    else:
        raw = pd.read_csv(descriptor["path"])
        col = spec["data"]["target"]
        if col not in raw.columns: raise ValueError(f"TARGET_MISSING: Không tìm thấy cột {col}.")
        target = raw[col]; frame = raw.drop(columns=[col])
    if target.isna().any(): raise ValueError("TARGET_MISSING: Nhãn không được để trống.")
    if spec["data"]["features"]: frame = frame[spec["data"]["features"]]
    if frame.empty: raise ValueError("EMPTY_FEATURES: Chọn ít nhất một đặc trưng.")
    if spec.get('task')=='regression':
        y=pd.to_numeric(target,errors='raise').to_numpy(dtype=float)
        if not np.isfinite(y).all():raise ValueError('NONFINITE_TARGET: Nhãn hồi quy cần là số hữu hạn.')
        numeric=frame.select_dtypes(include='number').to_numpy(dtype=float)
        if np.isinf(numeric).any():raise ValueError('NONFINITE: Dữ liệu chứa infinity.')
        return frame,y,[]
    enc = LabelEncoder(); y = enc.fit_transform(target.astype(str))
    if len(enc.classes_) != spec["classes"]: raise ValueError(f"CLASS_COUNT: Dữ liệu có {len(enc.classes_)} lớp; cấu hình có {spec['classes']}.")
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy(dtype=float)).all():
        if np.isinf(frame.select_dtypes(include="number").to_numpy(dtype=float)).any(): raise ValueError("NONFINITE: Dữ liệu chứa infinity.")
    return frame, y, enc.classes_.tolist()


def make_estimator(name,params,seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
    from sklearn.linear_model import LinearRegression,Ridge,Lasso
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
    params=copy.deepcopy(params)
    if params.get('class_weight')=='none':params['class_weight']=None
    if params.get('max_depth')==0:params['max_depth']=None
    if params.get('max_features')=='all':params['max_features']=None
    constructors={'LogisticRegression':LogisticRegression,'KNN':KNeighborsClassifier,'SVM':SVC,'DecisionTree':DecisionTreeClassifier,'RandomForest':RandomForestClassifier,'ExtraTrees':ExtraTreesClassifier,'GradientBoosting':GradientBoostingClassifier,'AdaBoost':AdaBoostClassifier}
    constructors.update(LinearRegression=LinearRegression,Ridge=Ridge,Lasso=Lasso,KNNRegressor=KNeighborsRegressor,SVR=SVR,DecisionTreeRegressor=DecisionTreeRegressor,RandomForestRegressor=RandomForestRegressor,GradientBoostingRegressor=GradientBoostingRegressor)
    if name not in constructors:raise ValueError('ESTIMATOR_UNSUPPORTED: '+name)
    if name not in {'KNN','KNNRegressor','LinearRegression','SVR'}:params['random_state']=seed
    if name in {'RandomForest','ExtraTrees','RandomForestRegressor'}:params['n_jobs']=1
    if name=='SVM':params['probability']=True
    return constructors[name](**params)


def sklearn_model(spec, frame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    graph = next(g for g in spec["graphs"] if g["kind"] == "data")
    nodes={n["id"]:n for n in graph["nodes"]}
    types = [nodes[k]["type"] for k in spec["execution_order"]["data"]]
    settings={n["type"]:n["params"] for n in graph["nodes"]}
    num = frame.select_dtypes(include="number").columns.tolist()
    cat = [c for c in frame.columns if c not in num]
    numeric = []
    if "Imputer" in types: numeric.append(("impute", SimpleImputer(strategy=settings.get("Imputer",{}).get("strategy","median"), keep_empty_features=True)))
    if "StandardScaler" in types: numeric.append(("scale", StandardScaler(**settings.get("StandardScaler",{}))))
    if "MinMaxScaler" in types: numeric.append(("scale", MinMaxScaler(**settings.get("MinMaxScaler",{}))))
    transforms = []
    if num: transforms.append(("numeric", Pipeline(numeric) if numeric else "passthrough", num))
    if cat:
        if "OneHotEncoder" not in types: raise ValueError("ENCODER_REQUIRED: Có cột phân loại; thêm OneHotEncoder.")
        steps = []
        if "Imputer" in types: steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        steps.append(("encode", OneHotEncoder(sparse_output=True,**(settings.get("OneHotEncoder") or {"handle_unknown":"ignore"}))))
        transforms.append(("categorical", Pipeline(steps), cat))
    n = next(g for g in spec["graphs"] if g["kind"] == "model")["nodes"][0]
    estimator = make_estimator(n["type"],n["params"],spec["training"]["seed"])
    steps=[("preprocess",ColumnTransformer(transforms))]
    if 'PCA' in types:
        from sklearn.decomposition import PCA
        steps.append(('reduce',PCA(**settings['PCA'],svd_solver='arpack',random_state=spec['training']['seed'])))
    return Pipeline([*steps,("model",estimator)])


def classification_result(y, pred, labels, scores=None, sample_ids=None):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
    ids = list(range(len(y))) if sample_ids is None else sample_ids
    result = dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y,pred,labels=list(range(len(labels)))).tolist(), labels=labels,
        report=classification_report(y,pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),
        predictions=[dict(index=int(ids[i]),actual=labels[int(a)],predicted=labels[int(b)],correct=bool(a==b),
                     confidence=float(max(scores[i])) if scores is not None else None) for i,(a,b) in enumerate(zip(y,pred))])
    if scores is not None:
        curves = []
        for k, label in enumerate(labels):
            truth = np.asarray(y)==k
            if len(np.unique(truth)) < 2: continue
            fpr,tpr,_ = roc_curve(truth, np.asarray(scores)[:,k]); precision,recall,_ = precision_recall_curve(truth,np.asarray(scores)[:,k])
            curves.append(dict(label=label, fpr=fpr.tolist(),tpr=tpr.tolist(),auc=float(auc(fpr,tpr)),precision=precision.tolist(),recall=recall.tolist()))
        result["curves"] = curves
    if scores is not None:result.update(classification_metrics(y,pred,scores))
    return result


def build_model(spec):
    import torch
    from torch import nn
    class DAG(nn.Module):
        def __init__(self):
            super().__init__()
            graph = next(g for g in spec["graphs"] if g["kind"] == "model")
            self.nodes = {n["id"]: n for n in graph["nodes"]}
            self.order = spec["execution_order"]["model"]
            self.parents = {k: [e["source"] for e in sorted(graph["edges"],key=lambda e:(e["order"],e["source"])) if e["target"]==k] for k in self.nodes}
            self.layers = nn.ModuleDict({key:make_layer(n['type'],n['params']) for key,n in self.nodes.items() if n['type'] not in ['Input','Output','Add','Multiply','Concatenate']})
        def forward(self,x):
            values={}
            for key in self.order:
                n=self.nodes[key]; t=n["type"]; xs=[values[k] for k in self.parents[key]]
                if t=="Input": values[key]=x
                elif t=="Output": return xs[0]
                elif t=="Add": values[key]=xs[0]+xs[1]
                elif t=="Multiply": values[key]=xs[0]*xs[1]
                elif t=="Concatenate":values[key]=torch.cat(xs,dim=n["params"]["dim"])
                else:values[key]=self.layers[key](xs[0].clone() if n["params"].get("inplace") else xs[0])
            raise ValueError("MODEL_OUTPUT_MISSING")
    return DAG()


def choose_device(requested, model_factory=None):
    import torch
    if requested=='cpu': return torch.device('cpu'), None
    failures=[]
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA không khả dụng trong PyTorch environment này.')
        if requested=='auto':
            candidates=list(range(torch.cuda.device_count()))
            try:candidates.sort(key=lambda i:torch.cuda.mem_get_info(i)[0],reverse=True)
            except RuntimeError:pass
        else:candidates=[int(requested.split(':')[1])]
        for index in candidates:
            try:
                d=torch.device(f'cuda:{index}')
                a=torch.ones((4,4),device=d,requires_grad=True);(a@a).sum().backward();torch.cuda.synchronize(d)
                return d,None
            except (RuntimeError,AssertionError) as e:
                failures.append(str(e))
                if requested!='auto':raise
        raise RuntimeError('; '.join(failures) or 'Không có CUDA device dùng được.')
    except (RuntimeError,AssertionError) as e:
        if requested!='auto':raise RuntimeError(f'GPU_UNAVAILABLE: {e}') from e
        return torch.device('cpu'),str(e)


def image_data(spec, descriptor, train_augmentation=False):
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets, transforms
    size=spec["data"]["image_size"]
    ops=[transforms.Resize((size,size))]
    if train_augmentation and spec["data"]["augment_crop"]: ops.append(transforms.RandomCrop(size,padding=4))
    if train_augmentation and spec["data"]["augment_flip"]: ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if spec["data"]["normalize"]:ops.append(transforms.Normalize((0.5,)*3,(0.5,)*3))
    transform=transforms.Compose(ops)
    if descriptor["kind"]=="cifar10":
        # Preserve the official test partition; only split the official training pool.
        a=datasets.CIFAR10(descriptor["path"],train=True,download=False,transform=transform)
        b=datasets.CIFAR10(descriptor["path"],train=False,download=False,transform=transform)
        from torch.utils.data import ConcatDataset
        ds=ConcatDataset([a,b]); targets=np.array(a.targets+b.targets); labels=a.classes
    else:
        ds=datasets.ImageFolder(descriptor["path"],transform=transform)
        targets=np.array(ds.targets); labels=ds.classes
    if len(labels)!=spec["classes"]:raise ValueError(f"CLASS_COUNT: Dữ liệu có {len(labels)} lớp, model có {spec['classes']}.")
    return ds,targets,labels


def image_splits(spec, descriptor, y):
    import numpy as np
    from sklearn.model_selection import train_test_split
    if descriptor.get("splits"):return resolved_splits(y,spec["data"],spec["training"]["seed"],descriptor)
    if descriptor["kind"]=="cifar10":
        ix=np.arange(50000); limit=spec["data"]["sample_limit"]
        if limit and limit<len(ix):ix,_=train_test_split(ix,train_size=limit,random_state=spec["training"]["seed"],stratify=y[ix])
        ratio=spec["data"]["validation"]/(spec["data"]["train"]+spec["data"]["validation"])
        a,b=train_test_split(ix,test_size=ratio,random_state=spec["training"]["seed"],stratify=y[ix] if spec["data"]["stratify"] else None)
        return dict(train=a.tolist(),validation=b.tolist(),test=list(range(50000,60000)))
    if spec["data"]["sample_limit"] is not None:raise ValueError("SUBSET_UNSUPPORTED: Giới hạn subset chỉ áp dụng CIFAR-10; dữ liệu ảnh riêng dùng toàn bộ.")
    return split_indices(y,spec["data"],spec["training"]["seed"])


def atomic_torch_save(path, obj):
    import torch
    path=Path(path); tmp=path.with_suffix(".pending")
    with open(tmp,"wb") as f:
        torch.save(obj,f); f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _epoch(model,loader,device,loss_fn,optimizer=None,scaler=None,amp=False,grad_clip=1,control=lambda:None,config=None):
    import torch
    model.train(optimizer is not None); total=correct=count=0;truth=[];probabilities=[]
    with torch.set_grad_enabled(optimizer is not None):
        for x,y in loader:
            control();x=x.to(device);y=y.to(device)
            with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16):
                try:logits=model(x)
                except ValueError as e:
                    if 'more than 1 value per channel' in str(e) or 'more than 1 spatial element' in str(e):raise ValueError('NORMALIZATION_BATCH: Increase batch/spatial size, enable drop_last for an incomplete training batch, or use LayerNorm for single-sample features.') from e
                    raise
                loss=loss_fn(logits,y)
            if not torch.isfinite(loss):raise ValueError("NONFINITE_LOSS: Loss NaN/Inf. Kiểm tra dữ liệu và learning rate.")
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
                scaler.step(optimizer);scaler.update()
            truth.extend(y.detach().cpu().tolist());probabilities.extend(logits.detach().softmax(1).cpu().tolist())
            total+=loss.detach().item()*len(y);correct+=(logits.argmax(1)==y).sum().item();count+=len(y)
    result=dict(loss=total/max(count,1),accuracy=correct/max(count,1),samples=count);result.update(classification_metrics(truth,[max(range(len(p)),key=p.__getitem__) for p in probabilities],probabilities))
    return add_custom(result,config or {},truth,[max(range(len(p)),key=p.__getitem__) for p in probabilities],probabilities)


class GPUPreflightError(RuntimeError):pass
class Cancelled(Exception):pass
class Stopped(Exception):pass


def verify_dataset(descriptor):
    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        return h.hexdigest()
    if descriptor['kind']=='csv' and sha(descriptor['path'])!=descriptor['hash']:
        raise ValueError('DATASET_CHANGED: File dữ liệu đã thay đổi. Nhập thành version mới.')
    if descriptor.get('inventory_path'):
        if sha(descriptor['inventory_path'])!=descriptor['inventory_hash']:raise ValueError('DATASET_CHANGED: Inventory đã thay đổi.')
        inventory=json.loads(Path(descriptor['inventory_path']).read_text(encoding='utf-8'))
        current={str(p.relative_to(descriptor['path'])) for p in Path(descriptor['path']).glob('*/*') if p.is_file()}
        if current!=set(inventory):raise ValueError('DATASET_CHANGED: Danh sách ảnh đã thay đổi.')
        for name,expected in inventory.items():
            if sha(Path(descriptor['path'])/name)!=expected:raise ValueError('DATASET_CHANGED: Nội dung ảnh đã thay đổi.')


def train(spec, descriptor, out, emit=lambda *args:None, control=lambda:None, stop_requested=lambda:False, resume=None, force_cpu_reason=None):
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    verify_dataset(descriptor)
    seed_all(spec["training"]["seed"],spec["training"]["deterministic"],include_torch=spec["backend"]=="torch")
    dump(out/"config.json",spec)
    t=spec["training"]; started=time.monotonic()
    manifest=dict(explicit_splits=bool(descriptor.get("splits")),schema_version="1.0",codegen_version=spec.get("codegen_version","reference"),spec_hash=digest(spec),execution_hash=execution_digest(spec),dataset_hash=descriptor.get("hash"),seed=t["seed"],
        requested_device=t["device"], backend=spec["backend"],task=spec.get("task","classification"),precision="fp32",metric_spec={"name":"classification-v1","average":"macro","zero_division":0},
        versions={})
    import importlib.metadata as metadata
    for name in ["numpy","pandas","scikit-learn","torch","torchvision","scipy","joblib","threadpoolctl","safetensors","pillow"]:
        try:manifest["versions"][name]=metadata.version(name)
        except metadata.PackageNotFoundError:pass
    if spec.get('task')=='regression':manifest['metric_spec']={'name':'regression-v1','metrics':['rmse','mae','r2']}
    emit("status",dict(state="PREPARING"))
    if spec["backend"]=="sklearn":
        import joblib
        frame,y,labels=tabular_data(spec,descriptor)
        splits=resolved_splits(y,{**spec["data"],"stratify":False} if spec.get("task")=="regression" else spec["data"],t["seed"],descriptor)
        # Duplicates can inflate validation/test scores; fail rather than silently leak.
        audit_duplicate_splits(frame.astype(str).apply(lambda r:digest(r.tolist()),axis=1).tolist(),splits)
        dump(out/"splits.json",splits);manifest.update(effective_device="cpu",fallback_reason="scikit-learn CPU backend",split_hash=digest(splits),labels=labels,features=frame.columns.tolist())
        dump(out/"manifest.json",manifest);emit("device",manifest)
        model=sklearn_model(spec,frame)
        estimator=model.named_steps['model']
        if hasattr(estimator,'n_neighbors') and estimator.n_neighbors>len(splits['train']):raise ValueError('KNN_NEIGHBORS: Số láng giềng không được lớn hơn số mẫu train.')
        control();emit("status",dict(state="RUNNING"))
        model.fit(frame.iloc[splits["train"]],y[splits["train"]]);control()
        tmp=out/"pipeline.pending";joblib.dump(model,tmp);os.replace(tmp,out/"pipeline.joblib")
        ix=splits['validation'];pred=model.predict(frame.iloc[ix])
        if spec.get('task')=='regression':
            result=add_custom(regression_result(y[ix],pred,ix),t,y[ix],pred)
            primary_score(result,t,'regression')
            emit('metric',dict(epoch=1,train_r2=float(model.score(frame.iloc[splits['train']],y[splits['train']])),val_r2=result['r2'],val_rmse=result['rmse'],validation={k:v for k,v in result.items() if isinstance(v,(int,float))},elapsed=time.monotonic()-started,ram_mb=process_ram()))
        else:
            scores=model.predict_proba(frame.iloc[ix]);result=add_custom(classification_result(y[ix],pred,labels,scores,ix),t,y[ix],pred,scores)
            primary_score(result,t,'classification')
            emit('metric',dict(epoch=1,train_accuracy=float(model.score(frame.iloc[splits['train']],y[splits['train']])),val_accuracy=result['accuracy'],val_f1=result['macro_f1'],validation={k:v for k,v in result.items() if isinstance(v,(int,float))},elapsed=time.monotonic()-started,ram_mb=process_ram(),samples_per_second=len(splits['train'])/max(time.monotonic()-started,.001)))
        manifest["estimator_parameters"]=estimator.get_params(deep=False)
        dump(out/"sample.json",frame.iloc[[ix[0]]].to_dict(orient="records"))
    else:
        import torch
        from torch.utils.data import DataLoader,Subset
        torch.set_num_threads(t["cpu_threads"])
        ds,y,labels=image_data(spec,descriptor);train_ds,_,_=image_data(spec,descriptor,True)
        splits=image_splits(spec,descriptor,y)
        if descriptor["kind"]!="cifar10":
            paths=[p for p,_ in ds.samples]
            audit_duplicate_splits([hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],splits)
        dump(out/"splits.json",splits)
        device,reason=choose_device("cpu" if force_cpu_reason else t["device"])
        if force_cpu_reason:reason=force_cpu_reason
        if device.type=="cuda":
            torch.cuda.set_device(device)
            fraction=t.get('gpu_memory_fraction',.2);free,total=torch.cuda.mem_get_info(device)
            if free<total*fraction+512*1024**2:raise RuntimeError('GPU_MEMORY_BUDGET: VRAM trống không đủ ngân sách đã chọn. Chọn GPU khác, giảm giới hạn hoặc dùng CPU.')
            torch.cuda.set_per_process_memory_fraction(fraction,device)
        if device.type=='cpu' and spec.get('resource_estimate'):
            try:
                import psutil
                estimate=spec['resource_estimate']
                required_mb=estimate['weights_mb']*4+estimate['layer_outputs_mb']*3
                if required_mb*1024**2 > psutil.virtual_memory().available*.6:
                    raise ValueError('MEMORY_BUDGET: Ước tính tensor vượt ngân sách RAM. Giảm batch size, image size hoặc kích thước layer.')
            except ImportError:pass
        model=build_model(spec).to(device);loss_fn=lambda prediction,target:loss_value(prediction,target,t,'classification')
        # A probe must not update BatchNorm running statistics or consume training RNG.
        buffers={key:value.detach().cpu().clone() for key,value in model.named_buffers()}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
        if t.get('drop_last') and len(splits['train'])<t['batch_size']:raise ValueError('BATCH_SIZE: Training split is smaller than batch_size with drop_last enabled.')
        # Forward/backward preflight. Shape/data errors propagate; do not treat them as GPU failures.
        x0,y0=next(iter(DataLoader(Subset(ds,splits["train"][:min(t['batch_size'],4)]),batch_size=4)))
        try:
            loss_fn(model(x0.to(device)),y0.to(device)).backward()
            if device.type=="cuda":torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:raise RuntimeError("CUDA_OOM: Giảm batch size hoặc chọn CPU.")
        except RuntimeError as e:
            device_failures=("no kernel image", "invalid device function", "cuda driver", "cudnn_status_not_supported", "cuda error: initialization error")
            if device.type=="cuda" and t["device"]=="auto" and any(k in str(e).lower() for k in device_failures):
                raise GPUPreflightError(str(e)) from e
            raise
        finally:
            with torch.no_grad():
                for key,value in model.named_buffers():value.copy_(buffers[key])
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,device)
        model.zero_grad(set_to_none=True)
        amp=t["precision"]=="amp" and device.type=="cuda"
        manifest.update(effective_device=str(device),fallback_reason=reason,precision="amp-fp16" if amp else "fp32",split_hash=digest(splits),labels=labels,
                        gpu_memory_fraction=t.get("gpu_memory_fraction",.2),parameters=sum(p.numel() for p in model.parameters()),deterministic=t["deterministic"],resume_from=str(resume) if resume else None)
        dump(out/"manifest.json",manifest);emit("device",manifest)
        constructors={"SGD":torch.optim.SGD,"Adam":torch.optim.Adam,"AdamW":torch.optim.AdamW,"RMSprop":torch.optim.RMSprop,"Adagrad":torch.optim.Adagrad}
        opts=dict(lr=t["learning_rate"],weight_decay=t["weight_decay"])
        if t["optimizer"]=='SGD':opts.update(momentum=t['momentum'],dampening=t.get('dampening',0),nesterov=t.get('nesterov',False))
        if t['optimizer'] in {'Adam','AdamW'}:opts.update(betas=(t.get('adam_beta1',.9),t.get('adam_beta2',.999)),eps=t.get('optimizer_eps',1e-8),amsgrad=t.get('amsgrad',False))
        if t['optimizer']=='RMSprop':opts.update(momentum=t['momentum'],alpha=t.get('rmsprop_alpha',.99),eps=t.get('optimizer_eps',1e-8),centered=t.get('centered',False))
        if t['optimizer']=='Adagrad':opts.update(eps=t.get('optimizer_eps',1e-8))
        optimizer=constructors[t["optimizer"]](model.parameters(),**opts)
        manifest["optimizer_defaults"]={k:v for k,v in optimizer.defaults.items() if isinstance(v,(str,int,float,bool,tuple,type(None)))}
        manifest["scheduler_step_semantics"]="after each epoch"
        dump(out/"manifest.json",manifest)
        scheduler=None
        if t["scheduler"]=="StepLR":scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=t["step_size"],gamma=t["gamma"])
        if t["scheduler"]=="CosineAnnealingLR":scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t["epochs"],eta_min=t.get("min_lr",0))
        if t['scheduler']=='ExponentialLR':scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=t['gamma'])
        if t['scheduler']=='ReduceLROnPlateau':scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=t['gamma'],patience=t.get('scheduler_patience',3),min_lr=t.get('min_lr',0))
        scaler=torch.amp.GradScaler("cuda",enabled=amp)
        generator=torch.Generator().manual_seed(t["seed"])
        start_epoch=0;best=-float('inf');bad=0
        if resume:
            state=torch.load(resume,map_location=device,weights_only=False) # Only own, hash-checked local checkpoints.
            same=state.get("execution_hash",state["spec_hash"])==execution_digest(spec)
            if not same:
                previous_path=Path(resume).with_name('config.json')
                if previous_path.is_file():
                    previous=json.loads(previous_path.read_text(encoding='utf-8'))
                    same=digest(previous)==state['spec_hash'] and compatible_execution(previous,spec)
            if not same or state["dataset_hash"]!=descriptor.get("hash") or state["split_hash"]!=digest(splits):raise ValueError("RESUME_INCOMPATIBLE: Config/dataset/split đã thay đổi.")
            if state.get("backend_version",torch.__version__).split('+')[0]!=torch.__version__.split('+')[0]:raise ValueError("RESUME_VERSION: Cần cùng phiên bản PyTorch để resume. Dùng inference export hoặc train mới khi đổi phiên bản.")
            parent_best=Path(resume).with_name('best.pt')
            if not parent_best.exists():raise ValueError('BEST_CHECKPOINT_MISSING: Thiếu best checkpoint của lần chạy trước.')
            if parent_best.resolve()!=(out/'best.pt').resolve():
                import shutil
                shutil.copy2(parent_best,out/'best.pt')
            model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
            if scheduler:scheduler.load_state_dict(state["scheduler"])
            if amp and state["scaler"]:scaler.load_state_dict(state["scaler"])
            start_epoch=state["epoch"];best=state["best"];bad=state["bad"]
            generator.set_state(state["loader_rng"].cpu());torch.set_rng_state(state["torch_rng"].cpu());np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"])
            if device.type=="cuda" and state.get("cuda_rng"):torch.cuda.set_rng_state_all(state["cuda_rng"][:torch.cuda.device_count()])
        # workers=0 provides epoch-boundary replay; multi-worker seeds follow DataLoader generator.
        kwargs=dict(batch_size=t["batch_size"],num_workers=t["loader_workers"],pin_memory=device.type=="cuda")
        if t["loader_workers"]>0:kwargs["multiprocessing_context"]="spawn"
        train_loader=DataLoader(Subset(train_ds,splits["train"]),shuffle=True,generator=generator,drop_last=t.get('drop_last',False),**kwargs)
        val_loader=DataLoader(Subset(ds,splits["validation"]),shuffle=False,**kwargs)
        emit("status",dict(state="RUNNING"))
        for epoch in range(start_epoch,t["epochs"]):
            control();tick=time.monotonic()
            tr=_epoch(model,train_loader,device,loss_fn,optimizer,scaler,amp,t["grad_clip"],control,config=t)
            va=_epoch(model,val_loader,device,loss_fn,amp=amp,control=control,config=t)
            lr=optimizer.param_groups[0]["lr"]
            if scheduler:
                if t['scheduler']=='ReduceLROnPlateau':scheduler.step(va['loss'])
                else:scheduler.step()
            score=primary_score(va,t,'classification');improved=score>best
            if improved:best=score;bad=0
            else:bad+=1
            state=dict(backend_version=torch.__version__,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,scaler=scaler.state_dict(),epoch=epoch+1,best=best,bad=bad,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None,numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=generator.get_state(),
                spec_hash=digest(spec),execution_hash=execution_digest(spec),dataset_hash=descriptor.get("hash"),split_hash=digest(splits))
            atomic_torch_save(out/"last.pt",state)
            if improved:atomic_torch_save(out/"best.pt",state)
            emit("metric",dict(epoch=epoch+1,train_loss=tr["loss"],train_accuracy=tr["accuracy"],val_loss=va["loss"],val_accuracy=va["accuracy"],train_metrics=tr,validation=va,primary_metric=t.get("primary_metric") or "accuracy",lr=lr,
                elapsed=time.monotonic()-started,epoch_seconds=time.monotonic()-tick,samples_per_second=tr["samples"]/max(time.monotonic()-tick,0.001),
                ram_mb=process_ram(),vram_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else None))
            if stop_requested():raise Stopped("Đã lưu checkpoint tại cuối epoch.")
            if bad>=t["patience"]:emit("log",dict(message="Early stopping based on the primary validation metric."));break
        if not (out/"best.pt").exists() and resume:
            # Preserve the best model of the parent attempt when no new epoch improved.
            import shutil
            shutil.copy2(Path(resume).with_name("best.pt"),out/"best.pt")
        checkpoint=torch.load(out/"best.pt",map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model"]);model.eval()
        from safetensors.torch import save_file
        save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(out/"model.safetensors"))
        result=image_evaluate(model,ds,splits["validation"],labels,device,t["batch_size"],control,config=t)
        sample=ds[splits["validation"][0]][0].numpy();np.save(out/"sample.npy",sample)
    result["split"]="validation";result["split_hash"]=manifest["split_hash"];result["metric_spec"]=manifest["metric_spec"]
    emit("status",dict(state="EVALUATING"));dump(out/"evaluation.json",result)
    manifest["duration_seconds"]=time.monotonic()-started;manifest["completed"]=True;dump(out/"manifest.json",manifest)
    return result


def audit_duplicate_splits(hashes,splits):
    owners={}
    for group,indices in splits.items():
        for i in indices:
            h=hashes[i]
            if h in owners and owners[h]!=group:raise ValueError("DATA_LEAKAGE: Sample trùng nội dung giữa các tập. Loại duplicate trước khi train.")
            owners[h]=group


def process_ram():
    try:
        import psutil
        return psutil.Process().memory_info().rss/1024**2
    except ImportError:return None


def image_evaluate(model,ds,indices,labels,device,batch_size,control=lambda:None,config=None):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader,Subset
    model.eval();ys=[];scores=[]
    with torch.inference_mode():
        for x,y in DataLoader(Subset(ds,indices),batch_size=batch_size):
            control();scores.extend(model(x.to(device)).softmax(1).cpu().tolist());ys.extend(y.tolist())
    return add_custom(classification_result(ys,np.asarray(scores).argmax(1),labels,scores,indices),config or {},np.asarray(ys),np.asarray(scores).argmax(1),np.asarray(scores))


def evaluate_holdout(spec,descriptor,out):
    verify_dataset(descriptor)
    import numpy as np
    out=Path(out);manifest=json.loads((out/"manifest.json").read_text(encoding='utf-8'));splits=json.loads((out/"splits.json").read_text(encoding='utf-8'))
    if spec["backend"]=="sklearn":
        import joblib
        frame,y,labels=tabular_data(spec,descriptor);model=joblib.load(out/"pipeline.joblib");ix=splits["test"]
        predicted=model.predict(frame.iloc[ix]);scores=None if spec.get('task')=='regression' else model.predict_proba(frame.iloc[ix])
        result=regression_result(y[ix],predicted,ix) if spec.get('task')=='regression' else classification_result(y[ix],predicted,labels,scores,ix)
        result=add_custom(result,spec['training'],y[ix],predicted,scores)
    else:
        import torch
        from safetensors.torch import load_file
        ds,_,labels=image_data(spec,descriptor);model=build_model(spec);model.load_state_dict(load_file(str(out/"model.safetensors")))
        result=image_evaluate(model,ds,splits["test"],labels,torch.device("cpu"),spec["training"]["batch_size"],config=spec["training"])
    result.update(split="test",split_hash=manifest["split_hash"],metric_spec=manifest["metric_spec"])
    dump(out/"test-evaluation.json",result);return result


def inference(out, sample=None):
    out=Path(out);spec=json.loads((out/"config.json").read_text(encoding='utf-8'));manifest=json.loads((out/"manifest.json").read_text(encoding='utf-8'))
    labels=manifest["labels"]
    if spec["backend"]=="sklearn":
        import joblib
        import pandas as pd
        rows=json.loads((out/"sample.json").read_text(encoding='utf-8')) if sample is None else sample
        model=joblib.load(out/"pipeline.joblib");frame=pd.DataFrame(rows)[manifest["features"]]
        if spec.get('task')=='regression':return [dict(value=float(value)) for value in model.predict(frame)]
        scores=model.predict_proba(frame)
    else:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        x=np.load(out/"sample.npy",allow_pickle=False) if sample is None else np.asarray(sample,dtype=np.float32)
        if x.ndim==3:x=x[None]
        model=build_model(spec);model.load_state_dict(load_file(str(out/"model.safetensors")));model.eval()
        with torch.inference_mode():scores=model(torch.from_numpy(x).float()).softmax(1).numpy()
    return [dict(label=labels[int(row.argmax())],scores=[float(x) for x in row]) for row in scores]


def cross_validate_training(spec,descriptor,folds=5,control=lambda:None,deadline=None):
    """Refit the entire pipeline per fold, exclusively inside the original train split."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold,KFold,GroupKFold
    from sklearn.metrics import accuracy_score,f1_score
    verify_dataset(descriptor)
    frame,y,_=tabular_data(spec,descriptor)
    regression=spec.get('task')=='regression'
    original=resolved_splits(y,{**spec['data'],'stratify':False} if regression else spec['data'],spec['training']['seed'],descriptor)
    source=np.asarray(original['train']);x=frame.iloc[source];target=y[source]
    if not regression and min(np.bincount(target))<folds:raise ValueError('CV_SAMPLES: Mỗi lớp trong train cần ít nhất số mẫu bằng số fold.')
    groups=x.astype(str).apply(lambda row:digest(row.tolist()),axis=1).to_numpy()
    grouped=len(set(groups))<len(groups)
    splitter=(StratifiedGroupKFold if grouped else StratifiedKFold)(n_splits=folds,shuffle=True,random_state=spec['training']['seed'])
    if regression:splitter=GroupKFold(n_splits=folds) if grouped else KFold(n_splits=folds,shuffle=True,random_state=spec['training']['seed'])
    plan=splitter.split(x,target,groups) if grouped else splitter.split(x,target)
    reports=[]
    for fold,(train_ix,val_ix) in enumerate(plan,1):
        control()
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('STUDY_BUDGET')
        if not regression and len(set(target[train_ix]))!=len(set(target)):raise ValueError('CV_GROUPS: Chưa đủ nhóm mẫu độc lập trong mỗi lớp; giảm số fold.')
        model=sklearn_model(spec,frame)
        model.fit(x.iloc[train_ix],target[train_ix]);pred=model.predict(x.iloc[val_ix])
        if regression:
            metrics=regression_result(target[val_ix],pred);reports.append(dict(fold=fold,r2=metrics['r2'],neg_rmse=-metrics['rmse'],train_count=len(train_ix),validation_count=len(val_ix)));continue
        reports.append(dict(fold=fold,accuracy=float(accuracy_score(target[val_ix],pred)),macro_f1=float(f1_score(target[val_ix],pred,average='macro',zero_division=0)),train_count=len(train_ix),validation_count=len(val_ix),train_indices_hash=digest(source[train_ix].tolist()),validation_indices_hash=digest(source[val_ix].tolist())))
    if regression:return dict(folds=reports,r2_mean=float(np.mean([r['r2'] for r in reports])),neg_rmse_mean=float(np.mean([r['neg_rmse'] for r in reports])),source_train_indices=source.tolist(),policy='Train-only refit per fold; regression',seed=spec['training']['seed'])
    return dict(folds=reports,accuracy_mean=float(np.mean([r['accuracy'] for r in reports])),accuracy_std=float(np.std([r['accuracy'] for r in reports])),macro_f1_mean=float(np.mean([r['macro_f1'] for r in reports])),source_train_indices=source.tolist(),policy='Entire pipeline refit per fold; original train only; exact duplicate feature rows grouped' if grouped else 'Entire pipeline refit per fold; original train only',seed=spec['training']['seed'])


def regression_result(actual,predicted,sample_ids=None):
    import numpy as np
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    if len(actual)<2:raise ValueError('REGRESSION_SAMPLES: Cần ít nhất hai mẫu ở tập đánh giá để tính R².')
    ids=list(range(len(actual))) if sample_ids is None else sample_ids
    return dict(mse=float(mean_squared_error(actual,predicted)),mape=float(np.mean(np.abs((np.asarray(actual)-np.asarray(predicted))/np.maximum(np.abs(actual),1e-8)))),task='regression',rmse=float(np.sqrt(mean_squared_error(actual,predicted))),mae=float(mean_absolute_error(actual,predicted)),r2=float(r2_score(actual,predicted)),predictions=[dict(index=int(ids[i]),actual=float(a),predicted=float(b),residual=float(a-b)) for i,(a,b) in enumerate(zip(actual,predicted))])
