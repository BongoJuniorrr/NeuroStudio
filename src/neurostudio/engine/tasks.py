"""Portable supervised task adapters. This file also runs inside exported packages.

Imports of optional ML libraries are lazy. No API, storage or application dependency.
Dataset manifests define explicit train/validation/test splits; nothing fits on test.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
import wave


try:
    from .objectives import classification_metrics,add_custom,primary_score,loss_value,text_scores
except ImportError:
    from objectives import classification_metrics,add_custom,primary_score,loss_value,text_scores


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8');os.replace(tmp,path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def child(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('DATASET_PATH: File thiếu hoặc nằm ngoài dataset.')
    return path


def load_dataset(path,task=None,control=lambda:None):
    path=Path(path);root=path.parent;data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('task') not in ['image_classification','segmentation','detection','text_classification','audio_classification','image_reconstruction','text_generation','time_series']:raise ValueError('DATASET_TASK: Loại bài toán chưa được hỗ trợ.')
    if data.get('format')!='neurostudio.task-dataset.v1':raise ValueError('DATASET_FORMAT: Cần manifest neurostudio.task-dataset.v1.')
    if task and data['task']!=task:raise ValueError('DATASET_TASK: Dataset và bài toán không tương thích.')
    labels=data.get('labels',[])
    if not isinstance(labels,list) or not all(isinstance(label,str) and label.strip() for label in labels) or not (1 if data['task']=='time_series' else 2)<=len(labels)<=1000 or len(set(labels))!=len(labels):raise ValueError('DATASET_LABELS: Cần 2–1000 nhãn duy nhất.')
    seen={};counts={};hashes={}
    for split in ['test','validation','train']:
        rows=data.get('splits',{}).get(split,[])
        if not rows or len(rows)>1000000:raise ValueError('DATASET_SPLIT: Cần train/validation/test riêng, tối đa 1 triệu mẫu mỗi tập.')
        counts[split]=len(rows)
        for row in rows:
            control()
            label=row.get('label')
            if type(label) is not int or not 0<=label<len(labels):raise ValueError('DATASET_LABEL: Nhãn phải là chỉ số lớp hợp lệ.')
            if data['task']=='time_series':
                values=row.get('values',[]);target=row.get('target')
                if not isinstance(values,list) or not 2<=len(values)<=256 or any(type(v) not in [int,float] or not math.isfinite(v) for v in values) or type(target) not in [int,float] or not math.isfinite(target):raise ValueError('SERIES_VALUES: Need 2-256 finite input values and one finite target.')
                identity=hashlib.sha256(json.dumps([values,target,row.get('source_start'),row.get('source_end')]).encode()).hexdigest()
            elif data['task'] in ['text_classification','text_generation']:
                text=row.get('text')
                if not isinstance(text,str) or not text.strip() or len(text)>100000:raise ValueError('DATASET_TEXT: Văn bản trống hoặc quá dài.')
                if data['task']=='text_generation' and 'target_text' in row and (not isinstance(row['target_text'],str) or not row['target_text'].strip() or len(row['target_text'])>100000):raise ValueError('DATASET_TARGET_TEXT: Target text must be non-empty and bounded.')
                identity=hashlib.sha256(' '.join(text.split()).lower().encode()).hexdigest()
            else:
                key='audio' if data['task']=='audio_classification' else 'image'
                file=child(root,row[key]);relative=str(file.relative_to(root.resolve()))
                hashes.setdefault(relative,digest(file));identity=hashes[relative]
                if file.stat().st_size>30*1024**2:raise ValueError('DATASET_FILE_SIZE: Mỗi file tối đa 30 MB.')
                if data['task']=='segmentation':
                    mask=child(root,row['mask']);hashes[str(mask.relative_to(root.resolve()))]=digest(mask)
                if data['task']=='detection':
                    if not isinstance(row.get('boxes'),list):raise ValueError('DATASET_BOXES: Thiếu bbox xyxy.')
                    box_labels=row.get('box_labels',[label]*len(row['boxes']))
                    if not isinstance(box_labels,list) or len(box_labels)!=len(row['boxes']) or any(type(v) is not int or not 1<=v<len(labels) for v in box_labels):raise ValueError('DATASET_BOX_LABEL: Mỗi bbox cần một label hợp lệ ≥ 1.')
                    if row['boxes'] and label==0 and 'box_labels' not in row:raise ValueError('DATASET_BOX_LABEL: Đối tượng phải có label ≥ 1; 0 là background.')
                    for box in row['boxes']:
                        if len(box)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in box) or not (0<=box[0]<box[2] and 0<=box[1]<box[3]):raise ValueError('DATASET_BOX: bbox phải là x1,y1,x2,y2 hữu hạn, kích thước dương.')
            if identity in seen and seen[identity]!=split:raise ValueError('DATASET_LEAKAGE: Mẫu giống nhau nằm trong nhiều tập.')
            seen[identity]=split
    data['root']=str(root.resolve());data['manifest_hash']=digest(path);data['file_hashes']=hashes;data['counts']=counts
    return data


def tokens(text):return re.findall(r"\w+|[^\w\s]",text.lower(),flags=re.UNICODE)


def vocabulary(rows,maximum):
    from collections import Counter
    count=Counter(token for row in rows for text in [row['text'],row.get('target_text','')] for token in tokens(text))
    return {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3,**{word:i+4 for i,(word,_) in enumerate(count.most_common(maximum-4)) if word not in ['<pad>','<unk>','<bos>','<eos>']}}


def waveform(path):
    import numpy as np
    import torch
    with wave.open(str(path),'rb') as f:
        if f.getcomptype()!='NONE' or f.getsampwidth()!=2 or f.getnchannels() not in [1,2]:raise ValueError('AUDIO_FORMAT: Cần WAV PCM 16-bit mono/stereo.')
        rate=f.getframerate();channels=f.getnchannels()
        if not 4000<=rate<=96000 or not 0<f.getnframes()/rate<=30:raise ValueError('AUDIO_LENGTH: WAV tối đa 30 giây; sample rate 4–96 kHz.')
        raw=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype('float32')/32768
    signal=torch.from_numpy(raw.reshape(-1,channels).mean(1)).view(1,1,-1)
    signal=torch.nn.functional.interpolate(signal,size=max(1,round(signal.numel()*8000/rate)),mode='linear',align_corners=False).flatten()
    signal=signal[:8000];signal=torch.nn.functional.pad(signal,(0,max(0,8000-len(signal))))
    spec=torch.stft(signal,n_fft=256,hop_length=80,window=torch.hann_window(256),return_complex=True).abs().log1p()
    return spec.unsqueeze(0)


def get_item(row,data,config,vocab=None,tokenizer=None):
    import numpy as np
    import torch
    task=config['task'];root=Path(data['root']);size=config['image_size']
    if task=='time_series':
        scale=config['series_scaler'];x=torch.tensor(row['values'],dtype=torch.float32);return ((x-scale['mean'])/scale['std']).unsqueeze(-1),torch.tensor(row['target'],dtype=torch.float32)
    if task=='text_generation':
        if tokenizer:
            encoded=tokenizer(row['text'],truncation=True,max_length=config['max_length']+1,add_special_tokens=True)
            ids=encoded['input_ids']
            if len(ids)<2:ids=ids+[tokenizer.eos_token_id]
            ids=ids[:config['max_length']+1];length=len(ids)-1
            x={'input_ids':torch.tensor(ids[:-1]+[tokenizer.pad_token_id]*(config['max_length']-length)),'attention_mask':torch.tensor([1]*length+[0]*(config['max_length']-length))}
            y=torch.tensor(ids[1:]+[-100]*(config['max_length']-length))
            return x,y
        maximum=config['max_length']
        def encode(text):return [vocab.get(t,1) for t in tokens(text)][:maximum-1]+[3]
        ids=encode(row['text'])
        if config['model']=='seq2seq_gru':
            target=encode(row.get('target_text',row['text']));teacher=[2]+target[:-1]
            return {'source':torch.tensor(ids+[0]*(maximum-len(ids))),'teacher':torch.tensor(teacher+[0]*(maximum-len(teacher)))},torch.tensor(target+[0]*(maximum-len(target)))
        shifted=[2]+ids
        return torch.tensor(shifted[:-1]+[0]*(maximum-len(ids))),torch.tensor(ids+[0]*(maximum-len(ids)))
    if task=='text_classification':
        if tokenizer:
            inputs=tokenizer(row['text'],padding='max_length',truncation=True,max_length=config['max_length'],return_tensors='pt')
            x={k:v[0] for k,v in inputs.items()}
        else:
            ids=[vocab.get(t,1) for t in tokens(row['text'])][:config['max_length']] or [1]
            x=torch.tensor(ids+[0]*(config['max_length']-len(ids)),dtype=torch.long)
    elif task=='audio_classification':x=waveform(child(root,row['audio']))
    else:
        from PIL import Image,ImageOps
        with Image.open(child(root,row['image'])) as raw:
            image=(raw if task in ['segmentation','detection'] or config.get('annotation_coordinates') else ImageOps.exif_transpose(raw)).convert('RGB');width,height=image.size
            if width*height>50_000_000:raise ValueError('IMAGE_SIZE: Ảnh quá lớn.')
            x=torch.from_numpy(np.array(image.resize((size,size))).copy()).permute(2,0,1).float()/255
        if task=='segmentation':
            with Image.open(child(root,row['mask'])) as mask:
                if mask.size!=(width,height):raise ValueError('MASK_SIZE: Mask và ảnh phải cùng kích thước.')
                array=np.array(mask.resize((size,size),Image.Resampling.NEAREST))
                if config.get('segmentation_mode','binary')=='multiclass':
                    if array.ndim!=2 or int(array.max())>=len(data['labels']):raise ValueError('MASK_CLASSES: Mask indexed grayscale cần giá trị 0..classes-1, không phải RGB hoặc 255.')
                    y=torch.from_numpy(array.astype('int64'))
                else:
                    foreground=(array[:,:,:3]>0).any(2) if array.ndim==3 else array>0
                    y=torch.from_numpy(foreground.astype('float32')).unsqueeze(0)
            return x,y
        if task=='detection':
            boxes=torch.tensor(row['boxes'],dtype=torch.float32).reshape(-1,4)
            if (boxes[:,2]>width).any() or (boxes[:,3]>height).any():raise ValueError('BBOX_BOUNDS: Bbox vượt khỏi ảnh.')
            boxes=boxes*torch.tensor([size/width,size/height,size/width,size/height])
            return x,{'boxes':boxes,'labels':torch.tensor(row.get('box_labels',[row['label']]*len(boxes)),dtype=torch.int64)}
        if task=='image_reconstruction':return x,x.clone()
        x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return x,torch.tensor(row['label'],dtype=torch.long)


def make_model(config,classes,folder,vocab=None,initialize=False):
    import torch
    from torch import nn
    name=config['model'];hidden=config['hidden_size'];folder=Path(folder);tokenizer=None
    if name in ['gru_forecaster','linear_forecaster']:
        class Forecaster(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=nn.GRU(1,hidden,batch_first=True) if name=='gru_forecaster' else None;self.head=nn.Linear(hidden if self.encoder else config['series_scaler']['window'],1)
            def forward(self,x):
                values=self.encoder(x)[0][:,-1] if self.encoder else x.flatten(1)
                return self.head(values).squeeze(-1)
        model=Forecaster()
    elif name in ['conv_autoencoder','vae']:
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__();size=config['image_size'];self.size=size;self.variational=name=='vae';self.encoder=nn.Sequential(nn.Conv2d(3,16,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),nn.Flatten());self.mean=nn.Linear(512,config['latent_size']);self.log_variance=nn.Linear(512,config['latent_size']);self.decoder=nn.Sequential(nn.Linear(config['latent_size'],512),nn.ReLU(),nn.Unflatten(1,(32,4,4)),nn.ConvTranspose2d(32,16,4,stride=2,padding=1),nn.ReLU(),nn.ConvTranspose2d(16,3,4,stride=2,padding=1))
            def decode(self,z):return nn.functional.interpolate(self.decoder(z),size=(self.size,self.size),mode='bilinear',align_corners=False).sigmoid()
            def forward(self,x):
                h=self.encoder(x);mean=self.mean(h);logvar=self.log_variance(h).clamp(-10,10)
                z=mean+torch.randn_like(mean)*torch.exp(.5*logvar) if self.variational and self.training else mean
                return self.decode(z),mean,logvar
        model=Autoencoder()
    elif name in ['causal_gru','seq2seq_gru']:
        class LanguageModel(nn.Module):
            def __init__(self):
                super().__init__();self.embedding=nn.Embedding(len(vocab),hidden,padding_idx=0);self.decoder=nn.GRU(hidden,hidden,batch_first=True);self.encoder=nn.GRU(hidden,hidden,batch_first=True) if name=='seq2seq_gru' else None;self.head=nn.Linear(hidden,len(vocab))
            def forward(self,x):
                if isinstance(x,dict):
                    source=x['source'];lengths=source.ne(0).sum(1).cpu();packed=nn.utils.rnn.pack_padded_sequence(self.embedding(source),lengths,batch_first=True,enforce_sorted=False);_,h=self.encoder(packed);output,_=self.decoder(self.embedding(x['teacher']),h)
                else:output,_=self.decoder(self.embedding(x))
                return self.head(output)
        model=LanguageModel()
    elif config['task']=='image_classification':
        from torchvision.models import get_model,get_model_weights
        weights=get_model_weights(name).DEFAULT if initialize and config['pretrained'] and not config.get('pretrained_path') else None
        model=get_model(name,weights=weights)
        if initialize and config.get('pretrained_path'):
            path=Path(config['pretrained_path']).expanduser()
            if path.suffix=='.safetensors':
                from safetensors.torch import load_file
                state=load_file(str(path))
            else:state=torch.load(path,map_location='cpu',weights_only=True)
            state=state.get('state_dict',state);expected=model.state_dict();head_keys=('fc.','classifier.','heads.','head.')
            unexpected=set(state)-set(expected)
            missing=set(expected)-set(state)
            if unexpected or any(not k.startswith(head_keys) for k in missing):raise ValueError('PRETRAINED_KEYS: State dictionary does not match the selected architecture.')
            incompatible=[k for k,v in state.items() if v.shape!=expected[k].shape]
            if any(not k.startswith(head_keys) for k in incompatible):raise ValueError('PRETRAINED_SHAPES: Backbone tensor shapes do not match.')
            model.load_state_dict({k:v for k,v in state.items() if k not in incompatible},strict=False)
            save_json(folder/'pretrained-source.json',dict(path=str(path),sha256=digest(path),architecture=name))
        if config['freeze_backbone']:
            for p in model.parameters():p.requires_grad=False
        if hasattr(model,'fc'):model.fc=nn.Linear(model.fc.in_features,classes)
        elif hasattr(model,'classifier'):
            if isinstance(model.classifier,nn.Linear):model.classifier=nn.Linear(model.classifier.in_features,classes)
            else:
                last=max(i for i,m in enumerate(model.classifier) if isinstance(m,nn.Linear));model.classifier[last]=nn.Linear(model.classifier[last].in_features,classes)
        elif hasattr(model,'heads'):model.heads.head=nn.Linear(model.heads.head.in_features,classes)
        elif hasattr(model,'head'):model.head=nn.Linear(model.head.in_features,classes)
        else:raise ValueError('MODEL_HEAD: Unsupported classification head.')
        if name=='vit_b_16' and config['image_size']!=224:raise ValueError('MODEL_IMAGE_SIZE: ViT-B/16 requires image size 224.')
    elif name=='resnet18':
        from torchvision.models import resnet18,ResNet18_Weights
        model=resnet18(weights=ResNet18_Weights.DEFAULT if initialize and config['pretrained'] else None)
        if config['freeze_backbone']:
            for p in model.parameters():p.requires_grad=False
        model.fc=nn.Linear(model.fc.in_features,classes)
    elif name=='unet':
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
        class UNet(nn.Module):
            def __init__(self):
                super().__init__();self.enc1=block(3,16);self.enc2=block(16,32);self.middle=block(32,64);self.dec2=block(96,32);self.dec1=block(48,16);self.head=nn.Conv2d(16,classes if config.get('segmentation_mode')=='multiclass' else 1,1)
            def forward(self,x):
                a=self.enc1(x);b=self.enc2(nn.functional.max_pool2d(a,2));c=self.middle(nn.functional.max_pool2d(b,2))
                d=self.dec2(torch.cat([nn.functional.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
                return self.head(self.dec1(torch.cat([nn.functional.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))
        model=UNet()
    elif name=='fasterrcnn_mobilenet':
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=None,weights_backbone=None,num_classes=classes,min_size=config['image_size'],max_size=config['image_size'],rpn_pre_nms_top_n_train=100,rpn_post_nms_top_n_train=50,rpn_pre_nms_top_n_test=100,rpn_post_nms_top_n_test=50,box_detections_per_img=30)
        # Batch size one is intentional for detection. Frozen BN avoids invalid 1x1 batch stats.
        from torchvision.ops.misc import FrozenBatchNorm2d
        def freeze_bn(module):
            for key,layer in list(module.named_children()):
                if isinstance(layer,nn.BatchNorm2d):
                    frozen=FrozenBatchNorm2d(layer.num_features,eps=layer.eps)
                    frozen.load_state_dict({k:v for k,v in layer.state_dict().items() if k!='num_batches_tracked'});setattr(module,key,frozen)
                else:freeze_bn(layer)
        freeze_bn(model)
    elif name in ['bigru','transformer']:
        class TextModel(nn.Module):
            def __init__(self):
                super().__init__();self.embedding=nn.Embedding(len(vocab),hidden,padding_idx=0)
                if name=='bigru':self.encoder=nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True)
                else:
                    self.position=nn.Embedding(config['max_length'],hidden)
                    self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.1,batch_first=True),num_layers=2,enable_nested_tensor=False)
                self.head=nn.Linear(hidden,classes)
            def forward(self,x):
                mask=x.ne(0);embedded=self.embedding(x)
                if name=='bigru':
                    lengths=mask.sum(1).cpu();packed=nn.utils.rnn.pack_padded_sequence(embedded,lengths,batch_first=True,enforce_sorted=False)
                    _,h=self.encoder(packed);pooled=torch.cat([h[-2],h[-1]],1)
                else:
                    embedded=embedded+self.position(torch.arange(x.shape[1],device=x.device))[None]
                    output=self.encoder(embedded,src_key_padding_mask=~mask);pooled=(output*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
                return self.head(pooled)
        model=TextModel()
    elif name in ['bert_tiny','bert_tiny_lora','bert_base','distilbert','deberta','hf_classifier','gpt2','distilgpt2','hf_causal']:
        try:from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoModelForCausalLM,AutoConfig
        except ImportError as e:raise RuntimeError('Cần môi trường NLP với transformers; xem hướng dẫn cài đặt trong máy & cài đặt.') from e
        if initialize:
            sources={'bert_tiny':'prajjwal1/bert-tiny','bert_tiny_lora':'prajjwal1/bert-tiny','bert_base':'google-bert/bert-base-uncased','distilbert':'distilbert/distilbert-base-uncased','deberta':'microsoft/deberta-base','gpt2':'openai-community/gpt2','distilgpt2':'distilbert/distilgpt2'}
            source=config.get('hf_model') or sources.get(name);revision=config.get('model_revision') or config.get('hf_revision','main')
            if not source:raise ValueError('HF_MODEL: Enter a Hugging Face model ID or local model folder.')
            tokenizer=AutoTokenizer.from_pretrained(source,revision=revision)
            causal=config['task']=='text_generation'
            if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
            model=AutoModelForCausalLM.from_pretrained(source,revision=revision) if causal else AutoModelForSequenceClassification.from_pretrained(source,revision=revision,num_labels=classes,ignore_mismatched_sizes=True)
            model.config.pad_token_id=tokenizer.pad_token_id
            tokenizer.save_pretrained(folder/'tokenizer');model.config.save_pretrained(folder/'bert-config')
            save_json(folder/'pretrained-source.json',dict(model=source,revision=getattr(model.config,'_commit_hash',revision)))
        else:
            tokenizer=AutoTokenizer.from_pretrained(folder/'tokenizer',local_files_only=True)
            model=(AutoModelForCausalLM if config['task']=='text_generation' else AutoModelForSequenceClassification).from_config(AutoConfig.from_pretrained(folder/'bert-config',local_files_only=True))
        if config['freeze_backbone']:
            for p in model.base_model.parameters():p.requires_grad=False
            if config['task']=='text_generation':
                for p in model.get_output_embeddings().parameters():p.requires_grad=True
        if name=='bert_tiny_lora':
            try:from peft import LoraConfig,get_peft_model,TaskType
            except ImportError as e:raise RuntimeError('LoRA cần optional package peft.') from e
            model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_CLS,r=config['lora_rank'],lora_alpha=config['lora_rank']*2,lora_dropout=.05,target_modules=['query','value']))
    elif name=='spectrogram_cnn':
        model=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,classes))
    else:raise ValueError('UNKNOWN_MODEL')
    return model,tokenizer


def to_device(value,device):
    if isinstance(value,dict):return {key:to_device(v,device) for key,v in value.items()}
    if isinstance(value,list):return [to_device(v,device) for v in value]
    return value.to(device)


def batches(rows,data,config,vocab,tokenizer,shuffle=False):
    import torch
    order=list(range(len(rows)))
    if shuffle:random.shuffle(order)
    for start in range(0,len(order),config['batch_size']):
        pairs=[get_item(rows[i],data,config,vocab,tokenizer) for i in order[start:start+config['batch_size']]]
        x,y=zip(*pairs)
        if config['task']=='detection':yield list(x),list(y)
        else:
            x={k:torch.stack([v[k] for v in x]) for k in x[0]} if isinstance(x[0],dict) else torch.stack(x)
            yield x,torch.stack(y)


def forward(model,x):
    if isinstance(x,dict) and 'source' not in x:
        import inspect
        if not hasattr(model,'_studio_input_names'):
            signature=inspect.signature(model.forward)
            model._studio_input_names=None if any(p.kind==inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()) else set(signature.parameters)
        if model._studio_input_names is not None:x={k:v for k,v in x.items() if k in model._studio_input_names}
        return model(**x).logits
    return model(x)


def box_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def one_class_detection_metrics(predictions,targets):
    """One-class interpolated all-points AP at IoU .5; includes unmatched GTs."""
    ranked=[];total=sum(len(t['boxes']) for t in targets)
    for image,pred in enumerate(predictions):
        ranked.extend((float(score),image,box) for score,box in zip(pred['scores'],pred['boxes']))
    ranked.sort(key=lambda item:-item[0]);matched=set();tp=0;fp=0;precisions=[];recalls=[]
    for score,image,box in ranked:
        gt=targets[image]['boxes'];options=[(box_iou(box,g),i) for i,g in enumerate(gt) if (image,i) not in matched]
        overlap,index=max(options,default=(0,-1))
        if overlap>=.5:tp+=1;matched.add((image,index))
        else:fp+=1
        precisions.append(tp/(tp+fp));recalls.append(tp/max(1,total))
    precision=[0]+precisions+[0];recall=[0]+recalls+[1]
    for i in range(len(precision)-2,-1,-1):precision[i]=max(precision[i],precision[i+1])
    ap=sum((recall[i]-recall[i-1])*precision[i] for i in range(1,len(recall)) if recall[i]!=recall[i-1])
    return dict(ap50=ap,recall50=tp/max(1,total),ground_truth_objects=total,predicted_objects=len(ranked),images=len(targets))



def detection_metrics(predictions,targets):
    classes=sorted({int(label) for row in targets for label in row.get('labels',[1]*len(row['boxes']))})
    # Legacy one-class fixtures without labels retain their original semantics.
    if not classes:classes=[1]
    per_class={}
    for label in classes:
        filtered_predictions=[];filtered_targets=[]
        for pred,target in zip(predictions,targets):
            labels=pred.get('labels',[1]*len(pred['boxes']))
            filtered_predictions.append(dict(boxes=[b for b,l in zip(pred['boxes'],labels) if l==label],scores=[s for s,l in zip(pred['scores'],labels) if l==label]))
            labels=target.get('labels',[1]*len(target['boxes']))
            filtered_targets.append(dict(boxes=[b for b,l in zip(target['boxes'],labels) if l==label]))
        per_class[str(label)]=one_class_detection_metrics(filtered_predictions,filtered_targets)
    return dict(ap50=sum(m['ap50'] for m in per_class.values())/len(per_class),recall50=sum(m['recall50'] for m in per_class.values())/len(per_class),ground_truth_objects=sum(len(t['boxes']) for t in targets),predicted_objects=sum(len(p['boxes']) for p in predictions),images=len(targets),per_class=per_class,metric='Macro AP at IoU .50 over classes present in ground truth; not COCO mAP.')


def reconstruction_loss(output,target,config):
    import torch
    reconstruction,mean,logvar=output
    mse=loss_value(reconstruction,target,config,'image_reconstruction')
    kl=-.5*(1+logvar-mean.square()-logvar.exp()).mean()
    return mse+config['kl_weight']*kl if config['model']=='vae' else mse


def segmentation_loss(logits,target,config):
    import torch
    if config.get('loss')=='custom':return loss_value(logits,target,config,'segmentation')
    if config.get('segmentation_mode')=='multiclass':
        if config.get('loss') not in [None,'auto','cross_entropy','focal']:raise ValueError('LOSS_MASK: Multi-class masks require cross_entropy/focal/custom loss.')
        if config.get('loss')=='focal':
            ce=torch.nn.functional.cross_entropy(logits,target,reduction='none');return (((1-torch.exp(-ce))**config.get('focal_gamma',2))*ce).mean()
        return torch.nn.functional.cross_entropy(logits,target)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    kind=config.get('loss','auto');kind=config.get('segmentation_loss','bce') if kind=='auto' else kind
    if kind=='cross_entropy':raise ValueError('LOSS_MASK: Binary masks require BCE/Dice/Jaccard/Focal or custom loss.')
    if kind=='bce':return bce
    if kind=='focal':
        ce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none');return (((1-torch.exp(-ce))**config.get('focal_gamma',2))*ce).mean()
    probability=logits.sigmoid();dimensions=tuple(range(1,probability.ndim))
    dice=(2*(probability*target).sum(dimensions)+1)/(probability.sum(dimensions)+target.sum(dimensions)+1)
    if kind=='dice':return (1-dice).mean()
    if kind=='jaccard':
        overlap=(probability*target).sum(dimensions);return (1-(overlap+1)/(probability.sum(dimensions)+target.sum(dimensions)-overlap+1)).mean()
    weight=config.get('dice_weight',.5)
    return (1-weight)*bce+weight*(1-dice).mean()


def image_preview(tensor):
    import base64,io
    import numpy as np
    from PIL import Image
    raw=(tensor.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
    image=Image.fromarray(raw);image.thumbnail((256,256));buffer=io.BytesIO();image.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def evaluate(model,rows,data,config,vocab,tokenizer,device,control=lambda:None):
    import torch
    import numpy as np
    task=config['task'];multiclass=config.get('segmentation_mode')=='multiclass';histogram=np.zeros((len(data['labels']),len(data['labels'])),dtype=np.int64);model.eval();truth=[];predictions=[];loss_sum=0.;count=0;intersection=0.;union=0.;predicted_area=0.;true_area=0.;preview=[];reconstruction_mae=0.;generation_objective_sum=0.
    with torch.inference_mode():
        for x,y in batches(rows,data,config,vocab,tokenizer):
            control();x=to_device(x,device);y=to_device(y,device);out=forward(model,x)
            if task=='detection':
                for image,pred,target in zip(x,out,y):
                    predictions.append({key:pred[key].cpu().tolist() for key in ['boxes','scores','labels']});truth.append({key:target[key].cpu().tolist() for key in ['boxes','labels']})
                    if len(preview)<4:preview.append(image_preview(image))
            elif task=='time_series':
                truth.extend(y.cpu().tolist());predictions.extend(out.cpu().tolist());loss_sum+=loss_value(out,y,config,task).item()*len(y);count+=len(y)
            elif task=='image_reconstruction':
                reconstruction=out[0]
                if config.get('metric_code'):truth.extend(y.flatten().cpu().tolist());predictions.extend(reconstruction.flatten().cpu().tolist())
                reconstruction_mae+=torch.nn.functional.l1_loss(reconstruction,y).item()*len(y);loss_sum+=torch.nn.functional.mse_loss(reconstruction,y).item()*len(y);count+=len(y)
                if len(preview)<4:preview.extend(dict(image=image_preview(a),reconstruction=image_preview(b)) for a,b in zip(x[:2],reconstruction[:2]))
            elif task=='text_generation':
                valid=y.ne(-100) if tokenizer else y.ne(0);n=int(valid.sum());generation_objective_sum+=loss_value(out,y,config,'text_generation').item()*n;loss_sum+=torch.nn.functional.cross_entropy(out.flatten(0,1),y.flatten(),ignore_index=-100 if tokenizer else 0,reduction='sum').item();count+=n
                intersection+=((out.argmax(-1)==y)&valid).sum().item()
                if config.get('metric_code'):truth.extend(y[valid].cpu().tolist());predictions.extend(out.argmax(-1)[valid].cpu().tolist())
            elif task=='segmentation':
                loss=segmentation_loss(out,y,config)
                if multiclass:
                    mask=out.argmax(1);actual=y
                    if config.get('metric_code'):truth.extend(actual.flatten().cpu().tolist());predictions.extend(mask.flatten().cpu().tolist())
                    n=len(data['labels']);histogram+=np.bincount((actual*n+mask).flatten().cpu().numpy(),minlength=n*n).reshape(n,n)
                    loss_sum+=loss.item()*len(y);count+=len(y)
                    if len(preview)<4:preview.extend(dict(image=image_preview(image),mask=m.to(torch.int64).cpu().tolist(),truth=t.to(torch.int64).cpu().tolist()) for image,m,t in zip(x[:2],mask[:2],actual[:2]))
                    continue
                mask=out.sigmoid()>=.5;actual=y.bool()
                if config.get('metric_code'):truth.extend(actual.flatten().cpu().tolist());predictions.extend(mask.flatten().cpu().tolist())
                intersection+=(mask&actual).sum().item();union+=(mask|actual).sum().item();predicted_area+=mask.sum().item();true_area+=actual.sum().item()
                loss_sum+=loss.item()*len(y);count+=len(y)
                if len(preview)<4:
                    preview.extend(dict(image=image_preview(image),mask=m[0].to(torch.uint8).cpu().tolist(),truth=t[0].to(torch.uint8).cpu().tolist()) for image,m,t in zip(x[:2],mask[:2],actual[:2]))
            else:
                loss=loss_value(out,y,config,task);loss_sum+=loss.item()*len(y);count+=len(y)
                truth.extend(y.cpu().tolist());predictions.extend(out.softmax(1).cpu().tolist())
    if task=='time_series':
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        metrics=dict(mse=float(mean_squared_error(truth,predictions)),mae=float(mean_absolute_error(truth,predictions)),r2=float(r2_score(truth,predictions)),samples=count);metrics['rmse']=math.sqrt(metrics['mse'])
        return add_custom(metrics,config,np.asarray(truth),np.asarray(predictions)),[dict(truth=a,predicted=b) for a,b in zip(truth,predictions)][:200]
    if task=='image_reconstruction':
        mse=loss_sum/max(1,count);metrics=dict(mse=mse,psnr=-10*math.log10(max(mse,1e-12)),samples=count)
        # MAE is accumulated separately to retain its distinct meaning.
        metrics['mae']=reconstruction_mae/max(1,count)
        return add_custom(metrics,config,np.asarray(truth),np.asarray(predictions)),preview[:4]
    if task=='text_generation':
        loss=loss_sum/max(1,count);metrics=dict(loss=generation_objective_sum/max(1,count),perplexity=math.exp(min(50,loss)),token_accuracy=intersection/max(1,count),tokens=count)
        if any(m in config.get('metrics',[]) for m in ['bleu','rouge_l']):
            references=[];hypotheses=[]
            for row in rows:
                control();words=tokens(row.get('target_text',row['text']))[:config['max_length']]
                prompt=row['text']
                if config['model']!='seq2seq_gru':
                    cut=max(1,len(words)//2);prompt=' '.join(words[:cut]);words=words[cut:]
                references.append(words);hypotheses.append(tokens(generate_text(model,config,vocab,prompt,tokenizer)['text']))
            metrics.update(text_scores(references,hypotheses))
        return add_custom(metrics,config,np.asarray(truth),np.asarray(predictions)),[]
    if task=='detection':return add_custom(detection_metrics(predictions,truth),config,truth,predictions),[dict(prediction=p,truth=t,image=image) for p,t,image in zip(predictions,truth,preview)][:4]
    if task=='segmentation' and multiclass:
        diagonal=histogram.diagonal();total=histogram.sum(0)+histogram.sum(1);union=total-diagonal;valid=union>0
        iou=np.divide(diagonal,union,out=np.zeros(len(union),dtype=float),where=valid);dice=np.divide(2*diagonal,total,out=np.zeros(len(total),dtype=float),where=total>0)
        metrics=dict(loss=loss_sum/max(1,count),iou=float(iou[valid].mean()),dice=float(dice[valid].mean()),samples=count,confusion=histogram.tolist(),per_class=[dict(label=label,iou=float(iou[i]),dice=float(dice[i]),present=bool(valid[i])) for i,label in enumerate(data['labels'])])
        return add_custom(metrics,config,np.asarray(truth),np.asarray(predictions)),preview[:4]
    if task=='segmentation':return add_custom(dict(loss=loss_sum/max(1,count),iou=intersection/union if union else 1.,dice=2*intersection/(predicted_area+true_area) if predicted_area+true_area else 1.,samples=count),config,np.asarray(truth),np.asarray(predictions)),preview[:4]
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    labels=np.argmax(predictions,axis=1).tolist()
    metrics=dict(loss=loss_sum/max(1,count),accuracy=float(accuracy_score(truth,labels)),macro_f1=float(f1_score(truth,labels,labels=list(range(len(data['labels']))),average='macro',zero_division=0)),confusion=confusion_matrix(truth,labels,labels=list(range(len(data['labels'])))).tolist(),samples=count)
    metrics.update(classification_metrics(truth,labels,predictions));metrics=add_custom(metrics,config,np.asarray(truth),np.asarray(labels),np.asarray(predictions))
    return metrics,[dict(truth=t,predicted=p,scores=s) for t,p,s in zip(truth,labels,predictions)][:200]


def select_device(requested,fraction):
    import torch
    reason=None
    if requested=='cpu':return torch.device('cpu'),'Chọn CPU.'
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA không khả dụng trong môi trường PyTorch.')
        index=int(requested.split(':')[1]) if requested.startswith('cuda:') else max(range(torch.cuda.device_count()),key=lambda i:torch.cuda.mem_get_info(i)[0])
        device=torch.device(f'cuda:{index}');free,total=torch.cuda.mem_get_info(device)
        # Reserve at least 512 MB beyond this process budget. Never evict other processes.
        if free<total*fraction+512*1024**2:raise RuntimeError('GPU không còn đủ bộ nhớ cho ngân sách đã chọn.')
        torch.cuda.set_per_process_memory_fraction(fraction,device)
        probe=torch.ones((8,8),device=device);(probe@probe).sum().item();del probe
        return device,None
    except Exception as e:
        if requested!='auto':raise RuntimeError(f'GPU_PREFLIGHT: {e}') from e
        reason=str(e)
    return torch.device('cpu'),reason


def train(config,dataset_path,out,emit=lambda *args:None,control=lambda:None):
    import torch
    import numpy as np
    from safetensors.torch import save_file
    import importlib.metadata
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    torch.set_num_threads(2);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    data=load_dataset(dataset_path,config['task'],control)
    if config['task']=='segmentation' and config.get('segmentation_mode','binary')=='binary' and len(data['labels'])!=2:raise ValueError('MASK_MODE: Chọn multi-class cho dataset nhiều lớp.')
    rows=data['splits']['train'];validation=data['splits']['validation']
    def limited(items,limit):
        if not limit or len(items)<=limit:return list(items)
        result=list(items);random.Random(config['seed']).shuffle(result);return result[:limit]
    rows=limited(rows,config.get('train_limit'));validation=limited(validation,config.get('validation_limit'))
    if config['task']=='time_series':
        values=np.asarray([r['values'] for r in rows],dtype=float)
        config={**config,'series_scaler':dict(mean=float(values.mean()),std=max(float(values.std()),1e-8),window=values.shape[1])}
    device,reason=select_device(config['device'],config['memory_fraction'])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    vocab=vocabulary(rows,config['max_vocab']) if config['model'] in ['bigru','transformer','causal_gru','seq2seq_gru'] else None
    if vocab:save_json(out/'vocabulary.json',vocab)
    save_json(out/'config.json',config);save_json(out/'labels.json',data['labels'])
    save_json(out/'data-audit.json',dict(manifest_hash=data['manifest_hash'],file_hashes=data['file_hashes'],source_counts=data['counts'],used_train=len(rows),used_validation=len(validation),test_used_for_training=False))
    model,tokenizer=make_model(config,len(data['labels']),out,vocab,initialize=True);model=model.to(device)
    if (out/'pretrained-source.json').is_file():
        config['model_revision']=json.loads((out/'pretrained-source.json').read_text(encoding='utf-8'))['revision'];save_json(out/'config.json',config)
    parameters=[p for p in model.parameters() if p.requires_grad]
    options=dict(lr=config['learning_rate'],weight_decay=config['weight_decay'])
    if config['optimizer']=='SGD':options['momentum']=config['momentum']
    optimizer=getattr(torch.optim,config['optimizer'])(parameters,**options)
    manifest=dict(completed=False,task=config['task'],labels=data['labels'],model=config['model'],effective_device=str(device),fallback_reason=reason,memory_fraction=config['memory_fraction'],trainable_parameters=sum(p.numel() for p in parameters),total_parameters=sum(p.numel() for p in model.parameters()),dataset_hash=data['manifest_hash'],versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','safetensors','scikit-learn']},subset=bool(config.get('train_limit') or config.get('validation_limit')))
    version_names=['numpy','pillow']+(['transformers','tokenizers','huggingface-hub','accelerate','peft'] if config['model'] in ['bert_tiny','bert_tiny_lora','bert_base','distilbert','deberta','hf_classifier','gpt2','distilgpt2','hf_causal'] else [])
    manifest['versions'].update({name:importlib.metadata.version(name) for name in version_names})
    save_json(out/'manifest.json',manifest);emit('status',dict(state='RUNNING'));best=-math.inf;waiting=0;history=[]
    for epoch in range(1,config['epochs']+1):
        control();model.train()
        if config['task']=='image_classification' and config['freeze_backbone']:
            model.eval()
            for name,module in model.named_modules():
                if any(p.requires_grad for p in module.parameters(recurse=False)):module.train()
        training_loss=0.;samples=0
        for x,y in batches(rows,data,config,vocab,tokenizer,shuffle=True):
            control();x=to_device(x,device);y=to_device(y,device);optimizer.zero_grad(set_to_none=True)
            if config['task']=='detection':loss=sum(model(x,y).values())
            else:
                logits=forward(model,x)
                loss=loss_value(logits,y,config,'time_series') if config['task']=='time_series' else reconstruction_loss(logits,y,config) if config['task']=='image_reconstruction' else loss_value(logits,y,config,'text_generation') if config['task']=='text_generation' else segmentation_loss(logits,y,config) if config['task']=='segmentation' else loss_value(logits,y,config,config['task'])
            if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS: Hãy giảm learning rate và kiểm tra dữ liệu.')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,config['grad_clip']);optimizer.step()
            n=len(y);training_loss+=loss.item()*n;samples+=n
        metrics,preview=evaluate(model,validation,data,config,vocab,tokenizer,device,control)
        score=primary_score(metrics,config,config['task'])
        event=dict(epoch=epoch,train_loss=training_loss/max(1,samples),validation=metrics,elapsed_seconds=time.monotonic()-start,peak_vram_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0)
        history.append(event);save_json(out/'history.json',history);emit('metrics',event)
        if score>best:
            best=score;waiting=0
            # Cloning removes shared storage aliases (e.g. PEFT modules_to_save).
            save_file({k:v.detach().cpu().clone().contiguous() for k,v in model.state_dict().items()},str(out/'model.safetensors.pending'));os.replace(out/'model.safetensors.pending',out/'model.safetensors')
            save_json(out/'evaluation.json',metrics);save_json(out/'predictions.json',preview);manifest['best_epoch']=epoch
        else:waiting+=1
        if waiting>=config['patience']:break
    manifest.update(completed=True,epochs_completed=len(history),elapsed_seconds=time.monotonic()-start,peak_vram_mb=history[-1]['peak_vram_mb'],weights_sha256=digest(out/'model.safetensors'))
    save_json(out/'manifest.json',manifest)
    return manifest


def restore(out,device='cpu'):
    import torch
    from safetensors.torch import load_file
    out=Path(out);config=json.loads((out/'config.json').read_text(encoding='utf-8'));labels=json.loads((out/'labels.json').read_text(encoding='utf-8'));manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if digest(out/'model.safetensors')!=manifest['weights_sha256']:raise ValueError('WEIGHTS_CHECKSUM: Trọng số đã thay đổi.')
    vocab=json.loads((out/'vocabulary.json').read_text(encoding='utf-8')) if (out/'vocabulary.json').is_file() else None
    model,tokenizer=make_model(config,len(labels),out,vocab,initialize=False);model.load_state_dict(load_file(str(out/'model.safetensors')));model.to(torch.device(device)).eval()
    return model,tokenizer,vocab,config,labels


def holdout(out,dataset_path,control=lambda:None):
    model,tokenizer,vocab,config,labels=restore(out)
    data=load_dataset(dataset_path,config['task'],control);audit=json.loads((Path(out)/'data-audit.json').read_text(encoding='utf-8'))
    if data['manifest_hash']!=audit['manifest_hash'] or data['file_hashes']!=audit['file_hashes']:raise ValueError('DATASET_CHANGED: Dataset khác với lần huấn luyện.')
    metrics,preview=evaluate(model,data['splits']['test'],data,config,vocab,tokenizer,'cpu',control)
    save_json(Path(out)/'test-evaluation.json',metrics);save_json(Path(out)/'test-predictions.json',preview)
    return metrics


def predict(out,sample):
    import torch
    model,tokenizer,vocab,config,labels=restore(out)
    task=config['task']
    if task=='time_series':
        values=json.loads(sample) if isinstance(sample,str) else sample
        if not isinstance(values,list) or len(values)!=config['series_scaler']['window'] or any(type(v) not in [int,float] or not math.isfinite(v) for v in values):raise ValueError('SERIES_INPUT: Input must be a finite numeric array matching the training window.')
        x,_=get_item(dict(values=values,target=0),{'root':'.'},config)
        with torch.inference_mode():value=float(model(x[None])[0])
        return dict(prediction=value,window=len(values))
    key='text' if task in ['text_classification','text_generation'] else 'audio' if task=='audio_classification' else 'image'
    if key=='text':row=dict(text=sample,label=0);data={'root':'.'}
    else:
        path=Path(sample).resolve();row={key:path.name,'label':0};data={'root':str(path.parent)}
    if task=='text_generation':return generate_text(model,config,vocab,sample,tokenizer)
    # Segmentation/detection inference doesn't require ground truth annotation.
    input_config={**config,'task':'image_classification','annotation_coordinates':True} if task in ['segmentation','detection'] else config
    x,_=get_item(row,data,input_config,vocab,tokenizer)
    if task in ['segmentation','detection']:
        x=x*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]
    x={k:v[None] for k,v in x.items()} if isinstance(x,dict) else x[None]
    with torch.inference_mode():
        output=forward(model,[x[0]] if task=='detection' else x)
    if task=='image_reconstruction':return dict(reconstruction=image_preview(output[0][0]),mse=float(torch.nn.functional.mse_loss(output[0],x)))
    if task=='detection':return {k:v.cpu().tolist() for k,v in output[0].items()}
    if task=='segmentation' and config.get('segmentation_mode')=='multiclass':return dict(mask=output[0].argmax(0).to(torch.int64).tolist(),labels=labels)
    if task=='segmentation':return dict(mask=(output[0,0].sigmoid()>=.5).to(torch.uint8).tolist())
    scores=output.softmax(1)[0].tolist();return dict(label=labels[max(range(len(scores)),key=scores.__getitem__)],scores=scores)


def study(candidates,dataset_path,out,emit=lambda *args:None,control=lambda:None):
    """Bounded validation search. Reserved test rows are never evaluated or ranked."""
    import gc
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True);reports=[]
    initial_hash=digest(dataset_path)
    for index,config in enumerate(candidates):
        control()
        if digest(dataset_path)!=initial_hash:raise ValueError('DATASET_CHANGED: Manifest changed between trials.')
        directory=out/f'trial-{index+1}'
        manifest=train(config,dataset_path,directory,control=control)
        metric=json.loads((directory/'evaluation.json').read_text(encoding='utf-8'))
        objective=config.get('primary_metric') or 'accuracy'
        score=float(metric[objective]);maximize=config.get('primary_mode','auto')=='max' or config.get('primary_mode','auto')=='auto' and objective not in ['mse','mae','rmse','mape','log_loss','perplexity','loss']
        report=dict(trial=index+1,config=config,validation=metric,objective=objective,score=score,maximize=maximize,epochs_completed=manifest['epochs_completed'])
        reports.append(report);save_json(out/'result.json',dict(trials=reports,test_used=False,selection_split='validation',completed=False));emit('metrics',dict(trial=index+1,score=score,objective=objective))
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    best=max(reports,key=lambda r:r['score'] if r['maximize'] else -r['score'])
    result=dict(completed=True,trials=reports,best_config=best['config'],best_trial=best['trial'],test_used=False,selection_split='validation',note='Validation used to select hyperparameters; use the untouched test split only after final selection.')
    save_json(out/'result.json',result);return result


def generate_text(model,config,vocab,sample,tokenizer=None):
    import torch
    if tokenizer:
        device=next(model.parameters()).device;inputs=tokenizer(sample,return_tensors='pt',truncation=True,max_length=config['max_length'])
        inputs={k:v.to(device) for k,v in inputs.items()};length=inputs['input_ids'].shape[1]
        with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=config['max_length'],do_sample=False,pad_token_id=tokenizer.pad_token_id)
        ids=output[0,length:].tolist();return dict(text=tokenizer.decode(ids,skip_special_tokens=True),tokens=ids,decoding='greedy',maximum_tokens=config['max_length'])
    maximum=config['max_length'];ids=[vocab.get(t,1) for t in tokens(sample)][:maximum-1];inverse={v:k for k,v in vocab.items()};generated=[]
    device=next(model.parameters()).device;source=torch.tensor([ids+[3]+[0]*(maximum-len(ids)-1)],device=device)
    context=[2] if config['model']=='seq2seq_gru' else [2]+ids
    with torch.inference_mode():
        for _ in range(maximum-len(context)+1):
            inputs={'source':source,'teacher':torch.tensor([context],device=device)} if config['model']=='seq2seq_gru' else torch.tensor([context],device=device)
            logits=forward(model,inputs)[0,-1].clone();logits[[0,2]]=-torch.inf;index=int(logits.argmax())
            if index==3:break
            generated.append(index);context.append(index)
    return dict(text=' '.join(inverse.get(i,'<unk>') for i in generated),tokens=generated,decoding='greedy',maximum_tokens=maximum)
