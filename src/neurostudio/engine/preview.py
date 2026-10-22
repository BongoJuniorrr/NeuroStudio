from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import random


def png(image):
    buffer=io.BytesIO();image.save(buffer,format='PNG');return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def preview(spec,descriptor):
    import numpy as np
    from .portable import tabular_data,resolved_splits,sklearn_model,image_data,image_splits,seed_all,verify_dataset
    verify_dataset(descriptor)
    if spec['backend']=='sklearn':
        frame,y,labels=tabular_data(spec,descriptor);splits=resolved_splits(y,{**spec['data'],'stratify':False} if spec.get('task')=='regression' else spec['data'],spec['training']['seed'],descriptor)
        pipeline=sklearn_model(spec,frame);prep=pipeline[:-1];prep.fit(frame.iloc[splits['train']])
        subset=frame.iloc[splits['validation'][:5]]
        matrix=prep.transform(subset);features=prep.get_feature_names_out().tolist();visible=matrix[:,:100]
        if hasattr(visible,'toarray'):visible=visible.toarray()
        return dict(kind='tabular',before=json.loads(subset.to_json(orient='records')),after=visible.tolist(),features=features[:100],total_features=len(features),truncated=len(features)>100,fit_samples=len(splits['train']),source='validation transformed using train-fitted state')
    from PIL import Image
    seed_all(spec['training']['seed'])
    ds,y,_=image_data(spec,descriptor);aug,_,_=image_data(spec,descriptor,True);ix=image_splits(spec,descriptor,y)['train'][0]
    def render(tensor):
        a=tensor.detach().cpu().numpy().transpose(1,2,0)
        if spec['data']['normalize']:a=a*.5+.5
        return png(Image.fromarray((a.clip(0,1)*255).astype('uint8')))
    return dict(kind='image',before=render(ds[ix][0]),after=render(aug[ix][0]),seed=spec['training']['seed'],source='training sample; stochastic augmentation disabled for validation/test')


def explain():
    import numpy as np
    import torch
    x=np.linspace(-5,5,101)
    image=torch.arange(25,dtype=torch.float32).reshape(1,1,5,5)/25
    kernel=torch.tensor([[[[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]]]])
    conv=torch.nn.functional.conv2d(image,kernel)
    pool=torch.nn.functional.max_pool2d(image,2)
    return dict(activations=[dict(x=float(v),ReLU=float(max(v,0)),Sigmoid=float(1/(1+np.exp(-v))),Tanh=float(np.tanh(v))) for v in x],
        input=image[0,0].tolist(),kernel=kernel[0,0].tolist(),convolution=conv[0,0].tolist(),pool=pool[0,0].tolist(),
        note='Actual PyTorch conv2d/max_pool2d on a toy 5×5 tensor; not a learned feature map.')


def feature_maps(directory,layer=None):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from .portable import build_model
    root=Path(directory);spec=json.loads((root/'config.json').read_text(encoding='utf-8'))
    if spec['backend']!='torch':raise ValueError('FEATUREMAP_BACKEND: Cần model PyTorch.')
    import time
    began=time.perf_counter()
    model=build_model(spec);model.load_state_dict(load_file(str(root/'model.safetensors')));model.eval()
    names=[n for n,m in model.layers.items() if isinstance(m,torch.nn.Conv2d)]
    if not names:raise ValueError('FEATUREMAP_LAYER: Model không có Conv2D.')
    name=layer or names[0]
    if name not in names:raise ValueError('FEATUREMAP_LAYER: Chọn Conv2D thuộc run này.')
    captured=[]
    handle=model.layers[name].register_forward_hook(lambda module,args,output:captured.append(output.detach().cpu()))
    try:
        sample=np.load(root/'sample.npy',allow_pickle=False)
        with torch.inference_mode():model(torch.from_numpy(sample[None]).float())
    finally:handle.remove()
    maps=captured[0][0];images=[]
    for index,m in enumerate(maps[:8]):
        a=m.numpy();lo,hi=float(a.min()),float(a.max());v=(a-lo)/max(hi-lo,1e-8)
        rgb=np.stack([20+v*100,45+v*150,30+v*100],axis=-1).astype('uint8')
        images.append(dict(channel=index,min=lo,max=hi,image=png(Image.fromarray(rgb))))
    tensor=torch.from_numpy(sample[None]).float().requires_grad_(True)
    logits=model(tensor);predicted=int(logits.detach().argmax(1)[0]);logits[0,predicted].backward()
    saliency=tensor.grad.detach().abs().amax(1)[0].numpy();lo,hi=float(saliency.min()),float(saliency.max());heat=(saliency-lo)/max(hi-lo,1e-12)
    saliency_image=png(Image.fromarray(np.stack([heat*155+50,heat*105+40,heat*80+110],axis=-1).astype('uint8')))
    gradients=[dict(layer=n,gradient_l2=float(p.grad.norm()),parameter_l2=float(p.detach().norm())) for n,p in model.named_parameters() if p.grad is not None]
    return dict(layer=name,shape=list(captured[0].shape),images=images,
        diagnostics=dict(predicted_class=predicted,saliency_image=saliency_image,gradient_norms=gradients,elapsed_seconds=time.perf_counter()-began,objective='Predicted-class logit sensitivity, CPU eval mode; not training loss gradients or a causal explanation.'),
        note='Best checkpoint, one included validation sample, CPU eval mode. Each channel is normalized independently; color intensity is not comparable across channels.')
