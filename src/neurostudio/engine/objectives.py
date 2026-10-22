"""Portable task-aware metrics, loss functions and explicit local custom code."""
import ast,math,operator
from functools import lru_cache
CLASSIFICATION=['accuracy','macro_f1','weighted_f1','micro_f1','macro_precision','macro_recall','balanced_accuracy','mcc','cohen_kappa','log_loss','roc_auc']
METRICS={'classification':CLASSIFICATION,'image_classification':CLASSIFICATION,'text_classification':CLASSIFICATION,'audio_classification':CLASSIFICATION,'regression':['rmse','mae','r2','mse','mape'],'image_reconstruction':['mse','mae','psnr'],'segmentation':['dice','iou','loss'],'detection':['ap50','recall50'],'text_generation':['perplexity','token_accuracy','loss','bleu','rouge_l'],'time_series':['rmse','mae','mse','r2']}
DEFAULTS={'classification':'accuracy','image_classification':'accuracy','text_classification':'accuracy','audio_classification':'accuracy','regression':'rmse','image_reconstruction':'mse','segmentation':'dice','detection':'ap50','text_generation':'perplexity','time_series':'rmse'}
LOSSES={'classification':['auto','cross_entropy','nll','focal','bce_logits','multi_margin','kl_div','custom'],'image_classification':['auto','cross_entropy','nll','focal','bce_logits','multi_margin','kl_div','custom'],'text_classification':['auto','cross_entropy','nll','focal','bce_logits','multi_margin','kl_div','custom'],'audio_classification':['auto','cross_entropy','nll','focal','bce_logits','multi_margin','kl_div','custom'],'text_generation':['auto','cross_entropy','nll','focal','custom'],'segmentation':['auto','bce','dice','bce_dice','jaccard','cross_entropy','focal','custom'],'image_reconstruction':['auto','mse','mae','huber','smooth_l1','custom'],'time_series':['auto','mse','mae','huber','smooth_l1','custom']}
MINIMIZE={'loss','mse','mae','rmse','mape','log_loss','perplexity'}


def formula(expression,values):
    if len(expression)>400:raise ValueError('FORMULA_SIZE: Maximum 400 characters.')
    tree=ast.parse(expression,mode='eval');functions={'min':min,'max':max,'abs':abs,'sqrt':math.sqrt,'log':math.log,'exp':math.exp};ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow}
    if len(list(ast.walk(tree)))>80:raise ValueError('FORMULA_SIZE: Too many operations.')
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in [int,float] and abs(node.value)<=1e6:return node.value
        if isinstance(node,ast.Name) and node.id in values:return values[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):return (-1 if isinstance(node.op,ast.USub) else 1)*visit(node.operand)
        if isinstance(node,ast.BinOp) and type(node.op) in ops:
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Pow) and abs(b)>10:raise ValueError('FORMULA_POWER: Exponent is limited to 10.')
            return ops[type(node.op)](a,b)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions and not node.keywords and len(node.args)<=8:return functions[node.func.id](*[visit(a) for a in node.args])
        raise ValueError('FORMULA_SYNTAX: Use metric names, numbers, arithmetic, min/max/abs/sqrt/log/exp.')
    value=float(visit(tree.body))
    if not math.isfinite(value):raise ValueError('FORMULA_NONFINITE: Metric must be finite.')
    return value


@lru_cache(maxsize=16)
def custom_function(code,name):
    import numpy as np
    import torch
    namespace={'np':np,'torch':torch,'math':math}
    exec(compile(code,'<user-'+name+'>','exec'),namespace)
    fn=namespace.get(name)
    if not callable(fn):raise ValueError(f'CUSTOM_FUNCTION: Define {name}.')
    return fn


def add_custom(result,config,y_true=None,y_pred=None,y_score=None):
    result=dict(result)
    if config.get('metric_code'):
        extras=custom_function(config['metric_code'],'custom_metrics')(y_true,y_pred,y_score)
        if not isinstance(extras,dict):raise ValueError('CUSTOM_METRICS: Return a dictionary of numeric metrics.')
        for key,value in extras.items():
            if not isinstance(key,str) or not key.startswith('custom_') or not math.isfinite(float(value)):raise ValueError('CUSTOM_METRICS: Use custom_ names and finite numeric values.')
            result[key]=float(value)
    for name,expression in config.get('metric_formulas',{}).items():
        if name in result:raise ValueError('METRIC_NAME: Custom formulas must not overwrite built-in metrics.')
        result[name]=formula(expression,result)
    return result


def classification_metrics(truth,predictions,scores):
    import numpy as np
    from sklearn import metrics as m
    truth=np.asarray(truth);predictions=np.asarray(predictions);scores=np.asarray(scores,dtype=float)
    if scores.ndim==2:scores=scores/np.maximum(scores.sum(1,keepdims=True),1e-15)
    result=dict(accuracy=float(m.accuracy_score(truth,predictions)),macro_f1=float(m.f1_score(truth,predictions,average='macro',zero_division=0)),weighted_f1=float(m.f1_score(truth,predictions,average='weighted',zero_division=0)),micro_f1=float(m.f1_score(truth,predictions,average='micro',zero_division=0)),macro_precision=float(m.precision_score(truth,predictions,average='macro',zero_division=0)),macro_recall=float(m.recall_score(truth,predictions,average='macro',zero_division=0)),balanced_accuracy=float(m.balanced_accuracy_score(truth,predictions)),mcc=float(m.matthews_corrcoef(truth,predictions)))
    kappa=float(m.cohen_kappa_score(truth,predictions));result['cohen_kappa']=kappa if math.isfinite(kappa) else 0.
    if scores.ndim==2:
        result['log_loss']=float(m.log_loss(truth,scores,labels=list(range(scores.shape[1]))))
        if len(set(truth))==scores.shape[1]:result['roc_auc']=float(m.roc_auc_score(truth,scores[:,1] if scores.shape[1]==2 else scores,multi_class='ovr',average='macro'))
    return result


def primary_score(metrics,config,task):
    name=config.get('primary_metric') or DEFAULTS[task]
    if name not in metrics or not math.isfinite(float(metrics[name])):raise ValueError(f'PRIMARY_METRIC: {name} is unavailable for this validation split. Check labels, metrics and custom code.')
    minimize=config.get('primary_mode','auto')=='min' or config.get('primary_mode','auto')=='auto' and name in MINIMIZE
    return -float(metrics[name]) if minimize else float(metrics[name])


def loss_value(prediction,target,config,task):
    import torch
    from torch.nn import functional as f
    name=config.get('loss','auto')
    if name=='CrossEntropyLoss':name='cross_entropy'
    if name=='custom':
        loss=custom_function(config.get('loss_code',''),'custom_loss')(prediction,target)
    elif task in ['image_reconstruction','time_series']:
        name='mse' if name=='auto' else name
        fn={'mse':f.mse_loss,'mae':f.l1_loss,'huber':f.huber_loss,'smooth_l1':f.smooth_l1_loss}[name];loss=fn(prediction,target)
    else:
        if task=='text_generation':
            prediction=prediction.flatten(0,1);target=target.flatten();mask=target.ne(-100) if config.get('model') in ['gpt2','distilgpt2','hf_causal'] else target.ne(0);prediction=prediction[mask];target=target[mask]
        smooth=config.get('label_smoothing',0)
        if name in ['auto','cross_entropy']:loss=f.cross_entropy(prediction,target,label_smoothing=smooth)
        elif name=='nll':loss=f.nll_loss(prediction.log_softmax(1),target)
        elif name=='focal':
            ce=f.cross_entropy(prediction,target,reduction='none');loss=((1-torch.exp(-ce))**config.get('focal_gamma',2)*ce).mean()
        elif name=='bce_logits':loss=f.binary_cross_entropy_with_logits(prediction,f.one_hot(target,prediction.shape[1]).to(prediction.dtype))
        elif name=='multi_margin':loss=f.multi_margin_loss(prediction,target)
        elif name=='kl_div':loss=f.kl_div(prediction.log_softmax(1),f.one_hot(target,prediction.shape[1]).to(prediction.dtype),reduction='batchmean')
        else:raise ValueError('LOSS_UNSUPPORTED: Choose a loss supported by this task.')
    if not isinstance(loss,torch.Tensor) or loss.numel()!=1 or not torch.isfinite(loss):raise ValueError('CUSTOM_LOSS: Loss must be one finite scalar tensor.')
    if prediction.requires_grad and not loss.requires_grad:raise ValueError('CUSTOM_LOSS: Loss must retain gradients during training.')
    return loss


def text_scores(references,hypotheses):
    from collections import Counter
    clipped=[0]*4;counts=[0]*4;reference_length=0;hypothesis_length=0;rouge=[]
    for ref,hyp in zip(references,hypotheses):
        reference_length+=len(ref);hypothesis_length+=len(hyp)
        for n in range(1,5):
            r=Counter(tuple(ref[i:i+n]) for i in range(max(0,len(ref)-n+1)));h=Counter(tuple(hyp[i:i+n]) for i in range(max(0,len(hyp)-n+1)));clipped[n-1]+=sum((r&h).values());counts[n-1]+=sum(h.values())
        previous=[0]*(len(hyp)+1)
        for a in ref:
            current=[0]
            for j,b in enumerate(hyp):current.append(previous[j]+1 if a==b else max(previous[j+1],current[-1]))
            previous=current
        rouge.append(2*previous[-1]/max(1,len(ref)+len(hyp)))
    used=[i for i in range(4) if counts[i]>0]
    bleu=math.exp(min(0,1-reference_length/max(1,hypothesis_length)))*math.exp(sum(math.log((clipped[i]+1)/(counts[i]+1)) for i in used)/len(used)) if used and hypothesis_length else 0.
    return dict(bleu=bleu,rouge_l=sum(rouge)/max(1,len(rouge)),generation_samples=len(references))
