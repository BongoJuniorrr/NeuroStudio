"""Portable tensor-block contracts, shape inference and constructors.
No torch import at catalog/validation time. Copied into standalone exports.
"""
import math


def parameter(default,minimum=None,maximum=None,description='',options=None,types=None,**extra):
    return dict(default=default,minimum=minimum,maximum=maximum,description=description,options=options,type=type(default).__name__,**({'types':types} if types else {}),**extra)


def spatial(default,minimum=1,nullable=False,strings=False):
    return parameter(default,minimum,4096,types=['int','list']+(['NoneType'] if nullable else [])+(['str'] if strings else []),description='JSON scalar or per-axis array'+('; null uses the native default' if nullable else '')+('; "same" or "valid" padding is also accepted' if strings else ''),item_minimum=minimum)


LAYER_SPECS={};NN_NAMES={}
def register(name,native,category,description,params=None,inputs=1):
    LAYER_SPECS[name]=dict(category=category,description=description,params=params or {},inputs=inputs)
    if native:NN_NAMES[name]=native


for dimension in [1,2,3]:
    for transposed in [False,True]:
        name=f'Conv{"Transpose" if transposed else ""}{dimension}D'
        params=dict(in_channels=parameter(3,1,4096),out_channels=parameter(16,1,4096),kernel_size=spatial(3),stride=spatial(1),padding=spatial(1,0,strings=not transposed),dilation=spatial(1),groups=parameter(1,1,4096),bias=parameter(True),padding_mode=parameter('zeros',options=['zeros'] if transposed else ['zeros','reflect','replicate','circular']))
        if transposed:params['output_padding']=spatial(0,0)
        register(name,f'Conv{"Transpose" if transposed else ""}{dimension}d','Convolution',f'{dimension}D '+('transposed convolution' if transposed else 'convolution, including grouped and depthwise kernels')+'.',params)
    for family in ['MaxPool','AvgPool']:
        params=dict(kernel_size=spatial(2),stride=spatial(2,nullable=True),padding=spatial(0,0),ceil_mode=parameter(False))
        if family=='MaxPool':params.update(dilation=spatial(1),return_indices=parameter(False,options=[False],description='This tensor graph exposes the pooled tensor. Index-output/unpool ports are not supported.'))
        else:
            params['count_include_pad']=parameter(True)
            if dimension>1:params['divisor_override']=parameter(None,1,1000000,types=['int','NoneType'],description='null uses the native window divisor.')
        register(f'{family}{dimension}D',f'{family}{dimension}d','Pooling',f'{dimension}D pooling with native floor/ceil output sizing.',params)
    for family in ['AdaptiveAvgPool','AdaptiveMaxPool']:
        params=dict(output_size=spatial(1,nullable=True))
        if family=='AdaptiveMaxPool':params['return_indices']=parameter(False,options=[False],description='Tensor output only; no pooling-index port.')
        register(f'{family}{dimension}D',f'{family}{dimension}d','Pooling','Adaptive pooling; null per axis preserves its size.',params)
    for family in ['BatchNorm','InstanceNorm']:
        params=dict(num_features=parameter(32,1,100000),eps=parameter(.00001,.000000000001,1),momentum=parameter(.1,0,1,types=['float','int','NoneType'] if family=='BatchNorm' else ['float','int'],description='BatchNorm: null selects cumulative moving average.' if family=='BatchNorm' else ''),affine=parameter(family=='BatchNorm'),track_running_stats=parameter(family=='BatchNorm'))
        register(f'{family}{dimension}D',f'{family}{dimension}d','Normalization','Channel normalization with configurable affine parameters and running statistics.',params)
    register(f'Dropout{dimension}D',f'Dropout{dimension}d','Regularization','Drop entire feature channels during training.',dict(p=parameter(.2,0,1),inplace=parameter(False)))
    for family in ['ConstantPad','ReflectionPad','ReplicationPad','ZeroPad']:
        params=dict(padding=spatial(1,-4096))
        if family=='ConstantPad':params['value']=parameter(0.,-1000000,1000000)
        register(f'{family}{dimension}D',f'{family}{dimension}d','Padding','Per-side padding in PyTorch order: left, right, top, bottom, front, back.',params)

register('Linear','Linear','Dense','Linear transformation over the last dimension.',dict(in_features=parameter(32,1,1000000),out_features=parameter(10,1,100000),bias=parameter(True)))
register('BatchNorm2D','BatchNorm2d','Normalization','Normalize image channels [B,C,H,W]; running statistics are saved in checkpoints.',LAYER_SPECS['BatchNorm2D']['params'])
register('GroupNorm','GroupNorm','Normalization','Channel groups; num_channels must be divisible by num_groups.',dict(num_groups=parameter(1,1,4096),num_channels=parameter(32,1,100000),eps=parameter(.00001,.000000000001,1),affine=parameter(True)))
register('LayerNorm','LayerNorm','Normalization','Normalize the trailing dimensions listed in normalized_shape.',dict(normalized_shape=spatial([32]),eps=parameter(.00001,.000000000001,1),elementwise_affine=parameter(True),bias=parameter(True)))
register('RMSNorm','RMSNorm','Normalization','Root-mean-square normalization of trailing dimensions.',dict(normalized_shape=spatial([32]),eps=parameter(None,.000000000001,1,types=['float','int','NoneType']),elementwise_affine=parameter(True)))
register('LocalResponseNorm','LocalResponseNorm','Normalization','Local response normalization across channels.',dict(size=parameter(5,1,4096),alpha=parameter(.0001,0,100),beta=parameter(.75,0,100),k=parameter(1.,.000001,1000)))
for name in ['Dropout','AlphaDropout','FeatureAlphaDropout']:
    register(name,name,'Regularization','Training-only stochastic regularization.',dict(p=parameter(.2,0,1),inplace=parameter(False)))

activation_params={
    'ReLU':dict(inplace=parameter(False)),'ReLU6':dict(inplace=parameter(False)),
    'LeakyReLU':dict(negative_slope=parameter(.01,0,100),inplace=parameter(False)),
    'PReLU':dict(num_parameters=parameter(1,1,100000),init=parameter(.25,-100,100)),
    'RReLU':dict(lower=parameter(.125,0,100),upper=parameter(1/3,0,100),inplace=parameter(False)),
    'ELU':dict(alpha=parameter(1.,0,100),inplace=parameter(False)),
    'CELU':dict(alpha=parameter(1.,.000001,100),inplace=parameter(False)),
    'SELU':dict(inplace=parameter(False)),'GELU':dict(approximate=parameter('none',options=['none','tanh'])),
    'SiLU':dict(inplace=parameter(False)),'Mish':dict(inplace=parameter(False)),
    'Hardswish':dict(inplace=parameter(False)),'Hardsigmoid':dict(inplace=parameter(False)),
    'Hardtanh':dict(min_val=parameter(-1.,-1000000,1000000),max_val=parameter(1.,-1000000,1000000),inplace=parameter(False)),
    'Softplus':dict(beta=parameter(1.,.000001,100),threshold=parameter(20.,-1000000,1000000)),
    'Softshrink':dict(lambd=parameter(.5,0,1000000)),'Hardshrink':dict(lambd=parameter(.5,0,1000000)),
    'Threshold':dict(threshold=parameter(0.,-1000000,1000000),value=parameter(0.,-1000000,1000000),inplace=parameter(False)),
    'Sigmoid':{},'Tanh':{},'Softsign':{},'LogSigmoid':{},'Tanhshrink':{},'Identity':{},
    'Softmax':dict(dim=parameter(1,-8,8)),'LogSoftmax':dict(dim=parameter(1,-8,8)),
    'Softmin':dict(dim=parameter(1,-8,8)),'GLU':dict(dim=parameter(-1,-8,8))}
for name,params in activation_params.items():register(name,name,'Activation','Elementwise activation.' if name not in ['Softmax','LogSoftmax','Softmin','GLU'] else 'Dimension-aware activation.',params)
register('Flatten','Flatten','Shape','Flatten a dimension range while preserving the batch axis.',dict(start_dim=parameter(1,1,8),end_dim=parameter(-1,-8,8)))
register('Reshape',None,'Shape','Reshape after batch; one -1 dimension may be inferred.',dict(shape=parameter([16,8,8],types=['list'])))
register('Permute',None,'Shape','Reorder axes. Keep the batch axis at index 0.',dict(dims=parameter([0,2,3,1],types=['list'])))
register('Transpose',None,'Shape','Swap two non-batch dimensions.',dict(dim0=parameter(1,-8,8),dim1=parameter(2,-8,8)))
register('Unsqueeze',None,'Shape','Insert a singleton non-batch dimension.',dict(dim=parameter(1,-8,8)))
register('Squeeze',None,'Shape','Remove singleton non-batch dimensions; null removes all of them.',dict(dim=parameter(None,-8,8,types=['int','NoneType'])))
register('Unflatten','Unflatten','Shape','Split one non-batch dimension into a shape; one -1 is allowed.',dict(dim=parameter(1,-8,8),unflattened_size=parameter([3,32,32],types=['list'])))
for name in ['ReduceMean','ReduceSum','ReduceMax']:
    register(name,None,'Shape','Reduce selected non-batch dimensions.',dict(dim=parameter(1,-8,8,types=['int','list']),keepdim=parameter(False)))
for name in ['Upsample','Interpolate']:
    register(name,None,'Resampling','Resize spatial axes. Choose size OR scale_factor, not both.',dict(size=spatial(None,nullable=True),scale_factor=parameter(2.,.000001,4096,types=['float','int','list','NoneType'],item_minimum=.000001),mode=parameter('nearest',options=['nearest','linear','bilinear','bicubic','trilinear','area','nearest-exact']),align_corners=parameter(None,types=['bool','NoneType']),recompute_scale_factor=parameter(None,types=['bool','NoneType']),**({'antialias':parameter(False)} if name=='Interpolate' else {})))
for name,key in [('PixelShuffle','upscale_factor'),('PixelUnshuffle','downscale_factor'),('ChannelShuffle','groups')]:register(name,name,'Resampling','Rearrange channels and spatial samples without interpolation.',{key:parameter(2,1,1024)})
for name in ['Unfold','Fold']:
    params=dict(kernel_size=spatial(3),dilation=spatial(1),padding=spatial(0,0),stride=spatial(1))
    if name=='Fold':params['output_size']=spatial([32,32])
    register(name,name,'Shape','Extract image patches.' if name=='Unfold' else 'Sum overlapping image patches.',params)
for name in ['Add','Multiply']:
    register(name,None,'Merge','Two inputs with identical shape; no implicit broadcasting.',inputs=2)
register('Concatenate',None,'Merge','Concatenate two tensors along a non-batch dimension.',dict(dim=parameter(1,-8,8)),2)
for name in ['RNN','GRU','LSTM']:
    params=dict(input_size=parameter(32,1,100000),hidden_size=parameter(32,1,4096),num_layers=parameter(1,1,32),bias=parameter(True),batch_first=parameter(True,options=[True],description='Sequence tensors use [B,L,F].'),dropout=parameter(0.,0,1),bidirectional=parameter(False),output_mode=parameter('sequence',options=['sequence','last']))
    if name=='LSTM':params['proj_size']=parameter(0,0,4096)
    if name=='RNN':params['nonlinearity']=parameter('tanh',options=['tanh','relu'])
    register(name,None,'Sequence','Stateless recurrent layer. Exposes sequence or final hidden features; state ports are not exposed.',params)
register('SelfAttention',None,'Sequence','Batch-first self-attention. Exposes attention output, not weight/state ports.',dict(embed_dim=parameter(32,1,4096),num_heads=parameter(4,1,256),dropout=parameter(0.,0,1),bias=parameter(True),add_bias_kv=parameter(False),add_zero_attn=parameter(False),causal=parameter(False)))
register('TransformerEncoderLayer','TransformerEncoderLayer','Sequence','Batch-first encoder layer; no external attention/padding-mask ports.',dict(d_model=parameter(32,1,4096),nhead=parameter(4,1,256),dim_feedforward=parameter(128,1,16384),dropout=parameter(.1,0,1),activation=parameter('relu',options=['relu','gelu']),layer_norm_eps=parameter(.00001,.000000000001,1),batch_first=parameter(True,options=[True]),norm_first=parameter(False),bias=parameter(True)))


def validate_parameter(value,rule):
    kinds=rule.get('types',[rule['type']]);kind=type(value).__name__
    if kind not in kinds and not(kind=='int' and 'float' in kinds):return False
    if isinstance(value,float) and not math.isfinite(value):return False
    if rule.get('options') is not None and value not in rule['options']:return False
    if type(value) in (int,float):
        if rule.get('minimum') is not None and value<rule['minimum']:return False
        if rule.get('maximum') is not None and value>rule['maximum']:return False
    if isinstance(value,list):
        if not value or len(value)>8:return False
        # Shapes/permutations have signed integer entries. Spatial vectors carry bounds.
        if any(type(x) not in (int,float,type(None)) or isinstance(x,float) and not math.isfinite(x) for x in value):return False
        for x in value:
            if x is None:
                if 'NoneType' not in kinds:return False
            elif rule.get('item_minimum') is not None and x<rule['item_minimum']:return False
            elif rule.get('maximum') is not None and x>rule['maximum']:return False
    return True


def axes(value,count,allow_none=False):
    values=list(value) if isinstance(value,(list,tuple)) else [value]*count
    if len(values)!=count or any(type(x) is not int and not(allow_none and x is None) for x in values):raise ValueError(f'Expected {count} integer spatial values'+(' or null' if allow_none else '')+'.')
    return values


def dim_index(value,rank,insertion=False):
    limit=rank+int(insertion)
    if type(value) is not int or not -limit<=value<limit:raise ValueError('Dimension is outside tensor rank.')
    index=value%limit
    if index==0:raise ValueError('Keep the batch axis unchanged.')
    return index


def inferred_shape(target,size):
    if not target or any(type(x) is not int or x==0 or x< -1 for x in target) or target.count(-1)>1:raise ValueError('Use positive shape dimensions and at most one -1.')
    product=math.prod(x for x in target if x!=-1)
    if size%product or -1 not in target and product!=size:raise ValueError('Reshape must preserve element count.')
    return [size//product if x==-1 else x for x in target]


def infer_layer(name,p,inputs):
    """Return tensor shape and number of trainable scalar parameters."""
    s=list(inputs[0]);params=0;rank=len(s)
    if name.startswith(('Conv1','Conv2','Conv3','ConvTranspose')):
        d=int(name[-2]);transposed='Transpose' in name
        if rank!=d+2 or s[1]!=p['in_channels']:raise ValueError(f'{name} needs [B,C,spatial axes] with C = in_channels.')
        if p['in_channels']%p['groups'] or p['out_channels']%p['groups']:raise ValueError('Both channel counts must be divisible by groups.')
        k=axes(p['kernel_size'],d);stride=axes(p['stride'],d);dilation=axes(p['dilation'],d)
        padding=p['padding']
        if isinstance(padding,str):
            if padding not in ['same','valid'] or padding=='same' and any(v!=1 for v in stride):raise ValueError('Padding must be same/valid; same requires stride 1.')
            padding=[0]*d if padding=='valid' else [(dilation[i]*(k[i]-1))//2 for i in range(d)]
            result=list(s[2:]) if p['padding']=='same' else [(x-dilation[i]*(k[i]-1)-1)//stride[i]+1 for i,x in enumerate(s[2:])]
        else:
            padding=axes(padding,d)
            result=[(x+2*padding[i]-dilation[i]*(k[i]-1)-1)//stride[i]+1 for i,x in enumerate(s[2:])]
        if transposed:
            output=axes(p['output_padding'],d)
            if any(output[i]>=max(stride[i],dilation[i]) for i in range(d)):raise ValueError('output_padding must be smaller than stride or dilation on each axis.')
            result=[(x-1)*stride[i]-2*padding[i]+dilation[i]*(k[i]-1)+output[i]+1 for i,x in enumerate(s[2:])]
        elif p['padding_mode']=='reflect' and any(padding[i]>=x or p['padding']=='same' and dilation[i]*(k[i]-1)-padding[i]>=x for i,x in enumerate(s[2:])):raise ValueError('Reflection padding must be smaller than its input dimension.')
        s=[s[0],p['out_channels'],*result];params=p['out_channels']*(p['in_channels']//p['groups'])*math.prod(k)+int(p['bias'])*p['out_channels']
    elif name.startswith(('MaxPool','AvgPool')):
        d=int(name[-2])
        if rank!=d+2:raise ValueError(f'{name} needs rank {d+2}.')
        k=axes(p['kernel_size'],d);stride=axes(p['stride'] if p['stride'] is not None else p['kernel_size'],d);padding=axes(p['padding'],d);dilation=axes(p.get('dilation',1),d)
        if any(padding[i]>k[i]//2 for i in range(d)):raise ValueError('Pooling padding cannot exceed half the kernel size.')
        result=[]
        for i,x in enumerate(s[2:]):
            numerator=x+2*padding[i]-dilation[i]*(k[i]-1)-1
            out=(numerator+(stride[i]-1 if p['ceil_mode'] else 0))//stride[i]+1
            if p['ceil_mode'] and (out-1)*stride[i]>=x+padding[i]:out-=1
            result.append(out)
        s=s[:2]+result
    elif name.startswith('Adaptive'):
        d=int(name[-2])
        if rank!=d+2:raise ValueError(f'{name} needs rank {d+2}.')
        if p['output_size'] is None:raise ValueError('Use a per-axis array such as [null, 1] to preserve an adaptive spatial axis.')
        output=axes(p['output_size'],d,allow_none=d>1);s=s[:2]+[x if y is None else y for x,y in zip(s[2:],output)]
    elif name.startswith(('BatchNorm','InstanceNorm')):
        d=int(name[-2]);allowed=[2,3] if name=='BatchNorm1D' else [d+2]
        if rank not in allowed or s[1]!=p['num_features']:raise ValueError(f'{name} needs matching channel count and tensor rank {allowed}.')
        params=2*p['num_features'] if p['affine'] else 0
    elif name=='GroupNorm':
        if rank<2 or s[1]!=p['num_channels'] or p['num_channels']%p['num_groups']:raise ValueError('GroupNorm channels must match and divide evenly into groups.')
        params=2*p['num_channels'] if p['affine'] else 0
    elif name in ['LayerNorm','RMSNorm']:
        shape=p['normalized_shape'] if isinstance(p['normalized_shape'],list) else [p['normalized_shape']]
        if any(type(x) is not int or x<1 for x in shape) or len(shape)>=rank or s[-len(shape):]!=shape:raise ValueError('normalized_shape must match trailing non-batch dimensions.')
        params=math.prod(shape)*(1+int(p.get('bias',False))) if p['elementwise_affine'] else 0
    elif name=='LocalResponseNorm':
        if rank<3:raise ValueError('LocalResponseNorm needs rank >=3.')
    elif name=='Linear':
        if s[-1]!=p['in_features']:raise ValueError('Linear in_features must match the last tensor dimension.')
        s[-1]=p['out_features'];params=p['out_features']*(p['in_features']+int(p['bias']))
    elif name=='PReLU':
        if p['num_parameters'] not in [1,s[1] if rank>1 else 1]:raise ValueError('PReLU num_parameters must be 1 or the channel count.')
        params=p['num_parameters']
    elif name=='RReLU' and p['lower']>p['upper']:raise ValueError('RReLU lower must not exceed upper.')
    elif name=='Hardtanh' and p['min_val']>=p['max_val']:raise ValueError('Hardtanh min_val must be below max_val.')
    elif name in ['Softmax','LogSoftmax','Softmin','GLU']:
        axis=dim_index(p['dim'],rank)
        if name=='GLU':
            if s[axis]%2:raise ValueError('GLU requires an even dimension.')
            s[axis]//=2
    elif name in ['Add','Multiply']:
        if len(inputs)!=2 or inputs[0]!=inputs[1]:raise ValueError('Merge inputs must have identical shapes.')
    elif name=='Concatenate':
        axis=dim_index(p['dim'],rank)
        if len(inputs)!=2 or rank!=len(inputs[1]) or any(a!=b for i,(a,b) in enumerate(zip(s,inputs[1])) if i!=axis):raise ValueError('Concatenation needs matching non-concatenated dimensions.')
        s[axis]+=inputs[1][axis]
    elif name=='Flatten':
        first=dim_index(p['start_dim'],rank);last=dim_index(p['end_dim'],rank)
        if first>last:raise ValueError('Flatten start_dim must not exceed end_dim.')
        s=s[:first]+[math.prod(s[first:last+1])]+s[last+1:]
    elif name=='Reshape':s=[s[0],*inferred_shape(p['shape'],math.prod(s[1:]))]
    elif name=='Unflatten':
        axis=dim_index(p['dim'],rank);s=s[:axis]+inferred_shape(p['unflattened_size'],s[axis])+s[axis+1:]
    elif name=='Permute':
        dims=p['dims']
        if sorted(dims)!=list(range(rank)) or dims[0]!=0:raise ValueError('dims must permute every axis once and preserve batch first.')
        s=[s[i] for i in dims]
    elif name=='Transpose':
        a=dim_index(p['dim0'],rank);b=dim_index(p['dim1'],rank);s[a],s[b]=s[b],s[a]
    elif name=='Unsqueeze':s.insert(dim_index(p['dim'],rank,True),1)
    elif name=='Squeeze':
        if p['dim'] is None:s=[s[0],*[v for v in s[1:] if v!=1]]
        else:
            axis=dim_index(p['dim'],rank)
            if s[axis]==1:s.pop(axis)
    elif name.startswith('Reduce'):
        dims=p['dim'] if isinstance(p['dim'],list) else [p['dim']];dims=[dim_index(d,rank) for d in dims]
        if len(dims)!=len(set(dims)):raise ValueError('Reduction dimensions must be unique.')
        s=[1 if p['keepdim'] and i in dims else x for i,x in enumerate(s) if p['keepdim'] or i not in dims]
    elif name in ['Upsample','Interpolate']:
        d=rank-2
        if d not in [1,2,3] or (p['size'] is None)==(p['scale_factor'] is None):raise ValueError('Resize needs 1-3 spatial axes and exactly one of size or scale_factor.')
        modes={'linear':1,'bilinear':2,'bicubic':2,'trilinear':3}
        if p['mode'] in modes and modes[p['mode']]!=d:raise ValueError('Interpolation mode is incompatible with the spatial rank.')
        if p['mode'] not in modes and p['align_corners'] is not None:raise ValueError('align_corners must be null for nearest/area interpolation.')
        if p.get('antialias') and (d!=2 or p['mode'] not in ['bilinear','bicubic']):raise ValueError('Antialias requires 2D bilinear/bicubic interpolation.')
        if p['size'] is not None:
            if p['recompute_scale_factor']:raise ValueError('recompute_scale_factor requires scale_factor, not size.')
            output=axes(p['size'],d)
        else:
            scale=p['scale_factor'] if isinstance(p['scale_factor'],list) else [p['scale_factor']]*d
            if len(scale)!=d or any(type(v) not in (int,float) or v<=0 for v in scale):raise ValueError('Use one positive scale factor per spatial axis.')
            output=[math.floor(x*y) for x,y in zip(s[2:],scale)]
        s=s[:2]+output
    elif name in ['PixelShuffle','PixelUnshuffle']:
        if rank<4:raise ValueError('Pixel shuffle requires image spatial dimensions.')
        factor=p.get('upscale_factor',p.get('downscale_factor'))
        if name=='PixelShuffle':
            if s[-3]%(factor*factor):raise ValueError('Channels must divide by upscale_factor squared.')
            s[-3]//=factor**2;s[-2]*=factor;s[-1]*=factor
        else:
            if s[-2]%factor or s[-1]%factor:raise ValueError('Spatial dimensions must divide by downscale_factor.')
            s[-3]*=factor**2;s[-2]//=factor;s[-1]//=factor
    elif name=='ChannelShuffle':
        if rank<3 or s[1]%p['groups']:raise ValueError('Channels must divide by shuffle groups.')
    elif 'Pad' in name:
        d=int(name[-2])
        if rank!=d+2:raise ValueError('Padding rank does not match the tensor.')
        pad=axes(p['padding'],2*d)
        for i in range(d):
            axis=rank-1-i;a,b=pad[2*i:2*i+2]
            if name.startswith('Reflection') and (a>=s[axis] or b>=s[axis]):raise ValueError('Reflection padding must be smaller than its input dimension.')
            s[axis]+=a+b
    elif name in ['Unfold','Fold']:
        k=axes(p['kernel_size'],2);dilation=axes(p['dilation'],2);padding=axes(p['padding'],2);stride=axes(p['stride'],2)
        if name=='Unfold':
            if rank!=4:raise ValueError('Unfold needs [B,C,H,W].')
            windows=[(x+2*padding[i]-dilation[i]*(k[i]-1)-1)//stride[i]+1 for i,x in enumerate(s[2:])]
            if any(x<1 for x in windows):raise ValueError('Unfold produces no patches.')
            s=[s[0],s[1]*math.prod(k),math.prod(windows)]
        else:
            size=axes(p['output_size'],2);windows=[(x+2*padding[i]-dilation[i]*(k[i]-1)-1)//stride[i]+1 for i,x in enumerate(size)]
            if rank!=3 or s[1]%math.prod(k) or s[2]!=math.prod(windows) or min(windows)<1:raise ValueError('Fold patch count/channels do not match its kernel and output size.')
            s=[s[0],s[1]//math.prod(k),*size]
    elif name in ['RNN','GRU','LSTM']:
        if rank!=3 or s[-1]!=p['input_size']:raise ValueError('Recurrent layers need [B,L,input_size].')
        h=p['hidden_size'];projection=p.get('proj_size',0)
        if projection and projection>=h:raise ValueError('LSTM proj_size must be smaller than hidden_size.')
        directions=2 if p['bidirectional'] else 1;output=projection or h;gates={'RNN':1,'GRU':3,'LSTM':4}[name]
        for layer in range(p['num_layers']):params+=directions*(gates*h*((p['input_size'] if layer==0 else directions*output)+output+2*int(p['bias']))+(h*projection if projection else 0))
        s=[s[0],s[1],directions*output] if p['output_mode']=='sequence' else [s[0],directions*output]
    elif name in ['SelfAttention','TransformerEncoderLayer']:
        e=p.get('embed_dim',p.get('d_model'));heads=p.get('num_heads',p.get('nhead'))
        if rank!=3 or s[-1]!=e or e%heads:raise ValueError('Attention needs [B,L,embedding] with embedding divisible by heads.')
        if name=='SelfAttention':params=4*e*e+4*e*int(p['bias'])+2*e*int(p['add_bias_kv'])
        else:params=4*e*e+2*e*p['dim_feedforward']+2*e+int(p['bias'])*(7*e+p['dim_feedforward'])
    elif name in ['Dropout1D','Dropout2D','Dropout3D']:
        if rank!=int(name[-2])+2:raise ValueError('Channel dropout rank must match the tensor.')
    elif name=='FeatureAlphaDropout' and rank<2:raise ValueError('FeatureAlphaDropout needs a channel dimension.')
    if any(type(x) is int and x<1 for x in s):raise ValueError('Layer produces a non-positive tensor dimension.')
    if len(s)>8:raise ValueError('At most eight tensor dimensions are supported.')
    return s,params


def make_layer(name,params):
    import torch
    from torch import nn
    from torch.nn import functional as F
    p={**{k:rule['default'] for k,rule in LAYER_SPECS[name]['params'].items()},**params}
    if name in NN_NAMES:return getattr(nn,NN_NAMES[name])(**p)
    class TensorOperation(nn.Module):
        def forward(self,x):
            if name=='Reshape':return x.reshape(x.shape[0],*p['shape'])
            if name=='Permute':return x.permute(*p['dims'])
            if name=='Transpose':return x.transpose(p['dim0'],p['dim1'])
            if name=='Unsqueeze':return x.unsqueeze(p['dim'])
            if name=='Squeeze':
                if p['dim'] is not None:return x.squeeze(p['dim'])
                for axis in range(x.ndim-1,0,-1):
                    if x.shape[axis]==1:x=x.squeeze(axis)
                return x
            if name.startswith('Reduce'):
                dim=tuple(p['dim']) if isinstance(p['dim'],list) else p['dim']
                return {'ReduceMean':torch.mean,'ReduceSum':torch.sum,'ReduceMax':torch.amax}[name](x,dim=dim,keepdim=p['keepdim'])
            if name in ['Upsample','Interpolate']:return F.interpolate(x,**p)
            raise ValueError('UNSUPPORTED_TENSOR_OPERATION: '+name)
    if name in ['RNN','GRU','LSTM']:
        mode=p.pop('output_mode')
        class Recurrent(nn.Module):
            def __init__(self):super().__init__();self.recurrent=getattr(nn,name)(**p)
            def forward(self,x):
                output,state=self.recurrent(x)
                if mode=='sequence':return output
                hidden=state[0] if name=='LSTM' else state;directions=2 if p['bidirectional'] else 1
                return hidden[-directions:].transpose(0,1).reshape(x.shape[0],-1)
        return Recurrent()
    if name=='SelfAttention':
        causal=p.pop('causal')
        class Attention(nn.Module):
            def __init__(self):super().__init__();self.attention=nn.MultiheadAttention(batch_first=True,**p)
            def forward(self,x):
                mask=torch.ones((x.shape[1],x.shape[1]),device=x.device,dtype=torch.bool).triu(1) if causal else None
                return self.attention(x,x,x,attn_mask=mask,need_weights=False)[0]
        return Attention()
    if name not in LAYER_SPECS:raise ValueError('UNSUPPORTED_LAYER: '+name)
    return TensorOperation()
