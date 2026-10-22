import messages from './messages.en.json';
export type Language='vi'|'en';
let current:Language=localStorage.getItem('neurostudio-language')==='vi'?'vi':'en';
document.documentElement.lang=current;
export function getLanguage(){return current}
export function setLanguagePreference(value:Language){current=value;localStorage.setItem('neurostudio-language',value);document.documentElement.lang=value}
const catalog=messages as Record<string,string>;
export function t(value:string):string{
 if(typeof value!=='string')return value;
 value=value.replace(/[\u2013\u2014]/g,'-');
 if(current!=='en')return value;
 if(catalog[value]!==undefined)return catalog[value];
 if(value.startsWith('Value error, '))return 'Value error, '+t(value.slice(13));
 const code=value.match(/^([A-Z][A-Z0-9_]+):/);if(code&&catalog['error.'+code[1]])return code[1]+': '+catalog['error.'+code[1]];
 const patterns:[RegExp,(...args:any[])=>string][]=[
 [/^(\d+) tác vụ đang chờ \/ chạy$/,(_,n)=>`${n} queued / running jobs`],
 [/^Có (\d+) dòng trùng; cần xử lý trước train để tránh overlap\.$/,(_,n)=>`${n} duplicate rows; resolve them before training to avoid split overlap.`],
 [/^Giữ nguyên 10.000 ảnh test chính thức. train\/validation chia trong 50.000 ảnh train\.$/,()=> 'The 10,000 official test images are reserved. Training and validation use the 50,000 training images.'],
 [/^CUDA không khả dụng trong PyTorch environment này\.$/,()=> 'CUDA is unavailable in this PyTorch environment.']
 ];
 for(const [pattern,render] of patterns)if(pattern.test(value))return value.replace(pattern,render);
 return value;
}

const humanKeys=new Set(['message','error','reason','fallback_reason','warnings','warning','detail','note','msg']);
export function localizePayload(value:any,key=''):any{
 if(typeof value==='string')return humanKeys.has(key)?t(value):value;
 if(Array.isArray(value))return value.map(v=>localizePayload(v,key));
 if(value&&typeof value==='object'){const result=Object.fromEntries(Object.entries(value).map(([k,v])=>[k,localizePayload(v,k)]));if(current==='en'&&typeof value.code==='string'&&graphMessages[value.code]&&typeof value.message==='string')result.message=catalog[value.message]||graphMessages[value.code];return result;}
 return value;
}

const graphMessages:Record<string,string>={TASK_BACKEND:'Choose scikit-learn for tabular regression.',GRAPH_KINDS:'Use exactly one data graph, model graph and experiment graph.',DUPLICATE_ID:'Block IDs must be unique.',ORPHAN_EDGE:'A connection refers to a missing block.',PORT:'Connect an output port to an input port.',DUPLICATE_EDGE:'This connection already exists.',CYCLE:'The graph contains a cycle. Remove a connection in that cycle.',BLOCK_VERSION:'This block or block version is unsupported.',PORT_TYPE:'The block belongs to a different graph type.',BACKEND:'Choose blocks compatible with the project backend.',TASK_ESTIMATOR:'Choose an estimator compatible with this task.',UNKNOWN_PARAMETER:'The block contains an unsupported parameter.',PARAMETER:'A block parameter has an invalid type or is outside its allowed range.',ARITY:'The block has an incorrect number of incoming connections.',DATA_ROOT:'Include one data source and one split step.',DATA_BRANCH:'Use a sequential data preprocessing pipeline.',LEAKAGE:'Place Split before fitted transforms. Fit transforms on training data only.',DUPLICATE_TRANSFORM:'Use each preprocessing transform once.',TRANSFORM_ORDER:'Connect preprocessing in this order: Imputer, Scaler, OneHotEncoder, PCA.',SCALER:'Choose one scaler.',MODALITY:'Choose data and preprocessing compatible with the project.',EXPERIMENT:'Connect Train, Evaluate and Export in that order.',ESTIMATOR:'Choose exactly one estimator.',MODEL_BOUNDARY:'Include exactly one Input and one Output.',UNUSED_NODE:'This block does not lead to Output. Connect or remove it.',LOSS_HEAD:'Use logits as classification output. Remove Softmax before cross entropy.',SHAPE:'Block dimensions are incompatible. Inspect tensor shapes and block parameters.',MODEL_LIMIT:'The model exceeds the supported parameter budget.'};
