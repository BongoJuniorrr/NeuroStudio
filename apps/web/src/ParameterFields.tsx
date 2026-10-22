import {t} from './i18n';
import React,{useEffect,useState} from 'react';
type Any=any;
export function ParameterFields({rules,values,onChange,prefix=''}:Any){
 return <div className="parameter-fields">{Object.entries(rules||{}).map(([name,rule]:[string,Any])=><ParameterField key={prefix+name} name={name} rule={rule} value={Object.prototype.hasOwnProperty.call(values,name)?values[name]:rule.default} onChange={(value:Any)=>onChange(name,value)}/>)}</div>
}
function ParameterField({name,rule,value,onChange}:Any){
 const json=Array.isArray(rule.default)||rule.types?.some((kind:string)=>['list','NoneType','str'].includes(kind));
 const format=(v:Any)=>json?JSON.stringify(v):String(v);
 const [text,setText]=useState(format(value));
 useEffect(()=>{setText(format(value))},[JSON.stringify(value)]);
 const hint=<>{rule.description&&<small>{t(rule.description)}</small>}{json&&<small>{t('JSON: dùng số, array hoặc null theo kiểu tham số.')}</small>}</>;
 if(typeof rule.default==='boolean')return <label className="check"><input aria-label={name} type="checkbox" checked={!!value} disabled={rule.options?.length===1} onChange={e=>onChange(e.target.checked)}/>{name}{hint}</label>;
 if(rule.options)return <label>{name}<select aria-label={name} value={value} onChange={e=>onChange(e.target.value)}>{rule.options.map((option:string)=><option key={option} value={option}>{option}</option>)}</select>{hint}</label>;
 return <label>{name}<input aria-label={name} type={json?'text':'number'} value={text} min={rule.minimum??undefined} max={rule.maximum??undefined} step={rule.type==='int'?1:'any'} spellCheck={false} onChange={e=>{
  const raw=e.target.value;setText(raw);e.target.setCustomValidity('');
  if(json){try{const parsed=JSON.parse(raw),kind=parsed===null?'NoneType':Array.isArray(parsed)?'list':typeof parsed==='number'?(Number.isInteger(parsed)?'int':'float'):typeof parsed==='boolean'?'bool':'str';const allowed=rule.types||['list'];if(!allowed.includes(kind)&&!(kind==='int'&&allowed.includes('float')))throw Error();if(typeof parsed==='number'&&((rule.minimum!=null&&parsed<rule.minimum)||(rule.maximum!=null&&parsed>rule.maximum)))throw Error();onChange(parsed)}catch{e.target.setCustomValidity(t('Nhập JSON đúng kiểu và giới hạn của tham số.'))}return}
  if(raw.trim()==='')return;
  const parsed=Number(raw);if(Number.isFinite(parsed)&&e.target.validity.valid)onChange(parsed);
 }} onBlur={e=>{if(!e.target.checkValidity())e.target.reportValidity()}} required/>{hint}{!json&&(rule.minimum!=null||rule.maximum!=null)&&<small>{rule.minimum??'-∞'} … {rule.maximum??'∞'}</small>}</label>
}
export function ModelSettings({draft,edit,registry}:Any){
 const graph=draft.graphs.find((g:Any)=>g.kind==='model');
 const [selected,setSelected]=useState(graph.nodes.find((n:Any)=>Object.keys(registry.find((r:Any)=>r.type===n.type)?.params||{}).length)?.id||graph.nodes[0]?.id);
 const node=graph.nodes.find((n:Any)=>n.id===selected)||graph.nodes[0];
 const spec=registry.find((r:Any)=>r.type===node?.type);
 const estimators=registry.filter((r:Any)=>r.backend==='sklearn'&&r.kind==='model'&&r.task===draft.task);
 const replace=(type:string)=>{const next=registry.find((r:Any)=>r.type===type);edit((p:Any)=>{const model=p.graphs.find((g:Any)=>g.kind==='model');const original=model.nodes[0];model.nodes=[{...original,type,version:1,params:Object.fromEntries(Object.entries(next.params).map(([k,v]:[string,Any])=>[k,v.default]))}];model.edges=[]});};
 return <section className="model-settings"><h3>{t("Tham số mô hình")}</h3><p className="fine">{t("Thay đổi tại đây được lưu vào project và bộ Python export. Cấu hình của run đã hoàn tất được giữ riêng.")}</p>
 {draft.backend==='sklearn'?<label>{t("Thuật toán")}<select aria-label={t("Thuật toán")} value={node?.type} onChange={e=>replace(e.target.value)}>{estimators.map((r:Any)=><option key={r.type}>{r.type}</option>)}</select></label>:<label>{t("Lớp cần cấu hình")}<select aria-label={t("Lớp cần cấu hình")} value={node?.id||''} onChange={e=>setSelected(e.target.value)}>{graph.nodes.map((n:Any)=><option key={n.id} value={n.id}>{n.type} · {n.id}</option>)}</select></label>}
 {spec&&node&&<><p className="fine">{t(spec.description)}</p><ParameterFields prefix={node.id+node.type} rules={spec.params} values={node.params} onChange={(key:string,value:Any)=>edit((p:Any)=>{p.graphs.find((g:Any)=>g.kind==='model').nodes.find((n:Any)=>n.id===node.id).params[key]=value})}/>{!Object.keys(spec.params).length&&<p className="fine">{t("Block này không có tham số cấu hình.")}</p>}</>}
 </section>
}
