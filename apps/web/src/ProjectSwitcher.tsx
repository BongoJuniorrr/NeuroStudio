import {t} from './i18n';
import React,{useEffect,useRef,useState} from 'react';
import {Check,ChevronDown,Search} from 'lucide-react';
type Any=any;
export function ProjectSwitcher({projects,value,onChange,disabled}:Any){
 const [open,setOpen]=useState(false),[query,setQuery]=useState(''),[index,setIndex]=useState(0);
 const root=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null);
 const items=projects.filter((p:Any)=>p.project.name.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
 const selected=projects.find((p:Any)=>p.id===value);
 useEffect(()=>{const close=(e:PointerEvent)=>{if(!root.current?.contains(e.target as Node))setOpen(false)};document.addEventListener('pointerdown',close);return()=>document.removeEventListener('pointerdown',close)},[]);
 const choose=(id:string)=>{setOpen(false);setQuery('');onChange(id);trigger.current?.focus()};
 return <div className="project-switcher" ref={root} onKeyDown={e=>{
  if(e.key==='Escape'){setOpen(false);trigger.current?.focus();e.preventDefault()}
  if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();if(!open){setOpen(true);setIndex(Math.max(0,items.findIndex((p:Any)=>p.id===value)));return}setIndex(i=>Math.max(0,Math.min(items.length-1,i+(e.key==='ArrowDown'?1:-1))))}
  if(e.key==='Enter'&&open&&items[index]){e.preventDefault();choose(items[index].id)}
 }}><button className="project-trigger" ref={trigger} role="combobox" aria-label={t("Chọn dự án")} aria-expanded={open} aria-haspopup="listbox" aria-controls="project-options" aria-activedescendant={open&&items[index]?'project-option-'+items[index].id:undefined} disabled={disabled} title={selected?.project.name} onClick={()=>{setOpen(!open);setIndex(Math.max(0,items.findIndex((p:Any)=>p.id===value)))}}><span>{selected?.project.name||t('Chọn dự án')}</span><ChevronDown size={16}/></button>
 {open&&<div className="project-popover"><div className="search"><Search size={16}/><input aria-label={t("Tìm dự án")} placeholder={t("Tìm dự án…")} value={query} onChange={e=>{setQuery(e.target.value);setIndex(0)}}/></div><div role="listbox" id="project-options" aria-label={t("Danh sách dự án")}>{items.map((p:Any,i:number)=><button role="option" id={'project-option-'+p.id} aria-selected={p.id===value} className={i===index?'focused':''} key={p.id} onMouseEnter={()=>setIndex(i)} onClick={()=>choose(p.id)}><span><strong>{p.project.name}</strong><small>{p.project.backend==='torch'?t('PyTorch · Ảnh'):t('scikit-learn · Bảng')}</small></span>{p.id===value&&<Check size={16}/>}</button>)}{!items.length&&<p className="fine">{t("Không tìm thấy dự án.")}</p>}</div></div>}
 </div>
}
