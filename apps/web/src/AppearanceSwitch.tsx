import React from 'react';
import {Sun,Moon,Contrast} from 'lucide-react';
import {t} from './i18n';
export function AppearanceSwitch({value,onChange}:{value:string,onChange:(value:string)=>void}){
 const themes=[{id:'light',label:t('Giao diện sáng'),Icon:Sun},{id:'dark',label:t('Giao diện tối'),Icon:Moon},{id:'contrast',label:t('Tương phản cao'),Icon:Contrast}];
 return <div className="appearance-switch" role="group" aria-label={t('Giao diện')}>{themes.map(({id,label,Icon})=><button type="button" key={id} className={value===id?'selected':''} aria-label={t(label)} title={t(label)} aria-pressed={value===id} onClick={()=>onChange(id)}><Icon size={17}/></button>)}</div>;
}
