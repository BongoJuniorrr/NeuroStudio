import {t} from './i18n';
import React from 'react';
import {TrainingObjectives} from './TrainingObjectives';
import {ModelSettings,ParameterFields} from './ParameterFields';
type Any=any;
const number=(value:number,min:number,max:number,description='')=>({default:value,minimum:min,maximum:max,description,type:Number.isInteger(value)&&Number.isInteger(min)&&Number.isInteger(max)?'int':'float'});
const choice=(value:string,options:string[])=>({default:value,options});
export function TrainingSettings({draft,edit,registry,gpus}:Any){
 const settings=draft.training;
 const update=(key:string,value:Any)=>edit((p:Any)=>{p.training[key]=value});
 const common={seed:number(42,0,2147483647),timeout_seconds:number(3600,10,604800)};
 const base={epochs:number(5,1,10000),batch_size:number(64,1,4096),learning_rate:number(.001,.000001,10),optimizer:choice('AdamW',['SGD','Adam','AdamW','RMSprop','Adagrad']),weight_decay:number(.0001,0,1)};
 let optimizer:Any={};
 if(['SGD','RMSprop'].includes(settings.optimizer))optimizer.momentum=number(.9,0,.999);
 if(settings.optimizer==='SGD')Object.assign(optimizer,{dampening:number(0,0,1),nesterov:{default:false}});
 if(['Adam','AdamW'].includes(settings.optimizer))Object.assign(optimizer,{adam_beta1:number(.9,0,.999999),adam_beta2:number(.999,0,.999999),optimizer_eps:number(1e-8,1e-12,1),amsgrad:{default:false}});
 if(settings.optimizer==='RMSprop')Object.assign(optimizer,{rmsprop_alpha:number(.99,0,.999999),optimizer_eps:number(1e-8,1e-12,1),centered:{default:false}});
 if(settings.optimizer==='Adagrad')optimizer.optimizer_eps=number(1e-8,1e-12,1);
 const scheduler:Any={scheduler:choice('none',['none','StepLR','CosineAnnealingLR','ExponentialLR','ReduceLROnPlateau'])};
 if(['StepLR','ExponentialLR','ReduceLROnPlateau'].includes(settings.scheduler))scheduler.gamma=number(.5,.00001,settings.scheduler==='ReduceLROnPlateau'?.99999:1);
 if(settings.scheduler==='StepLR')scheduler.step_size=number(5,1,10000);
 if(['CosineAnnealingLR','ReduceLROnPlateau'].includes(settings.scheduler))scheduler.min_lr=number(0,0,settings.learning_rate);
 if(settings.scheduler==='ReduceLROnPlateau')scheduler.scheduler_patience=number(3,0,1000);
 return <section className="card config training-settings"><h3>{t("Cấu hình huấn luyện")}</h3><ModelSettings draft={draft} edit={edit} registry={registry}/><TrainingObjectives config={settings} task={draft.task} update={update} hideLoss={draft.backend==='sklearn'}/>
 {draft.backend==='torch'?<><details open><summary>{t("Optimizer & tốc độ học")}</summary><ParameterFields prefix="training" rules={base} values={settings} onChange={update}/><ParameterFields prefix="optimizer" rules={optimizer} values={settings} onChange={update}/></details><details open><summary>{t("Scheduler & dừng sớm")}</summary><ParameterFields rules={scheduler} values={settings} onChange={update}/><ParameterFields rules={{patience:number(5,1,1000),grad_clip:number(1,.00001,100),label_smoothing:number(0,0,.999)}} values={settings} onChange={update}/></details><label>{t("Thiết bị")}<select aria-label={t("Thiết bị")} value={settings.device} onChange={e=>update('device',e.target.value)}><option value="auto">{t("Auto · ưu tiên GPU")}</option><option value="cpu">CPU</option>{(gpus.length?gpus:[0]).map((i:number)=><option key={i} value={'cuda:'+i}>CUDA GPU {i}</option>)}</select></label><ParameterFields rules={{precision:choice('fp32',['fp32','amp']),gpu_memory_fraction:number(.2,.05,.9,t('Tỷ lệ VRAM tối đa cho tiến trình này; ví dụ 0.2 = 20%. Giảm batch size nếu vượt ngân sách.'))}} values={settings} onChange={update}/><details><summary>{t("Hiệu năng & tái lập")}</summary><ParameterFields rules={{...common,cpu_threads:number(2,1,64),loader_workers:number(0,0,8),drop_last:{default:false,description:t('Bỏ batch train cuối nếu thiếu mẫu; hữu ích cho BatchNorm trên vector. Validation/test giữ đủ mẫu.')},deterministic:{default:true}}} values={settings} onChange={update}/></details></>:<><div className="callout"><p>{t("Thuật toán scikit-learn dùng CPU. Các tham số ở trên điều khiển mô hình trực tiếp; không áp dụng optimizer hoặc epoch của mạng neural.")}</p></div><ParameterFields rules={common} values={settings} onChange={update}/></>}
 </section>
}
