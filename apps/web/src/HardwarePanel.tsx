import {t} from './i18n';
import React from 'react';
import {Cpu,Activity,Check,AlertCircle} from 'lucide-react';
type Any=any;
export function HardwarePanel({hardware,system,doctor,check,install}:Any){
 const gpus=hardware?.gpus||[];
 const cpuBuild=system?.torch_version?.includes('+cpu');
 const checked=doctor?.engine_python===system?.engine_python;
 const usable=checked&&doctor?.device?.startsWith('cuda');
 return <section className="card hardware-panel"><div className="section-head"><h3><Cpu size={18}/>{t("Thiết bị & khả năng tăng tốc")}</h3><span className={'pill '+(usable?'mint':'')}>{usable?t('GPU sẵn sàng'):cpuBuild?'PyTorch CPU':t('Chưa xác minh GPU')}</span></div>
 <div className="hardware-grid"><div><small>{t("GPU NVIDIA trên máy")}</small><strong>{hardware?gpus.length+t(' thiết bị'):t('Đang kiểm tra…')}</strong></div><div><small>{t("PyTorch của engine")}</small><strong>{system?.torch_version||t('Chưa cài')}</strong></div></div>
 {gpus.map((g:Any)=><div className="gpu-device" key={g.index}><Cpu size={20}/><div><strong>GPU {g.index} · {g.name}</strong><small>{(g.memory_mb/1024).toFixed(1)} GB VRAM · NVIDIA {g.driver}</small></div></div>)}
 {gpus.length>0&&cpuBuild&&<div className="callout"><strong>{t("Đã nhận GPU, cần cài PyTorch có CUDA")}</strong><p>{t("Engine hiện tại là bản CPU nên chưa sử dụng được các GPU này. Cài profile tương thích rồi kiểm tra lại; dự án và dữ liệu được giữ nguyên.")}</p><button className="primary" disabled={system?.setup?.state==='RUNNING'} onClick={()=>install(hardware.recommended_profile)}>{t("Cài CUDA phù hợp")}</button></div>}
 {checked&&<div className={'hardware-result '+(usable?'success-text':'')}>
 {usable?<Check size={18}/>:<AlertCircle size={18}/>}<span>{usable?`${doctor.device_name} · ${doctor.device} ${t('đã chạy phép tính PyTorch')}`:t(doctor.reason||t('Engine đang dùng CPU.'))}</span></div>}
 <button className="secondary" disabled={system?.setup?.state==='RUNNING'} onClick={check}><Activity size={15}/>{t("Kiểm tra máy")}</button>
 <details className="technical-details"><summary>{t("Thông tin kỹ thuật")}</summary><p>Python engine: <code>{system?.engine_python}</code></p>{doctor&&<pre>{JSON.stringify(doctor,null,2)}</pre>}</details>
 </section>
}
