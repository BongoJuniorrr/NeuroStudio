"""Read physical NVIDIA inventory without importing the ML runtime."""
import csv
import io
import subprocess
from .setup import recommend_profile


def hardware_inventory():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version,compute_cap','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
        gpus=[]
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row)!=5:continue
            index,name,memory,driver,capability=[part.strip() for part in row]
            gpus.append(dict(index=int(index),name=name,memory_mb=int(memory),driver=driver,capability=capability))
        profile=recommend_profile('\n'.join(f"{g['capability']},{g['driver']}" for g in gpus))
        return dict(gpus=gpus,recommended_profile=profile,error=None)
    except (OSError,subprocess.SubprocessError,ValueError) as e:
        return dict(gpus=[],recommended_profile='cpu',error=str(e))


def system_resources():
    import psutil
    memory=psutil.virtual_memory();gpus=[];error=None
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row)==5:
                index,name,utilization,used,total=[v.strip() for v in row]
                gpus.append(dict(index=int(index),name=name,utilization_percent=float(utilization),memory_used_mb=float(used),memory_total_mb=float(total)))
    except (OSError,ValueError,subprocess.SubprocessError) as e:error=str(e)
    return dict(cpu_percent=psutil.cpu_percent(interval=0.1),ram=dict(used_gb=round(memory.used/1024**3,2),total_gb=round(memory.total/1024**3,2),percent=memory.percent),gpus=gpus,gpu_error=error,scope='Whole machine; not attributed exclusively to Neuro Studio.')
