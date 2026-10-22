"""Bounded, sequential classical-ML studies. Validation/test never rank trials."""
from pathlib import Path
import time
from ..util import atomic_json
from .portable import cross_validate_training


def study(task,out,emit,control):
    began=time.monotonic();deadline=began+task['budget_seconds']
    report=dict(mode=task['mode'],estimator=task['estimator'],metric=task['metric'],fold_count=task['folds'],trials=[],best=None,budget_seconds=task['budget_seconds'],budget_exhausted=False,source_run=task['source_run'])
    def save():
        report['elapsed_seconds']=time.monotonic()-began
        atomic_json(Path(out)/'result.json',report)
        atomic_json(Path(task['run_directory'])/'study.json',report)
    for index,candidate in enumerate(task['candidates']):
        control()
        if time.monotonic()>=deadline:report['budget_exhausted']=True;break
        try:
            result=cross_validate_training(candidate,task['dataset'],task['folds'],control,deadline)
            report.setdefault('source_train_indices',result.pop('source_train_indices'))
            parameters=next(g for g in candidate['graphs'] if g['kind']=='model')['nodes'][0]['params']
            trial=dict(index=index+1,state='COMPLETED',parameters=parameters,**result)
            report['trials'].append(trial)
            if report['best'] is None or trial[task['metric']]>report['best'][task['metric']]:report['best']=trial
            emit('study_trial',dict(index=index+1,score=trial[task['metric']],metric=task['metric']))
        except TimeoutError:report['budget_exhausted']=True;save();break
        except ValueError as e:
            report['trials'].append(dict(index=index+1,state='FAILED',error=str(e)))
        save()
    save()
    if not report['best'] and not report['budget_exhausted']:raise ValueError('STUDY_FAILED: '+report['trials'][0].get('error','Không có trial hợp lệ.'))
    return report
