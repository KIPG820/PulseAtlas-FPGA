"""Retrain from general YOLO11n; select deployment top-1 macro-F1 on temporal all-class val."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import yaml


def macro_f1(metrics):
    scores=[]
    for row in metrics['per_class'].values():
        scores.append(2*row['tp']/max(1,2*row['tp']+row['fp']+row['fn']))
    return sum(scores)/len(scores)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pulseatlas_rebuilt_v2.yaml')
    p.add_argument('--log',default='artifacts/pulseatlas/rebuilt_v2/train.log')
    p.add_argument('--resume')
    a=p.parse_args()
    output=Path('artifacts/pulseatlas/rebuilt_v2').resolve()
    output.mkdir(parents=True,exist_ok=True)
    sys.stdout=sys.stderr=Path(a.log).open('a',encoding='utf-8',buffering=1)
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path('artifacts/ultralytics').resolve()))
    os.environ.setdefault('MPLCONFIGDIR',str(Path('artifacts/matplotlib').resolve()))
    from ultralytics import YOLO
    import torch
    from ultralytics.utils.torch_utils import strip_optimizer
    from pulseatlas_audit import sha256,NAMES
    from evaluate_pulseatlas import collect,operating_metrics
    report=json.loads((output/'dataset_report.json').read_text())
    dataset=Path(report['dataset'])
    cfg=yaml.safe_load(Path(a.config).read_text())
    initial=Path(cfg.pop('model')).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    metadata=dict(status='running',pid=os.getpid(),started_unix=time.time(),dataset=str(dataset),
        config=cfg,initial=str(initial),initial_sha256=sha256(initial),
        first_stage_changed=False,selection='Internal temporal all-ten-class top-1 macro-F1 at conf .10 / NMS IoU .60; legacy sets excluded')
    (output/'training_run.json').write_text(json.dumps(metadata,indent=2))
    model=YOLO(a.resume or str(initial))
    best={'macro_f1':-1.0}
    history=[]
    last_reviewed=0
    if a.resume and (output/'deployment_selection.json').exists():
        previous=json.loads((output/'deployment_selection.json').read_text())
        best=previous['best']
        history=previous['history']

    def progress(trainer):
        nonlocal best,last_reviewed
        epoch=trainer.epoch+1
        status=dict(epoch=epoch,max_epochs=trainer.epochs,save_dir=str(trainer.save_dir),
            metrics={k:float(v) for k,v in trainer.metrics.items()},updated_unix=time.time())
        (output/'progress.json').write_text(json.dumps(status,indent=2))
        if epoch==last_reviewed or (epoch!=1 and epoch%5 and not trainer.stop):
            return
        if not Path(trainer.last).exists():
            return
        probe=YOLO(str(trainer.last))
        if dict(probe.names)!=dict(enumerate(NAMES)):
            raise RuntimeError('Incorrect class mapping')
        cases=collect(probe,dataset,'internal_val',960,.6)
        metrics=operating_metrics(cases,.1)
        record=dict(epoch=epoch,macro_f1=macro_f1(metrics),metrics=metrics,reviewed_unix=time.time())
        history.append(record)
        if (record['macro_f1'],metrics['f1'])>(best['macro_f1'],best.get('metrics',{}).get('f1',-1)):
            best=record
            shutil.copy2(trainer.last,output/'deployment_best_full.pt')
        (output/'deployment_selection.json').write_text(json.dumps(dict(best=best,history=history),indent=2))
        print('DEPLOYMENT_VAL '+json.dumps(record),flush=True)
        last_reviewed=epoch
        del probe
        gc.collect()
        torch.cuda.empty_cache()
    model.add_callback('on_fit_epoch_end',progress)
    try:
        if a.resume:
            model.train(resume=True,data=str(dataset/'data.yaml'))
        else:
            model.train(data=str(dataset/'data.yaml'),**cfg)
        if not (output/'deployment_best_full.pt').exists():
            raise RuntimeError('No deployment-metric selected checkpoint')
        strip_optimizer(str(output/'deployment_best_full.pt'),s=str(output/'electrode_rebuilt_v2.pt'))
        poses={p.name:sha256(p) for p in (Path(report['pose_package'])/'models/pose').iterdir() if p.is_file()}
        if poses!=report['pose_sha256']:
            raise RuntimeError('First-stage assets changed')
        metadata.update(status='finished',finished_unix=time.time(),save_dir=str(model.trainer.save_dir),
            actual_epochs=int(model.trainer.epoch+1),
            selected_epoch=best['epoch'],internal_val_macro_f1=best['macro_f1'],pose_hashes_unchanged=True,
            weights_sha256=sha256(output/'electrode_rebuilt_v2.pt'))
        (output/'training_run.json').write_text(json.dumps(metadata,indent=2))
        print('REBUILT TRAINING FINISHED '+json.dumps(metadata),flush=True)
    except Exception as error:
        metadata.update(status='failed',error=repr(error),failed_unix=time.time())
        (output/'training_run.json').write_text(json.dumps(metadata,indent=2))
        raise


if __name__=='__main__':
    main()
