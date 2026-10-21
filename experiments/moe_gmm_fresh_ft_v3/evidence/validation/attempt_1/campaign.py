"""Freeze all six GTdev-selected checkpoints after both adaptive training runs."""
import argparse,json,time
from pathlib import Path
import torch,yaml
from common import ROOT,write,sha,verify
from experiment import rank_top3

def freeze():
    if (ROOT/'freeze_manifest.json').exists():raise FileExistsError('already frozen')
    lock=verify();records=[];sealed={};histories=[];initial=[];runs=[]
    for config in json.loads((ROOT/'order.json').read_text()):
        cfg=yaml.safe_load(Path(config).read_text());folder=Path(cfg['data']['output']);mode=cfg['campaign']['encoder_mode']
        d=json.loads((folder/'completed.json').read_text());h=json.loads((folder/'history.json').read_text());init=json.loads((folder/'initialization.json').read_text())
        if d['status']!='completed' or d['epochs']!=len(h) or d['updates']!=len(h)*162:raise RuntimeError('incomplete training')
        if d['test_records_loaded'] or d['task_checkpoint_loaded'] or init['task_checkpoint_loaded']:raise RuntimeError('fresh training protocol violated')
        if d['version']!='fresh-ft-v3' or d['encoder_mode']!=mode:raise RuntimeError('wrong experiment')
        if any(r['encoder_changed']!=(mode=='fullft') for r in h):raise RuntimeError('encoder update policy mismatch')
        phases=d['phases']
        if [r['stage'] for r in phases]!=['A','M','C']:raise RuntimeError('missing phases')
        expected=[]
        for phase in phases:
            stage=phase['stage'];n=phase['epochs']
            if not cfg['training']['stopping']['min_epochs'][stage]<=n<=cfg['training']['stage_epochs'][stage]:raise RuntimeError('stage budget violated')
            if phase['reason'] not in ('early_stopping','maximum_budget'):raise RuntimeError('unrecorded stopping reason')
            expected+=[stage]*n
        if [r['stage'] for r in h]!=expected:raise RuntimeError('stage history mismatch')
        exact=rank_top3([dict(epoch=r['epoch'],stage_epoch=r['stage_epoch'],miou=r['miou'],checkpoint=str(folder/f'checkpoints/joint_epoch_{r["epoch"]:03d}.pt')) for r in h if r['stage']=='C'])
        if len(exact)!=3 or d['top3']!=exact:raise RuntimeError('not exact GTdev top3')
        if d['best_epoch']!=exact[0]['epoch'] or d['best_miou']!=exact[0]['miou']:raise RuntimeError('rank1 mismatch')
        for rank,record in enumerate(exact,1):
            path=Path(record['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['source_manifest']!=lock or saved['config']!=cfg or saved['stage']!='C' or saved['epoch']!=record['epoch']:raise RuntimeError('checkpoint lock mismatch')
            if saved['metrics']['miou']!=record['miou']:raise RuntimeError('metric mismatch')
            sealed[str(path)]=sha(path);records.append(dict(name=f'{folder.name}_rank{rank}',mode=mode,rank=rank,primary=rank==1,config=config,**record));del saved
        initial.append((init['model_state_sha256'],init['decoder_initial_sha256'],init['moe_initial_sha256']))
        histories.append({(r['stage'],r['stage_epoch']):r['augmentation_sha256'] for r in h});runs.append(d)
    if len(set(initial))!=1:raise RuntimeError('initial parameters differ')
    common=set(histories[0])&set(histories[1])
    if any(histories[0][k]!=histories[1][k] for k in common):raise RuntimeError('common stage/epoch augmentation differs')
    write(ROOT/'freeze_manifest.json',dict(checkpoints=sealed,records=records,runs=runs,source_manifest=lock,all_runs_completed=True,
        common_fresh_initialization=initial[0],common_stage_epoch_augmentation_matched=len(common),primary='GTdev rank1 per mode; ranks2/3 secondary; no CDL reranking',threshold=.5,created_at=time.time()))
    print('both fresh runs and all SIX top3 checkpoints frozen',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('freeze',));ap.parse_args();freeze()
