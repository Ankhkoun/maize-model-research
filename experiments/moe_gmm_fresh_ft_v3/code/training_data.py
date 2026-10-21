"""Train-only snapshot evidence refresh and spatially aligned frozen target batches."""
from __future__ import annotations
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import yaml
from common import ROOT,write,sha,state_sha,setup,seed_int,make_model,loader,forward,save,verify
from fusion import evidence,classification_ce,choose_target
from objectives import reliability_loss
from configuration import validate_config,alpha_at
from source_snapshot.data import USAAssets,check_protocol,load_rows,windows_to_tile_grid,spatial_transform
from source_snapshot.train import _move,_gtdev_metrics,_validate_gtdev_assets
from experiment import stage_at,policy,parameter_groups

@torch.inference_mode()
def refresh(model,rows,assets,cfg,previous,epoch,out,target_kind):
    model.eval();bank={};report=[];start=time.time();count=0;gcount=0;delta=[];no_observed=0
    t,p=cfg['training'],cfg['campaign']
    for step,raw in enumerate(loader(rows,assets,cfg,epoch,False),1):
        b=_move(raw,torch.device('cuda'));pred=forward(model,b,True)
        prior=windows_to_tile_grid(pred['reliability'].float())
        observed=windows_to_tile_grid((b['observed_support']>0).any(-1))
        e,rec=evidence(pred['logits'],b['target'],prior,observed,t,p)
        for index,sid in enumerate(raw['sample_id']):
            single={k:v[index] for k,v in e.items()}
            target,target_valid=choose_target(single,target_kind,previous[sid] if previous else None,float(t['ema']))
            bank[sid]={k:single[k].cpu() for k in ('qg','qf','prior','valid','gmm')}
            bank[sid].update(target=target.cpu(),target_valid=target_valid.cpu())
            report.append(dict(sample_id=sid,classes=[r for r in rec if r['batch']==index]))
        count+=int(e['valid'].sum());gcount+=int(e['gmm'].sum())
        no_observed+=int((e['label_valid']&~e['observed']).sum())
        if bool(e['gmm'].any()):delta.append(float((e['qf']-e['qg']).abs()[e['gmm']].mean()))
        if step==1 or step%50==0 or len(bank)==len(rows):
            write(out/'progress.json',dict(event='target_refresh',epoch=epoch,step=len(bank),total=len(rows),time=time.time()))
            print(json.dumps(dict(event='target_refresh',epoch=epoch,step=len(bank),total=len(rows))),flush=True)
    summary=dict(samples=len(bank),eligible_patches=count,gmm_patches=gcount,
        fused_absolute_difference=float(np.mean(delta)) if delta else 0.,seconds=time.time()-start,
        eligible_without_direct_observation=no_observed,moe_target=target_kind,target_ema=t['ema'],
        test_records_loaded=0,source='unaugmented Train ordinary CE; same snapshot MoE; raw Gaussian PDFs',
        fallback='unavailable GMM: qstar=r; GMM CE unit weight; BCE skipped; no rank',per_tile=report)
    write(out/f'evidence_epoch_{epoch:02d}.json',summary)
    return bank,{k:v for k,v in summary.items() if k!='per_tile'}

def bank_batch(raw,bank,device):
    transforms=raw.get('spatial_transform',[(0,False,False)]*len(raw['sample_id']))
    return {key:torch.stack([spatial_transform(bank[sid][key],transform)
        for sid,transform in zip(raw['sample_id'],transforms)]).to(device)
        for key in ('target','target_valid','valid','gmm','qg','qf','prior')}

