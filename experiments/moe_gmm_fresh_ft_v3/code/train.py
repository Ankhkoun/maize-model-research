"""Fresh two-mode task training with staged monitoring and GTdev-selected top3."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch,yaml
from torch.nn import functional as F
from common import ROOT,write,sha,state_sha,setup,seed_int,make_model,loader,forward,save,verify
from experiment import Progress,policy,parameter_groups,rank_top3
from configuration import validate_config
from training_data import refresh,bank_batch
from fusion import classification_ce
from objectives import reliability_loss
from source_snapshot.data import USAAssets,check_protocol,load_rows,windows_to_tile_grid
from source_snapshot.train import _move,_gtdev_metrics,_validate_gtdev_assets

def encoder_sha(model):
    return state_sha({k:v for k,v in model.state_dict().items() if k.startswith(('classifier.patch_embed.','classifier.backbone.'))})

def encoder_probe(model):
    parts=dict(patch_embed=model.classifier.patch_embed,**{f'swin_{i}':v for i,v in enumerate(model.classifier.backbone.layers)})
    return {key:dict(trainable=sum(x.numel() for x in module.parameters() if x.requires_grad),
        grad_norm=sum(float(x.grad.float().square().sum()) for x in module.parameters() if x.grad is not None)**.5) for key,module in parts.items()}

@torch.inference_mode()
def fixed_probe(model,rows,assets,cfg,targets):
    model.eval();loss=0.;entropy=0.;count=0
    for raw in loader(rows,assets,cfg,70000,False):
        b=_move(raw,torch.device('cuda'));pred=forward(model,b,True)
        r=windows_to_tile_grid(pred['reliability'].float()).clamp(cfg['campaign']['eps'],1-cfg['campaign']['eps'])
        for i,sid in enumerate(raw['sample_id']):
            q=targets[sid]['target'].to(r);mask=targets[sid]['target_valid'].to(r.device)
            loss+=float(F.binary_cross_entropy(r[i][mask],q[mask],reduction='sum'))
            qc=q[mask].clamp(1e-7,1-1e-7);entropy+=float((-(qc*qc.log()+(1-qc)*(1-qc).log())).sum());count+=int(mask.sum())
    return dict(bce=loss/count if count else None,entropy=entropy/count if count else None,
        excess_bce=(loss-entropy)/count if count else None,valid_patches=count,tiles=len(rows),
        meaning='fixed Train qG target fitting diagnostic, not correctness accuracy')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg=yaml.safe_load(a.config.read_text());validate_config(cfg);lock=verify();t=cfg['training'];p=cfg['campaign'];stop=t['stopping'];mode=p['encoder_mode'];seed=t['seed']
    if torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('one N16R4 4090 required')
    out=Path(cfg['data']['output']+('_smoke' if a.smoke else ''))
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'checkpoints').mkdir();write(out/'config.json',cfg);setup(seed)
    assets_dir=Path(cfg['data']['protocol_assets']);protocol=check_protocol(assets_dir,include_test=False)
    rows=[x for x in load_rows(assets_dir/'train_validation500.csv') if x['split']=='train'];dev=load_rows(assets_dir/'gtdev60.csv')
    if len(rows)!=1296 or len(dev)!=60 or {x['sample_id'] for x in rows}&{x['sample_id'] for x in dev}:raise RuntimeError('split mismatch')
    assets=USAAssets(Path(cfg['data']['root']),assets_dir/'normalization_train1296.json');_validate_gtdev_assets(assets,dev)
    if sha(cfg['data']['pretrained'])!=p['pretrained_sha256']:raise RuntimeError('pretrain changed')
    model=make_model(cfg);initial=state_sha(model.state_dict());initial_encoder=encoder_sha(model)
    init=dict(initialization=p['initialization'],task_checkpoint_loaded=False,origin_epoch=0,seed=seed,encoder_mode=mode,
        model_state_sha256=initial,initial_encoder_sha256=initial_encoder,decoder_initial_sha256=state_sha(model.classifier.decoder.state_dict()),
        moe_initial_sha256=state_sha(model.reliability.state_dict()),pretrained_sha256=p['pretrained_sha256'],pretrained=cfg['data']['pretrained'],
        load_report=model.classifier.load_report,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        optimizer='reset at A/M/C boundaries',encoder_objective='CE and existing F1-context reliability path when fullft; targets/weights detached',test_records_loaded=0)
    write(out/'initialization.json',init)
    if not a.smoke:write(out/'baseline_gtdev.json',_gtdev_metrics(model,assets,dev,torch.device('cuda'),threshold=.5,reliability_enabled=False))
    else:rows=rows[:4]
    history=[];phase_records=[];top=[];epoch=0;probe_targets={};probe_rows=[];smoke=[]
    for phase_index,stage in enumerate(('A','M','C')):
        counts=model.set_stage(stage);groups=parameter_groups(model);rates=dict(encoder=t['encoder_lr'],decoder=t['decoder_lr'],reliability=t['reliability_lr'])
        optimizer=torch.optim.AdamW([dict(params=v,lr=rates[k],name=k) for k,v in groups.items() if v],weight_decay=t['weight_decay'])
        write(out/f'optimizer_stage_{stage}.json',dict(counts=counts,learning_rates=rates,reset=True))
        dev_track=Progress(stop['dev_min_delta'],stop['min_epochs'][stage],stop['patience'][stage])
        probe_track=Progress(stop['probe_min_delta'],stop['min_epochs'][stage],stop['patience'][stage],maximize=False)
        bank=None;phase_best=None;phase_best_path=out/f'stage_{stage}_best.pt';reason='maximum_budget';initial_probe=None
        maximum=1 if a.smoke else int(t['stage_epochs'][stage])
        for local in range(1,maximum+1):
            epoch+=1;start=time.time();rules=policy(stage,local,t);data_epoch=(phase_index+1)*1000+local
            setup(seed_int(seed,data_epoch,'refresh')%(2**32-1));bank_report=dict(moe_target=None,samples=0)
            if rules['reliability']:
                bank,bank_report=refresh(model,rows,assets,cfg,bank,epoch,out,rules['target'])
                if stage=='M' and local==1:
                    available=[r for r in rows if bank[r['sample_id']]['target_valid'].any()]
                    probe_rows=sorted(available,key=lambda r:seed_int(seed,0,r['sample_id']))[:stop['probe_tiles']]
                    probe_targets={r['sample_id']:{k:bank[r['sample_id']][k].clone() for k in ('target','target_valid')} for r in probe_rows}
                    save(out/'moe_fixed_train_probe.pt',dict(targets=probe_targets,sample_ids=[r['sample_id'] for r in probe_rows],source='qG frozen at M entry, Train only'))
                    initial_probe=fixed_probe(model,probe_rows,assets,cfg,probe_targets);write(out/'moe_probe_initial.json',initial_probe)
            setup(seed_int(seed,data_epoch,'model')%(2**32-1));model.train();optimizer.zero_grad(set_to_none=True)
            trace=hashlib.sha256();totals={};norms=[];steps=0;gradient=None;dl=loader(rows,assets,cfg,data_epoch,True);accum=1 if a.smoke else t['gradient_accumulation']
            for step,raw in enumerate(dl,1):
                b=_move(raw,torch.device('cuda'));target=bank_batch(raw,bank,torch.device('cuda')) if bank else None
                trace.update(json.dumps(dict(ids=raw['sample_id'],transform=raw.get('spatial_transform')),sort_keys=True).encode())
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=t['precision']=='bf16'):
                    pred=forward(model,b,rules['reliability'])
                    if target is None:
                        mask=b['valid_pixel_mask'].bool()&(b['target']!=255)
                        pixel=F.cross_entropy(pred['logits'].float(),b['target'],ignore_index=255,reduction='none')
                        cls=(pixel*mask).sum()/mask.sum().clamp_min(1);aux=cls*0.;report={};am={}
                    else:
                        cls,report=classification_ce(pred['logits'],b['target'],b['valid_pixel_mask'],target,rules['arm'],rules['alpha'],t)
                        aux,am=reliability_loss(pred,target['target'],target['target_valid'],t,p)
                    loss=cls+t['lambda_reliability']*aux
                (loss/accum).backward()
                if step%accum==0 or step==len(dl):
                    if gradient is None:
                        gradient=encoder_probe(model)
                        if mode=='fullft' and any(v['trainable']<=0 or v['grad_norm']<=0 for v in gradient.values()):raise RuntimeError('full encoder gradient missing')
                        if mode=='frozen' and any(v['trainable'] or v['grad_norm'] for v in gradient.values()):raise RuntimeError('frozen encoder has gradients')
                    norms.append([float(torch.nn.utils.clip_grad_norm_(v,t['clip_norm'],error_if_nonfinite=True)) if v else 0. for v in groups.values()])
                    optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
                for k,v in dict(loss=float(loss.detach()),classification_ce=float(cls.detach()),reliability_loss=float(aux.detach()),**report,**am).items():totals[k]=totals.get(k,0.)+v
                if step==1 or step%50==0 or step==len(dl):
                    msg=dict(event='training',encoder_mode=mode,stage=stage,stage_epoch=local,epoch=epoch,step=step,total=len(dl),loss=float(loss.detach()),time=time.time())
                    write(out/'progress.json',msg);print(json.dumps(msg),flush=True)
            current_encoder=encoder_sha(model);changed=current_encoder!=initial_encoder
            if changed!=(mode=='fullft'):raise RuntimeError('encoder update policy violated')
            if a.smoke:
                smoke.append(dict(stage=stage,steps=steps,encoder_changed=changed,encoder_gradient=gradient,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,bank=bank_report));continue
            write(out/'progress.json',dict(event='gtdev',stage=stage,stage_epoch=local,epoch=epoch,time=time.time()))
            metrics=_gtdev_metrics(model,assets,dev,torch.device('cuda'),threshold=.5,reliability_enabled=False);metrics['stage_epoch']=local
            probe=fixed_probe(model,probe_rows,assets,cfg,probe_targets) if stage=='M' else None
            dev_done=dev_track.update(metrics['miou']);probe_done=False
            if probe and probe['excess_bce'] is not None:probe_done=probe_track.update(probe['excess_bce'])
            finish=dev_done and (stage!='M' or probe_done)
            rec=dict(epoch=epoch,stage=stage,stage_epoch=local,encoder_mode=mode,alpha=rules['alpha'],miou=metrics['miou'],metrics=metrics,
                components={k:v/len(dl) for k,v in totals.items()},evidence=bank_report,probe=probe,updates=steps,
                augmentation_sha256=trace.hexdigest(),elapsed_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                encoder_changed=changed,encoder_sha256=current_encoder,encoder_gradient=gradient,grad_norm_mean=np.mean(norms,axis=0).tolist(),
                stopping=dict(dev=dev_track.state(),probe=probe_track.state() if stage=='M' else None))
            history.append(rec);write(out/'history.json',history)
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,stage=stage,stage_epoch=local,config=cfg,metrics=metrics,
                protocol_hashes=protocol,source_manifest=lock,target_bank=bank,initial_model_sha256=initial,origin_epoch=0,encoder_mode=mode,seed=seed,
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),stopping=rec['stopping'])
            save(out/'last.pt',state)
            if stage!='C' and local>=stop['min_epochs'][stage] and (phase_best is None or metrics['miou']>phase_best['miou']):
                phase_best=dict(epoch=epoch,stage_epoch=local,miou=metrics['miou']);save(phase_best_path,state)
            if stage=='C':
                candidate=dict(epoch=epoch,stage_epoch=local,miou=metrics['miou'],checkpoint=str(out/f'checkpoints/joint_epoch_{epoch:03d}.pt'))
                selected=rank_top3(top+[candidate]);prior=top
                if candidate in selected:save(Path(candidate['checkpoint']),state)
                top=selected;write(out/'top3.json',[dict(rank=i+1,**r) for i,r in enumerate(top)])
                for old in prior:
                    if old not in top:
                        path=Path(old['checkpoint'])
                        if path.parent.resolve()!=(out/'checkpoints').resolve():raise RuntimeError('checkpoint pruning outside own folder')
                        path.unlink()
            del state
            write(out/'progress.json',dict(event='epoch_complete',stage=stage,stage_epoch=local,epoch=epoch,miou=metrics['miou'],stopping=rec['stopping'],time=time.time()))
            print(json.dumps(dict(event='epoch_complete',**rec)),flush=True)
            if finish:reason='early_stopping';break
        if a.smoke:continue
        phase_record=dict(stage=stage,epochs=local,reason=reason,selected=phase_best,dev=dev_track.state(),probe=probe_track.state() if stage=='M' else None)
        if stage!='C':
            if phase_best is None:raise RuntimeError('no eligible warmup checkpoint')
            saved=torch.load(phase_best_path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
            if stage=='M':phase_record['probe_at_selected_checkpoint']=fixed_probe(model,probe_rows,assets,cfg,probe_targets)
        phase_records.append(phase_record);write(out/'phase_summary.json',phase_records)
    if a.smoke:
        write(out/'smoke.json',dict(passed=True,encoder_mode=mode,initial_model_sha256=initial,stages=smoke,test_records_loaded=0));return
    if len(top)!=3:raise RuntimeError('three joint checkpoints required')
    write(out/'completed.json',dict(status='completed',version=p['version'],encoder_mode=mode,seed=seed,epochs=len(history),
        updates=sum(r['updates'] for r in history),phases=phase_records,best_epoch=top[0]['epoch'],best_miou=top[0]['miou'],last_miou=history[-1]['miou'],top3=top,
        initial_model_sha256=initial,encoder_changed=mode=='fullft',task_checkpoint_loaded=False,test_records_loaded=0,
        primary_checkpoint=top[0]['checkpoint'],primary_selection='GTdev rank1 within C; ranks2/3 secondary; no CDL reranking'))

if __name__=='__main__':main()
