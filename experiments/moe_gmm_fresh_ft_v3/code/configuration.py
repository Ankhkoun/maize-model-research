"""Validate fixed checkpoint contracts and tunable values before allocating a run."""
import math

ARMS=('ordinary','gmm','moe','mean','fusion','joint')

def validate_config(c):
    m,t,p=c['model'],c['training'],c['campaign']
    for key,value in t.items():
        if isinstance(value,(int,float)) and not math.isfinite(float(value)):raise ValueError('nonfinite '+key)
    allowed_training={'seed','precision','micro_batch','gradient_accumulation','stage_epochs','decoder_lr',
        'reliability_lr','encoder_lr','weight_decay','lambda_reliability','reliability_weight_floor','reliability_ramp_epochs',
        'use_reliability_weighted_ce','ema','gmm_steps','gmm_min_samples','gmm_min_separation','gmm_min_weight',
        'spatial_augmentation','num_workers','prefetch_factor','clip_norm','weight_threshold','weight_temperature',
        'balance_weight','monotonic_weight','separation_weight','center_min_separation','stopping'}
    if set(t)!=allowed_training:raise ValueError('missing or unsupported training keys: '+str(set(t)^allowed_training))
    if set(m)!={'num_experts','reliability_input','use_checkpointing','window_chunk_size','use_support_features',
                'use_patch_metadata','warmup_backbone','enable_phase_supervision','enable_film','enable_feature_residual'}:
        raise ValueError('unsupported model keys')
    fixed=dict(num_experts=4,reliability_input='a4_cube',use_support_features=True,
        use_patch_metadata=True,warmup_backbone=False,enable_phase_supervision=False,
        enable_film=False,enable_feature_residual=False)
    for key,value in fixed.items():
        if m[key]!=value:raise ValueError(f'{key} violates the common B1 K4 checkpoint contract')
    if p['arm'] not in ARMS:raise ValueError('unknown arm')
    if p['moe_target']!=('joint' if p['arm']=='joint' else 'gmm'):raise ValueError('arm/target mismatch')
    if p['version']!='fresh-ft-v3':raise ValueError('wrong version')
    if p['arm']!='joint' or p['encoder_mode'] not in ('frozen','fullft'):raise ValueError('unsupported comparison')
    if p['initialization']!='agrifm_pretrained_random_head_moe':raise ValueError('trained task checkpoints forbidden')
    if 'initial_checkpoint' in p or 'initial_sha256' in p:raise ValueError('continuation forbidden')
    if not 0<float(p['eps'])<.5 or not math.isfinite(float(p['eta'])) or float(p['eta'])<0:raise ValueError('invalid fusion controls')
    if t['precision'] not in ('bf16','fp32'):raise ValueError('unsupported precision')
    if not 0<=float(t['ema'])<1:raise ValueError('EMA must be in [0,1)')
    if not 0<=float(t['reliability_weight_floor'])<=1:raise ValueError('invalid floor')
    if not 0<=float(t['weight_threshold'])<=1 or float(t['weight_temperature'])<=0:raise ValueError('invalid weight mapping')
    if t['use_reliability_weighted_ce']!=(p['arm']!='ordinary'):raise ValueError('arm/weighted CE flag mismatch')
    for key in ('decoder_lr','encoder_lr','reliability_lr','clip_norm','gmm_min_separation','gmm_min_weight'):
        if float(t[key])<=0:raise ValueError('nonpositive '+key)
    if float(t['lambda_reliability'])<0 or float(t['weight_decay'])<0:raise ValueError('negative loss/optimizer coefficient')
    for key in ('micro_batch','gradient_accumulation','gmm_steps','gmm_min_samples'):
        if int(t[key])<1:raise ValueError('invalid '+key)
    if 1296%int(t['micro_batch']) or (1296//int(t['micro_batch']))%int(t['gradient_accumulation']):
        raise ValueError('batch and accumulation must divide Train1296 exactly')
    if int(t['reliability_ramp_epochs'])<0 or int(t['num_workers'])<0 or int(m['window_chunk_size'])<1:raise ValueError('invalid scheduling value')
    if not 0<float(t['gmm_min_weight'])<.5:raise ValueError('invalid mixture component weight')
    if int(p['epochs'])<1:raise ValueError('no training epochs')
    if float(t['center_min_separation'])<0:raise ValueError('negative center separation')
    for key in ('balance_weight','monotonic_weight','separation_weight'):
        if float(t[key])<0:raise ValueError('negative regularizer')
    if c['evaluation']['threshold_policy']!='fixed_0.50':raise ValueError('fixed .5 selection required')
    if c['evaluation']['selection_metric']!='miou' or c['evaluation']['grid']!='native30m':raise ValueError('unsupported evaluation protocol')
    schedule=t['stage_epochs']
    if set(schedule)!={'A','M','C'} or min(int(schedule[k]) for k in schedule)<1:raise ValueError('positive A/M/C stages required')
    if sum(int(v) for v in schedule.values())!=int(p['epochs']):raise ValueError('total stage budget mismatch')
    stop=t['stopping']
    if set(stop)!={'min_epochs','patience','dev_min_delta','probe_min_delta','probe_tiles'}:raise ValueError('stopping keys')
    if set(stop['min_epochs'])!={'A','M','C'} or set(stop['patience'])!={'A','M','C'}:raise ValueError('stopping stages')
    for stage in ('A','M','C'):
        if not 1<=int(stop['min_epochs'][stage])<=int(schedule[stage]) or int(stop['patience'][stage])<1:raise ValueError('stopping budget')
    if int(stop['min_epochs']['C'])<3 or int(stop['probe_tiles'])<1:raise ValueError('top3/probe budget')
    if any(not math.isfinite(float(stop[key])) or float(stop[key])<0 for key in ('dev_min_delta','probe_min_delta')):raise ValueError('invalid stopping delta')

def alpha_at(epoch,t):
    ramp=int(t['reliability_ramp_epochs'])
    return 1. if ramp==0 else min(1.,epoch/ramp)
