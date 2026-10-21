"""Generate only the two user-requested fresh seed42 experiments."""
import json,shutil,subprocess
from pathlib import Path
import yaml
from common import ROOT,REMOTE,BUNDLE,MODES,write,sha,manifest
from configuration import validate_config
from source_snapshot.data import check_protocol

def main():
    if (ROOT/'source_manifest.json').exists():raise FileExistsError('already sealed')
    if ROOT.as_posix()!=REMOTE:raise RuntimeError('dedicated N16R4 directory required')
    host=subprocess.check_output(['scontrol','show','config'],text=True)
    if not any('n16r4' in line.lower() for line in host.splitlines() if 'ClusterName' in line):raise RuntimeError('wrong cluster')
    assets=Path(BUNDLE)/'protocol_assets';check_protocol(assets,include_test=False)
    cfg=yaml.safe_load((Path(BUNDLE)/'configs/b1_linux.yaml').read_text())
    pre=Path(BUNDLE)/'weights/AgriFM.pth';digest=sha(pre)
    if digest!='3a8f4b43f466933c5607916c48f8cf9f17230f449f09e4efb7790ece3278d6a8':raise RuntimeError('pretrain identity')
    (ROOT/'weights').mkdir();shutil.copyfile(pre,ROOT/'weights/AgriFM.pth')
    (ROOT/'protocol_assets').mkdir()
    for name in ('train_validation500.csv','gtdev60.csv','all_candidates500.csv','normalization_train1296.json','test500.csv'):
        shutil.copyfile(assets/name,ROOT/'protocol_assets'/name)
    cfg['data'].update(protocol_assets=str(ROOT/'protocol_assets'),pretrained=str(ROOT/'weights/AgriFM.pth'),agrifm_source=str(ROOT/'vendor/AgriFM_source'))
    cfg['training'].update(num_workers=2,gradient_accumulation=8,decoder_lr=.00015,reliability_lr=.0001,encoder_lr=.00001,
        use_reliability_weighted_ce=True,stage_epochs=dict(A=10,M=10,C=30),ema=0.,clip_norm=1.,weight_threshold=.6,
        weight_temperature=.1,balance_weight=.01,monotonic_weight=.02,separation_weight=.01,center_min_separation=.15)
    for key in ('backbone_lr','feature_residual_ramp_epochs','phase_loss_weight','target_mode','max_train_samples','persistent_workers','log_every'):
        cfg['training'].pop(key,None)
    cfg['model'].update(enable_film=False,enable_feature_residual=False)
    cfg['training']['stopping']=dict(min_epochs=dict(A=3,M=3,C=8),patience=dict(A=3,M=3,C=6),
        dev_min_delta=.0002,probe_min_delta=.0001,probe_tiles=32)
    for key in ('feature_residual_max','feature_residual_ratio_max'):cfg['model'].pop(key,None)
    cfg['experiment_id']='USA-MoE-GMM-Fresh-Frozen-vs-FullFT-v3-20261021'
    (ROOT/'configs').mkdir();configs=[]
    for mode in MODES:
        c=json.loads(json.dumps(cfg));c['training']['seed']=42
        c['data']['output']=str(ROOT/f'runs/joint_{mode}_seed42')
        c['campaign']=dict(version='fresh-ft-v3',arm='joint',encoder_mode=mode,moe_target='joint',epochs=50,
            initialization='agrifm_pretrained_random_head_moe',pretrained_sha256=digest,eta=1.,eps=.0001,
            primary='GTdev60 rank1 within C, fixed0.5, earliest ties; ranks2/3 secondary',
            testing='both modes complete; freeze all SIX top3 weights before Native30m Test500; no test-based reranking')
        validate_config(c);path=ROOT/f'configs/joint_{mode}_seed42.yaml'
        path.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8');configs.append(str(path))
    write(ROOT/'order.json',configs);write(ROOT/'source_manifest.json',manifest())
    write(ROOT/'preparation.json',dict(initialization='pretrained encoder + fresh random decoder/MoE',task_checkpoint_loaded=False,
        pretrained_sha256=digest,source_files=len(manifest()),test_records_loaded=0,seeds=[42],modes=list(MODES),max_concurrent_gpus=2))
    print('prepared TWO fresh seed42 runs',flush=True)

if __name__=='__main__':main()
