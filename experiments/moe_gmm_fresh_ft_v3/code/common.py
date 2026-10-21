from __future__ import annotations
import hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from source_snapshot.data import USATrainDataset,collate_tiles,_augment_sample
from source_snapshot.model import AgriFMA4USA

ROOT=Path(__file__).resolve().parent
REMOTE='/data/run01/scze971/usa_moe_gmm_fresh_ft_v3_20261021'
BUNDLE='/data/run01/scze971/usa_b1_repro_hpc_20261019/usa_b1_repro_hpc_20261019'
PY='/data/run01/scze971/maize_paper_workspace/runs/france_agrifm_b1_fullft_n16r4_20261016/prep-venv/bin/python'
ARMS=('joint',); SEEDS=(42,); MODES=('frozen','fullft')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def state_sha(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def seed_int(seed,epoch,key):
    return int.from_bytes(hashlib.sha256(f'{seed}|{epoch}|{key}'.encode()).digest()[:8],'little')%(2**63-1)

def setup(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def make_model(cfg,device='cuda'):
    from configuration import validate_config
    from experiment import FreshAgriFMA4USA
    validate_config(cfg)
    c=cfg['model']
    m=FreshAgriFMA4USA(cfg['data']['agrifm_source'],cfg['data']['pretrained'],
        use_checkpointing=c['use_checkpointing'],use_support_features=c['use_support_features'],
        use_patch_metadata=c['use_patch_metadata'],warmup_backbone=c['warmup_backbone'],
        reliability_input=c['reliability_input'],num_experts=c['num_experts']).to(device)
    m.encoder_mode=cfg['campaign']['encoder_mode'];m.set_stage('A')
    m.set_film_scale(0.);m.set_feature_residual_scale(0.);m.set_feature_residual_ratio_max(0.)
    # Remove dormant parameters from trainable/optimizer counts.
    m.expert_feature_residual.requires_grad_(False);m.multiscale_reliability.requires_grad_(False)
    m.campaign_chunk=c['window_chunk_size']
    return m

def forward(model,b,reliability=True):
    return model(b['agrifm_cube'],b['a4_cube'],doy=b['doy'],observed_support=b['observed_support'],
        pseudo_class=b['pseudo_class'],pseudo_class_valid=b['pseudo_class_valid'],
        pseudo_purity=b.get('pseudo_purity'),pseudo_valid_fraction=b.get('pseudo_valid_fraction'),
        reliability_enabled=reliability,window_chunk_size=model.campaign_chunk)

class EpochDataset(USATrainDataset):
    def __init__(self,rows,assets,seed,epoch,augment):
        super().__init__(rows,assets,'train',augment=False,seed=seed)
        self.epoch,self.augment=epoch,augment
    def __getitem__(self,i):
        row=self.rows[i];x=self.assets.load_sample(row,need_label=True,need_reference=False)
        if self.augment:
            x=_augment_sample(x,generator=torch.Generator().manual_seed(seed_int(self.seed,self.epoch,row['sample_id'])))
        return x

def loader(rows,assets,cfg,epoch,shuffle):
    ds=EpochDataset(rows,assets,cfg['training']['seed'],epoch,shuffle and cfg['training']['spatial_augmentation'])
    workers=cfg['training']['num_workers']
    kw=dict(batch_size=cfg['training']['micro_batch'],shuffle=shuffle,num_workers=workers,collate_fn=collate_tiles,pin_memory=True,
        generator=torch.Generator().manual_seed(seed_int(cfg['training']['seed'],epoch,'loader')))
    if workers:kw.update(prefetch_factor=cfg['training']['prefetch_factor'],multiprocessing_context='spawn')
    return DataLoader(ds,**kw)

def save(path,payload):
    path=Path(path);temp=path.with_suffix('.tmp.pt');torch.save(payload,temp);temp.replace(path)

def manifest():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()
        and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.yaml','.md')
        and not any(part in ('runs','outputs','logs','validation') for part in p.relative_to(ROOT).parts)}

def verify():
    locked=json.loads((ROOT/'source_manifest.json').read_text())
    if manifest()!=locked:raise RuntimeError('source/config lock changed')
    return locked
