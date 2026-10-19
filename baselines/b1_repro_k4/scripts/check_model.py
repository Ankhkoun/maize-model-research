"""Construct the actual frozen model and report pretrained weight coverage."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from af2_usa_a4_rf_20261005.model import AgriFMA4USA

def make(cfg,device):
    m=cfg['model']
    return AgriFMA4USA(cfg['data']['agrifm_source'],cfg['data']['pretrained'],
        use_checkpointing=m['use_checkpointing'],use_support_features=m['use_support_features'],
        use_patch_metadata=m['use_patch_metadata'],warmup_backbone=m['warmup_backbone'],
        reliability_input=m['reliability_input'],num_experts=m['num_experts']).to(device)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    model=make(yaml.safe_load(a.config.read_text()),torch.device(a.device))
    print(json.dumps(dict(pretrained_load_report=model.classifier.load_report,
                          total_parameters=sum(p.numel() for p in model.parameters())),indent=2))
