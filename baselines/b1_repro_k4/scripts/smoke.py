"""One real tile, ordinary CE plus GMM reliability, backward and optimizer step."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
from check_model import make
from af2_usa_a4_rf_20261005.data import USAAssets,load_rows
from af2_usa_a4_rf_20261005.train import _seed,_optimizer
from af2_usa_a4_rf_20261005.targets import build_targets
from af2_usa_a4_rf_20261005.objective import stage_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for real-tile smoke')
    assets=USAAssets(Path(cfg['data']['root']),Path(cfg['data']['protocol_assets'])/'normalization_train1296.json')
    row=next(r for r in load_rows(Path(cfg['data']['protocol_assets'])/'train_validation500.csv') if r['split']=='train')
    sample=assets.load_sample(row,need_label=True)
    batch={k:v.unsqueeze(0).cuda() for k,v in sample.items() if isinstance(v,torch.Tensor)}
    model=make(cfg,torch.device('cuda'));model.set_stage('M');model.set_film_scale(0);model.set_feature_residual_scale(0)
    fields={k:batch[k] for k in ('doy','observed_support','pseudo_class','pseudo_class_valid','pseudo_purity','pseudo_valid_fraction')}
    model.eval()
    with torch.no_grad():
        raw=model(batch['agrifm_cube'],batch['a4_cube'],**fields,reliability_enabled=False)
        target,valid,_=build_targets(raw['logits'],batch['target'],ema=.95,target_mode='gmm',gmm_steps=40)
        del raw
    model.train();optimizer=_optimizer(model,'M',cfg);optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(batch['agrifm_cube'],batch['a4_cube'],**fields,reliability_enabled=True)
        loss,report=stage_loss('M',out,batch,target,valid,reliability_enabled=True,
            lambda_reliability=1,phase_weight=0,use_reliability_weighted_ce=False)
    loss.backward()
    def grad(prefix):
        return sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5
    decoder=grad('classifier.decoder.');rel=grad('reliability.')
    assert decoder>0 and rel>0 and torch.isfinite(loss)
    assert not any(p.grad is not None for p in model.classifier.backbone.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
    print(json.dumps(dict(passed=True,optimizer_steps=1,decoder_grad=decoder,reliability_grad=rel,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,test_accessed=False,**report),indent=2))

if __name__=='__main__':main()
