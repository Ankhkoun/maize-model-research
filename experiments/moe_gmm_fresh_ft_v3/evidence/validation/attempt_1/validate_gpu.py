import json,subprocess,sys
import torch,yaml
from common import ROOT,MODES,write,verify,setup,make_model,forward
from source_snapshot.data import USAAssets,load_rows,collate_tiles
from source_snapshot.train import _move

def main():
    lock=verify();subprocess.run([sys.executable,'-m','unittest','-v','test_v2'],check=True)
    if torch.cuda.device_count()!=1:raise RuntimeError('one GPU required')
    smoke={}
    for mode in MODES:
        subprocess.run([sys.executable,'train.py','--config',str(ROOT/f'configs/joint_{mode}_seed42.yaml'),'--smoke'],check=True)
        smoke[mode]=json.loads((ROOT/f'runs/joint_{mode}_seed42_smoke/smoke.json').read_text())
        if [x['stage'] for x in smoke[mode]['stages']]!=['A','M','C']:raise RuntimeError('missing stage probe')
        if not smoke[mode]['passed']:raise RuntimeError('smoke failed')
    if smoke['frozen']['initial_model_sha256']!=smoke['fullft']['initial_model_sha256']:raise RuntimeError('different initial weights')
    setup(42);cfg=yaml.safe_load((ROOT/'configs/joint_frozen_seed42.yaml').read_text());model=make_model(cfg);model.eval()
    rows=load_rows(ROOT/'protocol_assets/train_validation500.csv');row=next(x for x in rows if x['split']=='train')
    assets=USAAssets(cfg['data']['root'],ROOT/'protocol_assets/normalization_train1296.json')
    b=_move(collate_tiles([assets.load_sample(row,need_label=True,need_reference=False)]),torch.device('cuda'))
    with torch.inference_mode():
        full=forward(model,b,True);cls=forward(model,b,False)
        diff=float((full['logits']-cls['logits']).abs().max())
    if diff!=0:raise RuntimeError('classifier-only inference mismatch')
    write(ROOT/'validation/gpu.json',dict(passed=True,source_manifest=lock,smoke=smoke,
        full_vs_classifier_max_logit_diff=diff,real_tile=row['sample_id'],test_records_loaded=0,gpu=torch.cuda.get_device_name(0),torch=torch.__version__))

if __name__=='__main__':main()
