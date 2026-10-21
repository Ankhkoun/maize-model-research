"""Native30m-v1 test of all preselected top3 checkpoints, without test reranking."""
import argparse,json,sys
from pathlib import Path
import torch
from common import ROOT,write,sha,verify
from source_snapshot import evaluate_native30 as native

@torch.inference_mode()
def predict_classification(model,assets,row,device):
    sample=assets.load_sample(row,need_label=False)
    logits=model.classifier(sample['agrifm_cube'][None].to(device))
    if logits.shape!=(1,2,256,256) or not torch.isfinite(logits).all():raise RuntimeError('invalid logits')
    return logits.float().softmax(1)[0,1].cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
    seal=json.loads((ROOT/'freeze_manifest.json').read_text());lock=verify()
    if not seal['all_runs_completed'] or seal['source_manifest']!=lock or len(seal['records'])!=6:raise RuntimeError('unfrozen comparison')
    if not 0<=a.index<6 or torch.cuda.device_count()!=1:raise RuntimeError('test allocation/index')
    record=seal['records'][a.index];path=Path(record['checkpoint'])
    if sha(path)!=seal['checkpoints'][str(path)]:raise RuntimeError('checkpoint changed')
    if sha(native.__file__)!='52f7f409a54405b54fe0b46630740bc58a01442092898f524a03b8c0422dc7bc':raise RuntimeError('native evaluator changed')
    out=ROOT/'native30m'/record['name']
    if out.exists():raise FileExistsError(out)
    native.predict_probability=predict_classification
    sys.argv=[native.__file__,'--config',record['config'],'--checkpoint',str(path),'--output',str(out),
        '--cdl','/data/run01/scze971/usa/reference_sources/CDL2021/2021_30m_cdls.tif']
    native.main();write(out/'selection.json',record)

if __name__=='__main__':main()
