"""Keep the frozen training module intact; use Windows-like worker spawning."""
import argparse, importlib.metadata, json, os, platform, runpy, sys
from pathlib import Path
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or int(os.environ.get('SLURM_NTASKS','1'))!=1:
        raise RuntimeError('This reproduction is single GPU, not DDP. Request one task/GPU.')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one allocated CUDA device for this baseline.')
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    output=Path(cfg['data']['output'])
    if output.exists() and any(output.iterdir()):raise FileExistsError('Output is nonempty; select a new run directory')
    output.mkdir(parents=True,exist_ok=True)
    versions={}
    for name in ('torch','numpy','PyYAML','rasterio','affine'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
    (output/'migration_environment.json').write_text(json.dumps(dict(
        python=sys.version,platform=platform.platform(),packages=versions,
        torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        multiprocessing='spawn',historical_rng_bitwise_reproduction_guaranteed=False),indent=2),encoding='utf-8')
    torch.multiprocessing.set_start_method('spawn',force=True)
    sys.argv=['af2_usa_a4_rf_20261005.train','--config',str(a.config.resolve())]
    runpy.run_module('af2_usa_a4_rf_20261005.train',run_name='__main__')

if __name__=='__main__':main()
