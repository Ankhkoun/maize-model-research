"""Read-only migration checks; no training or test labels by default."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from af2_usa_a4_rf_20261005.data import USAAssets, check_protocol, load_rows
from af2_usa_a4_rf_20261005.train import _load_gtdev_target, _gtdev_target_path

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--full-train-hashes',action='store_true')
    p.add_argument('--final-test',action='store_true')
    a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    original=yaml.safe_load((ROOT/'configs/b1.yaml').read_text(encoding='utf-8'))
    permitted={'root','output','protocol_assets','agrifm_source','pretrained'}
    for key in original:
        if key!='data' and cfg.get(key)!=original[key]:raise ValueError(f'Training protocol changed: {key}')
    if set(cfg)!=set(original) or set(cfg['data'])!=set(original['data']):raise ValueError('Unexpected config keys')
    for key,value in original['data'].items():
        if key not in permitted and cfg['data'][key]!=value:raise ValueError(key)
    for name,expected in json.loads((ROOT/'docs/frozen_core_hashes.json').read_text()).items():
        if digest(ROOT/'af2_usa_a4_rf_20261005'/name)!=expected:raise ValueError('Changed frozen code: '+name)
    ad=Path(cfg['data']['protocol_assets'])
    protocol=check_protocol(ad,include_test=a.final_test)
    rows=load_rows(ad/'train_validation500.csv')
    train=[r for r in rows if r['split']=='train']
    validation=[r for r in rows if r['split']=='validation']
    dev=load_rows(ad/'gtdev60.csv')
    candidates=load_rows(ad/'all_candidates500.csv')
    assert (len(train),len(validation),len(dev),len(candidates))==(1296,330,60,2361)
    assert {r['sample_id'] for r in train}.isdisjoint(r['sample_id'] for r in dev)
    assert {r['sample_id'] for r in dev}.issubset(r['sample_id'] for r in validation)
    assets=USAAssets(Path(cfg['data']['root']),ad/'normalization_train1296.json')
    needed={r['sample_id']:r for r in train+dev}
    missing=[]
    for sid in needed:
        tile=assets.cube_root/sid
        for f in ('x_cube.npy','source_map.npy','metadata.json','time_quality.json'):
            if not (tile/f).is_file():missing.append(str(tile/f))
        pseudo=assets.pseudo_root/sid/'pseudo_label.npy'
        if not pseudo.is_file():missing.append(str(pseudo))
    if missing:raise FileNotFoundError(f'{len(missing)} required files missing, first 10: {missing[:10]}')
    devhashes={}
    for row in dev:
        target=_load_gtdev_target(assets,row['sample_id'])
        assert target.shape==(85,85) and np.isin(target,[0,1,255]).all()
        devhashes[row['sample_id']]=digest(_gtdev_target_path(assets,row['sample_id']))
    sample=assets.load_sample(train[0],need_label=True)
    normalization=json.loads((ad/'normalization_train1296.json').read_text())
    checked=train if a.full_train_hashes else train[:1]
    for row in checked:
        sid=row['sample_id']
        if digest(assets.cube_root/sid/'x_cube.npy')!=normalization['source_hashes'][sid]:
            raise ValueError('Historical Train cube mismatch: '+sid)
    weight_hash=digest(cfg['data']['pretrained'])
    expected='3a8f4b43f466933c5607916c48f8cf9f17230f449f09e4efb7790ece3278d6a8'
    if weight_hash!=expected:raise ValueError('Bundled AgriFM pretrained hash mismatch')
    if digest(Path(cfg['data']['agrifm_source'])/'AgriFM/models/video_swin_transformer.py')!='4917c5f6a37312158b69114cb41f03d109364241b4939df75cad7a9d3ddc3591':
        raise ValueError('AgriFM dependency source hash mismatch')
    if a.final_test:
        test=load_rows(ad/'test500.csv')
        assert len(test)==500
        assert {r['sample_id'] for r in test}.isdisjoint(r['sample_id'] for r in train+dev)
        for row in test:
            tile=assets.cube_dir(row['sample_id'])
            for f in ('source_map.npy','metadata.json','time_quality.json'):
                if not (tile/f).is_file():raise FileNotFoundError(tile/f)
    print(json.dumps(dict(passed=True,train=1296,validation=330,gtdev=60,
        train_cube_hashes_checked=len(checked),protocol_hashes=protocol,
        gtdev_reference_hashes=devhashes,pretrained_sha256=weight_hash,
        gtdev_reference_historical_hashes_available=False,
        warning='No original pseudo-label/source_map/GTdev hash inventory in cache; shape/path checks do not prove byte identity.',
        test='file layout checked without test pseudo labels' if a.final_test else 'not accessed'),indent=2))

if __name__=='__main__':main()
