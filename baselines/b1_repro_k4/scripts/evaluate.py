"""Unified B1 evaluation without changing the frozen reference evaluators."""
from __future__ import annotations
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_model import make
from af2_usa_a4_rf_20261005.data import USAAssets, check_protocol, load_rows
from af2_usa_a4_rf_20261005.train import _gtdev_metrics, _validate_gtdev_assets


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def validate_checkpoint_config(cfg, checkpoint):
    """Reject accidental evaluation of a different architecture/ablation."""
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError('Expected a full B1 training checkpoint with model state')
    if cfg['model'].get('num_experts', 4) != 4:
        raise ValueError('This entry point is for the K4 B1 baseline')
    if cfg['model'].get('reliability_input', 'a4_cube') != 'a4_cube':
        raise ValueError('Multiscale R1 is a different experiment')
    for key in ['enable_feature_residual', 'enable_film', 'enable_phase_supervision']:
        if cfg['model'].get(key, False):
            raise ValueError('B1 requires ' + key + '=false')
    if cfg['training'].get('use_reliability_weighted_ce', False):
        raise ValueError('B1 weighted CE is disabled')
    saved = checkpoint.get('config')
    if saved is None:
        raise ValueError('Checkpoint config is missing; cannot establish experiment identity')
    for section in ['model', 'training', 'evaluation']:
        if saved.get(section) != cfg.get(section):
            raise ValueError('Checkpoint/config mismatch in ' + section)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['gtdev', 'cdl', 'predict'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cdl', type=Path, help='Original nationwide CDL raster, required in cdl mode')
    parser.add_argument('--sample-id', help='One ID in all_candidates500.csv, required in predict mode')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.mode == 'cdl' and args.cdl is None:
        parser.error('--mode cdl requires --cdl')
    if args.mode == 'cdl' and args.device != 'cuda':
        parser.error('The frozen CDL evaluator requires CUDA')
    if args.mode == 'predict' and not args.sample_id:
        parser.error('--mode predict requires --sample-id')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError('Use a new or empty output directory')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint_config(cfg, payload)
    checkpoint_sha = digest(args.checkpoint)
    if args.mode == 'cdl':
        # Preserve the exact reference reprojection/masking/deduplication code.
        del payload
        sys.argv = ['evaluate_native30', '--config', str(args.config),
                    '--checkpoint', str(args.checkpoint), '--output', str(args.output),
                    '--cdl', str(args.cdl)]
        runpy.run_module('af2_usa_a4_rf_20261005.evaluate_native30', run_name='__main__')
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for diagnostic inference only')
    assets_dir = Path(cfg['data']['protocol_assets'])
    check_protocol(assets_dir, include_test=False)
    assets = USAAssets(Path(cfg['data']['root']), assets_dir / 'normalization_train1296.json')
    model = make(cfg, device)
    model.load_state_dict(payload['model'], strict=True)
    epoch = payload.get('epoch')
    del payload
    model.set_film_scale(0.0)
    model.set_feature_residual_scale(0.0)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(mode=args.mode, checkpoint_sha256=checkpoint_sha, epoch=epoch,
                    threshold=.5, device=str(device))
    if args.mode == 'gtdev':
        rows = load_rows(assets_dir / 'gtdev60.csv')
        if len(rows) != 60:
            raise ValueError('Expected the fixed 60-tile GTdev manifest')
        _validate_gtdev_assets(assets, rows)
        metrics = _gtdev_metrics(model, assets, rows, device, threshold=.5,
                                 reliability_enabled=False)
        result = identity | dict(grid='Derived85 GTdev60', metrics=metrics)
    else:
        matches = [r for r in load_rows(assets_dir / 'all_candidates500.csv')
                   if r['sample_id'] == args.sample_id]
        if len(matches) != 1:
            raise ValueError('sample-id must occur exactly once in all_candidates500.csv')
        # No pseudo-label or reference-label access for prediction.
        sample = assets.load_sample(matches[0], need_label=False)
        logits = model.classifier(sample['agrifm_cube'][None].to(device))
        if logits.shape != (1, 2, 256, 256) or not torch.isfinite(logits).all():
            raise ValueError('Invalid prediction logits')
        probability = logits.float().softmax(1)[0, 1].cpu().numpy().astype(np.float32)
        prediction = (probability >= .5).astype(np.uint8)
        np.save(args.output / 'corn_probability.npy', probability, allow_pickle=False)
        np.save(args.output / 'prediction.npy', prediction, allow_pickle=False)
        result = identity | dict(sample_id=args.sample_id, grid='Input 10m 256x256',
                                 probability='corn_probability.npy', prediction='prediction.npy',
                                 pseudo_labels_loaded=False, reference_labels_loaded=False)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
