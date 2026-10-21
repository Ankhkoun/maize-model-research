"""FP32 BCE on the mixture of expert probabilities, with the original regularizers."""
import torch
from torch.nn import functional as F
from source_snapshot.data import windows_to_tile_grid

def reliability_loss(output,target,valid,t,p):
    r=windows_to_tile_grid(output['reliability'])
    if r.shape!=target.shape or valid.shape!=r.shape:raise ValueError('target alignment mismatch')
    # BCE on a probability mixture must not be replaced by averaging expert logits.
    with torch.autocast(device_type=r.device.type,enabled=False):
        prediction=r.float();soft=target.detach().float();mask=valid.bool()
        clipped=prediction.clamp(p['eps'],1-p['eps'])
        if mask.any():
            bce=(F.binary_cross_entropy(clipped,soft,reduction='none')*mask).sum()/mask.sum()
            mse=((prediction-soft).square()*mask).sum()/mask.sum()
            clipped_fraction=float(((prediction!=clipped)&mask).sum()/mask.sum())
        else:bce=prediction.sum()*0.;mse=bce;clipped_fraction=0.
        importance=output['importance'].float().mean(dim=(0,1,2));balance=(importance-1/importance.numel()).square().sum()
        phi=output['phi'].float();mono=F.relu(phi[...,:-1]-phi[...,1:]).mean()
        centers=output['centers'].float();separation=F.relu(t['center_min_separation']-torch.diff(centers)).sum()
        total=bce+t['balance_weight']*balance+t['monotonic_weight']*mono+t['separation_weight']*separation
    return total,dict(reliability_bce=float(bce.detach()),reliability_mse_diagnostic=float(mse.detach()),
        balance=float(balance.detach()),monotonic=float(mono.detach()),separation=float(separation.detach()),
        probability_clip_fraction=clipped_fraction,target_pixels=int(valid.sum()))
