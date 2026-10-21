"""Prior-replaced posterior with explicit unavailable-evidence handling. No rank fallback."""
import numpy as np
import torch
from torch.nn import functional as F

def posterior(prior,log_low,log_high,*,eps,eta):
    r=np.clip(np.asarray(prior,dtype=np.float64),eps,1-eps)
    a=np.log(r)-np.log1p(-r)+eta*(log_low-log_high)
    return np.exp(-np.logaddexp(0.,-a)).astype(np.float32)

def fit_density(values,*,steps,min_samples,min_separation,min_weight):
    x=np.asarray(values,dtype=np.float64)
    if x.size<min_samples or not np.isfinite(x).all() or float(x.std())<1e-7:return None
    q05,q95=np.quantile(x,[.05,.95]);x=np.clip((x-q05)/max(float(q95-q05),1e-8),0.,1.)
    means=np.quantile(x,[.25,.75]);var=np.full(2,max(float(x.var()),1e-3));weights=np.full(2,.5)
    for _ in range(steps):
        lp=np.log(weights+1e-12)-.5*(np.log(2*np.pi*var)+(x[:,None]-means)**2/var)
        lp-=lp.max(axis=1,keepdims=True);q=np.exp(lp);q/=q.sum(axis=1,keepdims=True).clip(1e-12)
        mass=q.sum(axis=0).clip(1e-6);weights=mass/len(x);means=(q*x[:,None]).sum(axis=0)/mass
        var=((q*(x[:,None]-means)**2).sum(axis=0)/mass).clip(1e-5,1.)
    order=np.argsort(means);means,var,weights=means[order],var[order],weights[order]
    if means[1]-means[0]<min_separation or weights.min()<min_weight:return None
    pdf=-.5*(np.log(2*np.pi*var)+(x[:,None]-means)**2/var)
    lp=pdf+np.log(weights+1e-12);lp-=lp.max(axis=1,keepdims=True);q=np.exp(lp);q/=q.sum(axis=1,keepdims=True).clip(1e-12)
    return dict(q=q[:,0].astype(np.float32),log_low=pdf[:,0],log_high=pdf[:,1],pi=float(weights[0]),
        means=means.tolist(),variances=var.tolist(),q05=float(q05),q95=float(q95))

@torch.no_grad()
def evidence(logits,labels,prior,observed,t,p):
    shape=(logits.shape[0],128,128)
    if logits.shape[1:]!=(2,256,256) or prior.shape!=shape or observed.shape!=shape:raise ValueError('evidence shape mismatch')
    if not torch.isfinite(prior).all() or (prior<0).any() or (prior>1).any():raise ValueError('invalid MoE prior')
    labels=labels.long();valid=labels!=255
    ce=F.cross_entropy(logits.float(),labels,ignore_index=255,reduction='none')
    count=F.avg_pool2d(valid.float()[:,None],2,2)[:,0]*4
    loss=F.avg_pool2d((ce*valid)[:,None],2,2)[:,0]*4/count.clamp_min(1.)
    positives=F.avg_pool2d((labels==1).float()[:,None],2,2)[:,0]*4;ratio=positives/count.clamp_min(1.)
    label_valid=(count>=3)&((ratio>=.75)|(ratio<=.25));eligible=label_valid&observed.bool();cls=(ratio>=.5).long()
    r=prior.detach().float().clamp(p['eps'],1-p['eps'])
    qg=torch.full_like(r,.5);qf=r.clone();gmm=torch.zeros_like(eligible);reports=[]
    for b in range(logits.shape[0]):
        for c in (0,1):
            mask=eligible[b]&(cls[b]==c);x=loss[b][mask].cpu().numpy()
            if not len(x):reports.append(dict(batch=b,cls=c,mode='empty',count=0));continue
            fit=fit_density(x,steps=int(t['gmm_steps']),min_samples=int(t['gmm_min_samples']),
                min_separation=float(t['gmm_min_separation']),min_weight=float(t['gmm_min_weight']))
            if fit is None:
                reports.append(dict(batch=b,cls=c,mode='unavailable_prior_only',count=len(x)));continue
            qg[b][mask]=torch.as_tensor(fit['q'],device=logits.device)
            qf[b][mask]=torch.as_tensor(posterior(r[b][mask].cpu().numpy(),fit['log_low'],fit['log_high'],
                eps=p['eps'],eta=p['eta']),device=logits.device)
            gmm[b][mask]=True
            reports.append(dict(batch=b,cls=c,mode='gmm',count=len(x),**{k:v for k,v in fit.items() if k not in ('q','log_low','log_high')}))
    return dict(qg=qg.detach(),qf=qf.detach(),prior=r.detach(),valid=eligible.detach(),
        gmm=gmm.detach(),label_valid=label_valid.detach(),observed=observed.bool().detach()),reports

def choose_target(e,kind,previous,ema):
    """Same valid mask across all arms. Unavailable GMM never supplies an auxiliary target."""
    target=e['qf'] if kind=='joint' else e['qg'];target=target.detach().clone();mask=e['valid']&e['gmm']
    if previous is not None:
        old=mask&previous['target_valid'].to(mask.device)
        target[old]=ema*previous['target'].to(target)[old]+(1-ema)*target[old]
    return target.detach(),mask.detach()

def score_for_arm(e,arm):
    if arm=='gmm':return e['qg'],e['valid']&e['gmm']
    if arm=='moe':return e['prior'],e['valid']
    if arm=='mean':return torch.where(e['gmm'],(e['qg']+e['prior'])/2,e['prior']),e['valid']
    if arm in ('fusion','joint'):return e['qf'],e['valid']
    if arm=='ordinary':return e['qg'],torch.zeros_like(e['valid'])
    raise ValueError('unknown arm')

def classification_ce(logits,labels,valid,e,arm,alpha,t):
    valid=valid.bool()&(labels!=255);pixel=F.cross_entropy(logits.float(),labels.masked_fill(~valid,255),ignore_index=255,reduction='none')
    ordinary=(pixel*valid).sum()/valid.sum().clamp_min(1);score,available=score_for_arm(e,arm)
    score=F.interpolate(score.detach().float()[:,None],size=labels.shape[-2:],mode='nearest')[:,0]
    available=F.interpolate(available.float()[:,None],size=labels.shape[-2:],mode='nearest')[:,0].bool()
    mapped=t['reliability_weight_floor']+(1-t['reliability_weight_floor'])*torch.sigmoid((score-t['weight_threshold'])/t['weight_temperature'])
    weight=torch.where(available,mapped,torch.ones_like(mapped))*valid
    weighted=(pixel*weight).sum()/weight.sum().clamp_min(1.)
    return ordinary+alpha*(weighted-ordinary),dict(ordinary_ce=float(ordinary.detach()),weighted_ce=float(weighted.detach()),
        weight_mean=float(weight[valid].mean()) if valid.any() else 0.,
        ess_ratio=float(weight.sum().square()/(weight.square().sum().clamp_min(1e-12)*valid.sum().clamp_min(1))))
