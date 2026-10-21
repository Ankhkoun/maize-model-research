"""Full encoder tuning is explicit in requires_grad, optimizer, and train/eval mode."""
from source_snapshot.model import AgriFMA4USA


class FreshAgriFMA4USA(AgriFMA4USA):
    def set_stage(self,stage):
        super().set_stage(stage)
        full=getattr(self,'encoder_mode','frozen')=='fullft'
        self.classifier.patch_embed.requires_grad_(full)
        self.classifier.backbone.requires_grad_(full)
        self.stage_film.requires_grad_(False)
        self.multiscale_reliability.requires_grad_(False)
        self.expert_feature_residual.requires_grad_(False)
        return {name:sum(p.numel() for p in params) for name,params in parameter_groups(self).items()}

    def train(self,mode=True):
        super().train(mode)
        active=bool(mode and getattr(self,'encoder_mode','frozen')=='fullft')
        self.classifier.backbone.train(active)
        self.classifier.patch_embed.train(active)
        return self


def parameter_groups(model):
    groups=dict(encoder=[],decoder=[],reliability=[])
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        if name.startswith(('classifier.patch_embed.','classifier.backbone.')):key='encoder'
        elif name.startswith('classifier.decoder.'):key='decoder'
        elif name.startswith('reliability.'):key='reliability'
        else:raise RuntimeError('unexpected trainable parameter: '+name)
        groups[key].append(param)
    return groups


def stage_at(epoch,t):
    offset=0
    for stage in ('A','M','C'):
        count=int(t['stage_epochs'][stage])
        if offset<epoch<=offset+count:return stage,epoch-offset
        offset+=count
    raise ValueError('epoch outside schedule')


def policy(stage,epoch,t):
    from configuration import alpha_at
    if stage=='A':return dict(reliability=False,target=None,arm='ordinary',alpha=0.)
    if stage=='M':return dict(reliability=True,target='gmm',arm='ordinary',alpha=0.)
    if stage=='C':return dict(reliability=True,target='joint',arm='joint',alpha=alpha_at(epoch,t))
    raise ValueError(stage)


class Progress:
    """Track substantive improvements separately from exact checkpoint rankings."""
    def __init__(self,delta,minimum,patience,maximize=True):
        self.delta=delta;self.minimum=minimum;self.patience=patience;self.maximize=maximize
        self.best=None;self.bad=0;self.count=0
    def update(self,value):
        import math
        if not math.isfinite(value):raise ValueError('nonfinite monitoring metric')
        self.count+=1
        improved=self.best is None or (value>self.best+self.delta if self.maximize else value<self.best-self.delta)
        if improved:self.best=value;self.bad=0
        else:self.bad+=1
        return self.count>=self.minimum and self.bad>=self.patience
    def state(self):return dict(best_substantive=self.best,consecutive_without_improvement=self.bad,epochs=self.count)


def rank_top3(records):
    return sorted(records,key=lambda x:(-x['miou'],x['epoch']))[:3]
