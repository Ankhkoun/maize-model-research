import copy,unittest
import numpy as np
import torch
from fusion import posterior,fit_density,evidence,choose_target,classification_ce
from objectives import reliability_loss
from source_snapshot.targets import _fit_two_gaussians
from source_snapshot.data import tile_grid_to_windows
from configuration import validate_config,alpha_at

def settings(arm='joint'):
    t=dict(seed=42,precision='bf16',micro_batch=1,gradient_accumulation=8,stage_epochs=dict(A=10,M=10,C=30),encoder_lr=.00001,
        decoder_lr=.00015,reliability_lr=.0001,weight_decay=.0001,lambda_reliability=1.,reliability_weight_floor=.15,
        reliability_ramp_epochs=3,use_reliability_weighted_ce=arm!='ordinary',ema=0.,gmm_steps=40,gmm_min_samples=32,
        gmm_min_separation=.08,gmm_min_weight=.03,spatial_augmentation=True,num_workers=0,prefetch_factor=2,
        clip_norm=1.,weight_threshold=.6,weight_temperature=.1,balance_weight=.01,monotonic_weight=.02,
        separation_weight=.01,center_min_separation=.15,stopping=dict(min_epochs=dict(A=3,M=3,C=8),
        patience=dict(A=3,M=3,C=6),dev_min_delta=.0002,probe_min_delta=.0001,probe_tiles=32))
    p=dict(version='fresh-ft-v3',arm=arm,moe_target='joint' if arm=='joint' else 'gmm',eps=.0001,eta=1.,epochs=50,
        encoder_mode='frozen',initialization='agrifm_pretrained_random_head_moe')
    return t,p

class Contracts(unittest.TestCase):
    def data(self):
        torch.manual_seed(4);z=torch.randn(1,2,256,256);y=torch.zeros(1,256,256,dtype=torch.long);y[:,:,128:]=1
        r=torch.rand(1,128,128)*.8+.1;obs=torch.ones_like(r,dtype=torch.bool)
        return z,y,r,obs
    def test_formula_and_controls(self):
        np.testing.assert_allclose(posterior([.8],np.log([4]),np.zeros(1),eps=.0001,eta=1),[16/17],rtol=1e-6)
        self.assertAlmostEqual(float(posterior(0.,0.,0.,eps=.05,eta=0)),.05,places=6)
        a=posterior(.3,np.log(4),0.,eps=.0001,eta=0);self.assertAlmostEqual(float(a),.3,places=6)
        self.assertGreater(float(posterior(.3,np.log(4),0.,eps=.0001,eta=2)),float(a))
    def test_density_matches_original(self):
        rng=np.random.default_rng(18);x=np.r_[rng.normal(.1,.05,500),rng.normal(.8,.05,500)]
        f=fit_density(x,steps=40,min_samples=32,min_separation=.08,min_weight=.03)
        np.testing.assert_allclose(f['q'],_fit_two_gaussians(x)[0],atol=2e-7)
        self.assertIsNone(fit_density(x,steps=40,min_samples=2000,min_separation=.08,min_weight=.03))
    def test_equal_loss_has_no_position_rank(self):
        z=torch.zeros(1,2,256,256);y=torch.zeros(1,256,256,dtype=torch.long);r=torch.full((1,128,128),.7)
        t,p=settings();e,_=evidence(z,y,r,torch.ones_like(r,dtype=torch.bool),t,p)
        self.assertFalse(e['gmm'].any());self.assertEqual(len(torch.unique(e['qf'])),1)
        torch.testing.assert_close(e['qf'],r);torch.testing.assert_close(e['qg'],torch.full_like(r,.5))
        target,mask=choose_target(e,'joint',None,0.);self.assertFalse(mask.any())
    def test_unavailable_uses_prior_and_permutation(self):
        z,y,r,obs=self.data();t,p=settings();t['gmm_min_samples']=100000
        e,_=evidence(z,y,r,obs,t,p);torch.testing.assert_close(e['qf'],r)
        er,_=evidence(z.flip(-1),y.flip(-1),r.flip(-1),obs.flip(-1),t,p)
        torch.testing.assert_close(er['qf'].flip(-1),e['qf'])
        a,_=classification_ce(z,y,y!=255,e,'gmm',1.,t);b,_=classification_ce(z,y,y!=255,e,'ordinary',1.,t)
        torch.testing.assert_close(a,b)
    def test_no_observations_use_unit_ce_and_skip_bce(self):
        z,y,r,obs=self.data();t,p=settings();e,_=evidence(z,y,r,obs&False,t,p)
        self.assertFalse(e['valid'].any());target,v=choose_target(e,'joint',None,0.);self.assertFalse(v.any())
        a,_=classification_ce(z,y,y!=255,e,'joint',1.,t);b,_=classification_ce(z,y,y!=255,e,'ordinary',1.,t)
        torch.testing.assert_close(a,b)
    def test_target_source_detach_and_ema(self):
        z,y,r,obs=self.data();r.requires_grad_();t,p=settings();e,_=evidence(z,y,r,obs,t,p)
        qj,v=choose_target(e,'joint',None,0.);qg,vg=choose_target(e,'gmm',None,0.)
        self.assertTrue(torch.equal(v,vg));self.assertFalse(qj.requires_grad)
        self.assertGreater(float((qj-qg).abs()[v].sum()),0)
        prev=dict(target=torch.zeros_like(qj),target_valid=v)
        smoothed,_=choose_target(e,'joint',prev,.75);torch.testing.assert_close(smoothed[v],qj[v]*.25)
    def test_bce_detached_target_and_finite_gradient(self):
        t,p=settings();r=torch.full((1,128,128),.3,requires_grad=True);q=torch.full_like(r,.9,requires_grad=True)
        out=dict(reliability=tile_grid_to_windows(r),importance=torch.full((1,16,1024,4),.25),
            phi=torch.zeros(1,16,1024,26),centers=torch.tensor([.1,.3,.6,.9]))
        loss,report=reliability_loss(out,q,torch.ones_like(r,dtype=torch.bool),t,p);loss.backward()
        self.assertIsNone(q.grad);self.assertTrue(torch.isfinite(r.grad).all());self.assertLess(float(r.grad.mean()),0)
        self.assertAlmostEqual(report['reliability_bce'],-(.9*np.log(.3)+.1*np.log(.7)),places=5)
    def test_weight_parameters_and_no_gradient_to_prior(self):
        z,y,r,obs=self.data();z.requires_grad_();t,p=settings();e,_=evidence(z,y,r,obs,t,p)
        a,_=classification_ce(z,y,y!=255,e,'joint',1.,t);ga=torch.autograd.grad(a,z,retain_graph=True)[0]
        t2=dict(t,reliability_weight_floor=1.)
        b,_=classification_ce(z,y,y!=255,e,'joint',1.,t2);gb=torch.autograd.grad(b,z)[0]
        self.assertGreater(float((ga-gb).abs().sum()),0)
        self.assertEqual(alpha_at(2,dict(t,reliability_ramp_epochs=4)),.5)
    def test_configuration_rejects_silent_knobs(self):
        t,p=settings();m=dict(num_experts=4,reliability_input='a4_cube',use_checkpointing=True,window_chunk_size=1,
            use_support_features=True,use_patch_metadata=True,warmup_backbone=False,enable_phase_supervision=False,enable_film=False,enable_feature_residual=False)
        c=dict(training=t,campaign=p,model=m,evaluation=dict(threshold_policy='fixed_0.50',selection_metric='miou',grid='native30m'))
        validate_config(c);bad=copy.deepcopy(c);bad['training']['misspelled_ema']=.5
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(c);bad['model']['enable_film']=True
        with self.assertRaises(ValueError):validate_config(bad)
    def test_all_invalid_is_finite(self):
        z,y,r,obs=self.data();y[:]=255;z.requires_grad_();t,p=settings();e,_=evidence(z,y,r,obs,t,p)
        loss,report=classification_ce(z,y,y!=255,e,'joint',1.,t);self.assertEqual(float(loss.detach()),0.)
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
    def test_full_schedule_and_preheat_targets(self):
        from experiment import stage_at,policy
        t,p=settings()
        self.assertEqual([stage_at(e,t)[0] for e in range(1,51)],['A']*10+['M']*10+['C']*30)
        self.assertEqual(policy('A',1,t),dict(reliability=False,target=None,arm='ordinary',alpha=0.))
        self.assertEqual(policy('M',1,t),dict(reliability=True,target='gmm',arm='ordinary',alpha=0.))
        self.assertEqual(policy('C',1,t)['target'],'joint')
        self.assertAlmostEqual(policy('C',1,t)['alpha'],1/3)
    def test_early_stopping_and_top3(self):
        from experiment import Progress,rank_top3
        track=Progress(.0002,8,6)
        stopped=[track.update(v) for v in [.7,.71,.72,.719,.718,.717,.716,.715,.714]]
        self.assertFalse(any(stopped[:8]));self.assertTrue(stopped[8])
        track=Progress(.0002,3,3)
        for value in [.8,.799,.798]:self.assertFalse(track.update(value))
        self.assertFalse(track.update(.81));self.assertEqual(track.bad,0)
        rows=[dict(epoch=i,miou=v) for i,v in enumerate([.7,.8,.8,.79,.81],1)]
        self.assertEqual([x['epoch'] for x in rank_top3(rows)],[5,2,3])

if __name__=='__main__':unittest.main(verbosity=2)
