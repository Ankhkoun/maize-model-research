import hashlib,json,unittest
from pathlib import Path
import yaml
from af2_usa_a4_rf_20261005.data import check_protocol,load_rows

ROOT=Path(__file__).resolve().parents[1]

class RepositoryContracts(unittest.TestCase):
    def test_frozen_core_bytes(self):
        hashes=json.loads((ROOT/'docs/frozen_core_hashes.json').read_text())
        for name,expected in hashes.items():
            actual=hashlib.sha256((ROOT/'af2_usa_a4_rf_20261005'/name).read_bytes()).hexdigest()
            self.assertEqual(actual,expected,name)

    def test_fixed_split_and_normalization(self):
        p=ROOT/'protocol_assets'
        self.assertEqual(len(check_protocol(p,include_test=True)),5)
        rows=load_rows(p/'train_validation500.csv')
        train={r['sample_id'] for r in rows if r['split']=='train'}
        val={r['sample_id'] for r in rows if r['split']=='validation'}
        dev={r['sample_id'] for r in load_rows(p/'gtdev60.csv')}
        test={r['sample_id'] for r in load_rows(p/'test500.csv')}
        self.assertEqual((len(train),len(val),len(dev),len(test)),(1296,330,60,500))
        self.assertTrue(dev<=val)
        self.assertFalse(train&val or train&test or val&test)

    def test_published_config_matches_baseline(self):
        cfg=yaml.safe_load((ROOT/'configs/b1.yaml').read_text())
        self.assertEqual(cfg['training']['stage_epochs'],dict(A=3,M=3,J=3,C=0,D=0))
        self.assertEqual(cfg['training']['micro_batch']*cfg['training']['gradient_accumulation'],8)
        self.assertEqual(cfg['training']['seed'],42)
        self.assertEqual(cfg['model']['num_experts'],4)
        self.assertEqual(cfg['training']['gmm_steps'],40)
        self.assertEqual(cfg['training']['ema'],.95)
        for key in ['enable_feature_residual','enable_film','enable_phase_supervision']:
            self.assertFalse(cfg['model'][key])
        self.assertFalse(cfg['training']['use_reliability_weighted_ce'])

if __name__=='__main__':unittest.main()
