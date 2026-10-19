import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import yaml
from scripts import evaluate

ROOT=Path(__file__).resolve().parents[1]

class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((ROOT/'configs/b1.yaml').read_text())
        self.ckpt=dict(model={},epoch=9,config=copy.deepcopy(self.cfg))

    def test_matching_checkpoint(self):
        evaluate.validate_checkpoint_config(self.cfg,self.ckpt)

    def test_migration_paths_do_not_change_identity(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['data']['root']='another-data-location'
        evaluate.validate_checkpoint_config(cfg,self.ckpt)

    def test_wrong_architecture_or_training_rejected(self):
        for section,key,value in [('model','reliability_input','multiscale_agrifm'),
                                  ('model','enable_feature_residual',True),
                                  ('training','ema',0.0)]:
            cfg=copy.deepcopy(self.cfg);cfg[section][key]=value
            with self.assertRaises(ValueError):
                evaluate.validate_checkpoint_config(cfg,self.ckpt)

    def test_prediction_does_not_read_labels(self):
        class StubModel(torch.nn.Module):
            def set_film_scale(self,value):pass
            def set_feature_residual_scale(self,value):pass
            def classifier(self,image):
                return torch.zeros(1,2,256,256)
        class StubAssets:
            def __init__(self,*args):pass
            def load_sample(self,row,*,need_label):
                if need_label:raise AssertionError('Predict must not read pseudo labels')
                return {'agrifm_cube':torch.zeros(26,10,2,2)}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/'config.yaml';checkpoint=root/'checkpoint.pt'
            config.write_text(yaml.safe_dump(self.cfg));checkpoint.write_bytes(b'test-only')
            argv=['test.py','--mode','predict','--config',str(config),'--checkpoint',str(checkpoint),
                  '--output',str(root/'out'),'--sample-id','tile','--device','cpu']
            with patch('sys.argv',argv),patch.object(evaluate.torch,'load',return_value=self.ckpt),\
                 patch.object(evaluate,'make',return_value=StubModel()),\
                 patch.object(evaluate,'USAAssets',StubAssets),\
                 patch.object(evaluate,'check_protocol',return_value={}),\
                 patch.object(evaluate,'load_rows',return_value=[dict(sample_id='tile')]):
                evaluate.main()
            result=json.loads((root/'out/results.json').read_text())
            self.assertFalse(result['pseudo_labels_loaded'])
            self.assertFalse(result['reference_labels_loaded'])
            self.assertEqual(np.load(root/'out/prediction.npy').shape,(256,256))

if __name__=='__main__':unittest.main()
