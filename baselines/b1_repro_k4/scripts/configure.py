"""Change only the five filesystem paths of the historical B1 config."""
import argparse, json, sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--pretrained',type=Path,default=ROOT/'weights/AgriFM.pth')
    p.add_argument('--save',type=Path,default=ROOT/'configs/local.yaml')
    a=p.parse_args()
    if a.save.exists():raise FileExistsError(f'Config exists: {a.save}; use another --save')
    cfg=yaml.safe_load((ROOT/'configs/b1.yaml').read_text(encoding='utf-8'))
    cfg['data'].update(root=str(a.data_root.expanduser().resolve()),
                       output=str(a.output.expanduser().resolve()),
                       protocol_assets=str(ROOT/'protocol_assets'),
                       agrifm_source=str(ROOT/'vendor/AgriFM_source'),
                       pretrained=str(a.pretrained.expanduser().resolve()))
    a.save.parent.mkdir(parents=True,exist_ok=True)
    a.save.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    print(json.dumps({'config':str(a.save.resolve()),'data':cfg['data']},indent=2))

if __name__=='__main__':main()
