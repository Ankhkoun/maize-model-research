import json
from common import ROOT,write,verify

def main():
    lock=verify();seal=json.loads((ROOT/'freeze_manifest.json').read_text());rows=[]
    if not seal['all_runs_completed'] or seal['source_manifest']!=lock:raise RuntimeError('unfrozen')
    for record in seal['records']:
        r=json.loads((ROOT/'native30m'/record['name']/'results.json').read_text());meta=r['protocol_metadata']
        if r['status']!='completed' or r['test_tiles']!=500 or r['overall']['valid_pixels']!=1959211:raise RuntimeError('incomplete Native30m')
        if meta['checkpoint_sha256']!=seal['checkpoints'][record['checkpoint']] or meta['test_pseudo_labels_loaded']:raise RuntimeError('wrong checkpoint/protocol')
        if meta['evaluator_sha256']!='52f7f409a54405b54fe0b46630740bc58a01442092898f524a03b8c0422dc7bc':raise RuntimeError('evaluator changed')
        rows.append(dict(**record,native=r['overall'],per_region=r['per_region'],protocol=meta))
    primary={r['mode']:r['native']['miou'] for r in rows if r['primary']};delta=primary['fullft']-primary['frozen']
    write(ROOT/'outputs/results.json',dict(status='completed',rows=rows,primary=primary,fullft_minus_frozen_primary_miou=delta,source_manifest=lock,
        scope='fresh task initialization; pretrained encoder; seed42 only; adaptive A/M/C; top3 preselected by GTdev'))
    lines=['# fresh-ft-v3：冻结编码器 vs 全量微调','',
        '两组均从原始AgriFM预训练编码器和全新任务头/MoE开始。A/M/C按预定监测规则结束，实际轮数见phase_summary。',
        '每组top3均按GTdev提前选定并统一冻结；CDL rank1为主结果，rank2/3为次要结果，不按CDL重新排名。','',
        '|策略|GTdev排名|联合阶段轮|GTdev mIoU|CDL mIoU|玉米F1|OA|','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        n=r['native'];lines.append(f"|{r['mode']}|{r['rank']}|{r['stage_epoch']}|{r['miou']*100:.4f}%|{n['miou']*100:.4f}%|{n['corn_f1']*100:.4f}%|{n['oa']*100:.4f}%|")
    lines+=['',f'主结果全量微调相对冻结：{delta*100:+.4f}个百分点。','',
        '同一套阶段上限/早停规则，实际轮数可以不同，因此不是严格同更新数对照。',
        '单seed、同一预训练底座，没有GMM/普通CE消融，不支持融合机制的独立因果结论；Test500为复用协议。']
    (ROOT/'outputs/results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(primary),flush=True)

if __name__=='__main__':main()
