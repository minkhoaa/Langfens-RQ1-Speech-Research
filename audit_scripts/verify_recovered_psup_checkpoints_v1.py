"""Read-only safe CPU verification of both independently recovered P-SUP arms.

No model construction, backward pass, optimizer invocation, or dataset extraction.
Rechecks receipts against actual files and in-payload optimizer/scheduler counters.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery-manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.recovery_manifest.exists():
        raise FileExistsError('Preserve old verification receipts; use new outputs')
    import numpy as np
    import torch
    safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    results, inventory = {}, {}
    expected = dict(contract_sha256='4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2',
        split_manifest_sha256='7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b',
        zip_sha256='b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9',
        git_sha='dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f',
        backbone='facebook/wav2vec2-base-960h',backbone_revision='22aad52d435eb6dbaf354bdad9b0da84ce7d6156')
    for arm, objective, best_update in [('t_op','operation_ce',2700),('b_control','binary_collapsed',2250)]:
        receipt_path=args.benchmark_dir/'recovery'/arm/'RECOVERY_RECEIPT.json'
        receipt=json.loads(receipt_path.read_text())
        artifacts=receipt['recovered_artifacts']
        inventory[arm]=dict(receipt_sha256=sha(receipt_path), receipt=str(receipt_path), files={})
        for name in ['fixed_budget_3600.pt','best.pt','last.pt','training_summary.json','history.json']:
            item=artifacts[name]
            path=Path(item.get('scratch_path',item.get('absolute_scratch_path')))
            actual=sha(path)
            if actual != item['sha256'] or path.stat().st_size != item['size_bytes']:
                raise ValueError(f'Recovery receipt mismatch: {arm}/{name}')
            inventory[arm]['files'][name]=dict(local_path=str(path),sha256=actual,bytes=path.stat().st_size,
                remote_repository='minkhoaa/langfens-rq1-cross-fold-checkpoints',
                remote_revision='4aa9f2a68f0709889eacacf170b6446d3c45c73d',remote_path=f'psup/{arm}/{name}')
            if not name.endswith('.pt'):
                continue
            with torch.serialization.safe_globals(safe):
                data=torch.load(path,map_location='cpu',weights_only=True)
            required={'head','model','optimizer','scheduler','scaler','rng','metadata','history','successful_updates','skipped_amp_attempts_total'}
            if not required.issubset(data):
                raise ValueError('Incomplete checkpoint payload')
            updates=best_update if name=='best.pt' else 3600
            assert data['successful_updates']==updates
            assert data['skipped_amp_attempts_total']==0
            assert data['micro_steps']==updates*2
            assert data['is_fixed_budget_endpoint']==(name=='fixed_budget_3600.pt')
            metadata=data['metadata']
            assert metadata['objective']==objective
            for field,value in expected.items():
                assert metadata[field]==value,(field,metadata[field])
            states=data['head']
            assert len(states)==16
            assert sum(t.numel() for t in states.values())==266667
            for key,tensor in states.items():
                assert torch.isfinite(tensor).all(),key
                assert list(tensor.shape)==metadata['parameters'][key]['shape']
                assert torch.equal(tensor,data['model'][key])
            opt=data['optimizer']['state']
            assert len(opt)==16
            optimizer_steps=[]
            for state in opt.values():
                optimizer_steps.append(int(state['step']))
                assert int(state['step'])==updates
                for value in state.values():
                    if torch.is_tensor(value):
                        assert torch.isfinite(value).all()
            assert data['scheduler']['last_epoch']==updates
            assert set(data['rng'])=={'python','numpy','torch','cuda'}
            assert len(data['rng']['cuda'])==2
            assert data['scaler']['scale']>0
            history_updates=data['history'][-1]['successful_updates']
            if name=='fixed_budget_3600.pt':
                assert history_updates==3150, 'Unexpected fixed-endpoint embedded history accounting'
            else:
                assert history_updates==updates
            selected_history=max(h.get('selection_ap',h['val_ap']) for h in data['history'])
            assert data['best_ap']==selected_history
            partitions=[set(metadata[f'{p}_speakers']) for p in ['fit','selection','calibration','evaluation']]
            for i,a in enumerate(partitions):
                for b in partitions[i+1:]:
                    assert a.isdisjoint(b)
            assert partitions[-1]=={'NJS','RRBI'}
            results[f'{arm}/{name}']=dict(status='VERIFIED_LOCAL_PAYLOAD_AND_RECOVERY_IDENTITY',
                sha256=actual,objective=objective,successful_updates=updates,skipped_updates=0,
                micro_steps=data['micro_steps'],epoch=data['epoch'],optimizer_step_counters=optimizer_steps,
                scheduler_last_epoch=data['scheduler']['last_epoch'],parameter_count=266667,
                embedded_history_last_update=history_updates,
                history_qualification=('Fixed endpoint saved before final epoch history append; embedded history ends3150, while optimizer/scheduler/payload counters are3600. External history/last checkpoint contain3600.' if name=='fixed_budget_3600.pt' else None),
                all_head_and_optimizer_tensors_finite=True,rng_components=list(data['rng']),
                metadata=metadata,source_receipt_sha256=sha(receipt_path))
    verification=dict(status='VERIFIED',this_process_optimizer_steps=0,this_process_backward_passes=0,
        source_sha256=sha(__file__),checkpoints=results)
    recovery=dict(status='REQUIRED_ARTIFACTS_RECOVERED_AND_REVERIFIED',sources=inventory,
        source_sha256=sha(__file__),notes=[
            'Arm recovery reports retained as historical agent evidence, not blanket scientific approval.',
            'B-Control report identical-gradient wording is imprecise: gradients differ by within-error softmax weights; subtype semantics remain unidentified.',
            'Recovery timestamp strings have inconsistent UTC/local formatting; do not use them as training event times. Use checkpoint metadata created_at_utc and original logs.',
            'Kaggle notebooks reported is_private=false; no new sensitive artifact uploaded to them. HF mirror verified private.'])
    args.output.write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
    args.recovery_manifest.write_text(json.dumps(recovery,indent=2,allow_nan=False)+'\n')
    print({key:value['successful_updates'] for key,value in results.items()})


if __name__=='__main__':
    main()
