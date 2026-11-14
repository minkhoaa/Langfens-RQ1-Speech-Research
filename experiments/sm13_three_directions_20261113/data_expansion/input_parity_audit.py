from __future__ import annotations
import io, json, sys, zipfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel
ROOT=Path('/home/khoa/Projects/New-Method-Research')
sys.path.insert(0,str(ROOT/'experiments/rq1_encoder_bench_01/common'))
from dataset_util import DatasetManager
from extract_pseudo_features import extract_target_windows

torch.set_num_threads(2)
uid='ABA_arctic_a0003'
record=next(json.loads(x) for x in (ROOT/'data/derived/l2arctic/manifests/train.jsonl').read_text().splitlines() if json.loads(x)['utterance_id']==uid)
archive=ROOT/'experiments/sm13_three_directions_20261113/data_expansion/raw_archives/ABA.zip'
with zipfile.ZipFile(archive) as zf:
    raw_blob=zf.read('ABA/wav/arctic_a0003.wav')
raw,sr=sf.read(io.BytesIO(raw_blob),dtype='float32',always_2d=True)
raw=raw.mean(1)
if sr!=16000:
    resampled=torchaudio.transforms.Resample(orig_freq=int(sr),new_freq=16000)(torch.from_numpy(raw).unsqueeze(0))
    encoded=io.BytesIO();torchaudio.save(encoded,resampled,16000,format='wav');encoded.seek(0)
    raw,sr_out=sf.read(encoded,dtype='float32',always_2d=False)
    assert sr_out==16000
local,_=sf.read(ROOT/record['audio_path'],dtype='float32',always_2d=True)
local=local.mean(1).astype(np.float32)
fe=AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h',local_files_only=True)
assert fe.do_normalize
xraw=fe(raw,sampling_rate=16000,return_tensors='pt')['input_values']
xlocal=fe(local,sampling_rate=16000,return_tensors='pt')['input_values']
model=AutoModel.from_pretrained('facebook/wav2vec2-base-960h',local_files_only=True).eval()
with torch.inference_mode():
    hraw=model(xraw,output_hidden_states=True).hidden_states
    hloc=model(xlocal,output_hidden_states=True).hidden_states
align=np.load(ROOT/'experiments/rq1_encoder_bench_01/features/alignment_centers.npz')
centers=align[f'centers__{uid}']; nframes=int(align[f'nframes__{uid}'])
fraw=extract_target_windows(hraw,centers,nframes)
floc=extract_target_windows(hloc,centers,nframes)
dm=DatasetManager(); sl=dm.target_slices[uid]
cache=np.stack([np.load(ROOT/f'experiments/rq1_encoder_bench_01/features/target_acoustic_w2v2_960h_l{i}.npy',mmap_mode='r')[sl] for i in range(13)],axis=1)
def metrics(a,b):
 d=a.astype(np.float64)-b.astype(np.float64)
 denom=np.linalg.norm(a.ravel())*np.linalg.norm(b.ravel())
 return {'max_abs':float(np.max(np.abs(d))),'rmse':float(np.sqrt(np.mean(d*d))),'cosine':float(np.dot(a.ravel(),b.ravel())/denom)}
result={'probe_uid':uid,'archive_sr':int(sr),'archive_samples_after_resample':len(raw),'local_samples':len(local),'feature_extractor_do_normalize':fe.do_normalize,'waveform_resample_match':metrics(raw,local),'normalized_input_match':metrics(xraw.numpy(),xlocal.numpy()),'cached_layer6_pool_match_from_resampled_source':metrics(fraw[:,6,:],cache[:,6,:]),'cached_layer6_pool_match_from_local_source':metrics(floc[:,6,:],cache[:,6,:]),'all_layer_rmse_from_local':{str(i):metrics(floc[:,i,:],cache[:,i,:]) for i in range(13)}}
out=ROOT/'experiments/sm13_three_directions_20261113/data_expansion/INPUT_PARITY_AUDIT.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
