"""Frozen-backbone SM13 pooling with explicit geometry and cache provenance.

Inference API accepts hidden states and frozen predicted alignment only, never
an annotation row, gold timestamps, error types or realized phone supervision.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class Geometry:
    hidden_dim: int
    layer_indices: tuple[int,...]
    hidden_state_count: int

    def __post_init__(self):
        if self.hidden_dim<=0 or self.hidden_state_count<=0:
            raise ValueError('positive backbone geometry required')
        if (len(self.layer_indices)!=13 or len(set(self.layer_indices))!=13 or
                tuple(sorted(self.layer_indices))!=self.layer_indices or
                any(not isinstance(i,int) or not 0<=i<self.hidden_state_count for i in self.layer_indices)):
            raise ValueError('exactly13 unique ordered registered layers required')

    @property
    def target_dim(self): return 3*self.hidden_dim

    @property
    def gap_dim(self): return 2*self.hidden_dim


W2V2=Geometry(768,tuple(range(13)),13)
WAVLM=Geometry(1024,tuple(range(0,25,2)),25)


def as_hidden(value):
    if hasattr(value,'detach'):
        if value.requires_grad: raise ValueError('frozen inference hidden states required')
        value=value.detach().float().cpu().numpy()
    value=np.asarray(value)
    if value.ndim!=3 or value.shape[0]!=1 or not value.shape[1] or value.dtype!=np.float32:
        raise ValueError('float32 batch1 hidden states required')
    if not np.isfinite(value).all(): raise ValueError('nonfinite hidden representation')
    return value[0]


def extract(hidden_states,centers,alignment_frames,geometry):
    if len(hidden_states)!=geometry.hidden_state_count:
        raise ValueError('backbone hidden-state inventory changed')
    centers=np.asarray(centers,dtype=np.float32)
    if centers.ndim!=1 or not np.isfinite(centers).all(): raise ValueError('invalid predicted centers')
    if not isinstance(alignment_frames,(int,np.integer)) or alignment_frames<=0:
        raise ValueError('positive alignment clock required')
    if np.any(centers<0) or np.any(centers>=alignment_frames): raise ValueError('alignment outside valid clock')
    # Predicted fallback centers may be nonmonotonic; preserve legacy exact order,
    # report their presence in the audit instead of silently sorting/dropping IDs.
    targets=[];gaps=[];frames=None
    for index in geometry.layer_indices:
        hidden=as_hidden(hidden_states[index])
        if hidden.shape[1]!=geometry.hidden_dim: raise ValueError('backbone hidden width changed')
        if frames is not None and len(hidden)!=frames: raise ValueError('hidden clocks disagree across layers')
        frames=len(hidden)
        scaled=np.rint(centers*(frames/alignment_frames)).astype(int)
        ix=np.clip(scaled[:,None]+np.arange(-8,9)[None,:],0,frames-1)
        windows=hidden[ix]
        central=windows[:,3:14,:]
        target=np.concatenate([central.mean(1),central.std(1,ddof=0),central[:,-1]-central[:,0]],axis=1)
        left=np.zeros((len(centers)+1,geometry.hidden_dim),np.float32)
        right=np.zeros_like(left)
        if len(centers):
            left[1:]=windows[:,11:16,:].mean(1)
            right[:-1]=windows[:,1:6,:].mean(1)
        targets.append(target);gaps.append(np.concatenate([left,right],axis=1))
    return np.stack(targets,axis=1),np.stack(gaps,axis=1)


def cache_identity(*,model_sha256,config_sha256,normalization,alignment_sha256,manifest_sha256,
                   ordered_ids_sha256,geometry,source_sha256,dtype='float32'):
    for digest in [model_sha256,config_sha256,alignment_sha256,manifest_sha256,ordered_ids_sha256,source_sha256]:
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('exact lowercase SHA256 provenance required')
    if dtype not in ['float32','float16'] or not normalization:
        raise ValueError('declared normalization/precision required')
    record={'model_sha256':model_sha256,'config_sha256':config_sha256,'normalization':normalization,
            'alignment_sha256':alignment_sha256,'manifest_sha256':manifest_sha256,
            'ordered_ids_sha256':ordered_ids_sha256,'geometry':asdict(geometry),
            'source_sha256':source_sha256,'dtype':dtype,'pooling':'SM13_RADIUS8_TARGET3:14_GAP11:16_1:6_V1'}
    encoded=json.dumps(record,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(encoded).hexdigest(),record


def require_independent_backbone(fit_speakers,evaluation_speakers,*,evaluation_scope):
    overlap=set(fit_speakers)&set(evaluation_speakers)
    if overlap:
        raise ValueError(f'backbone already fit evaluation speakers: {sorted(overlap)}')
    if evaluation_scope not in ['EXPLORATORY_DEV_SEARCH','INDEPENDENT_GROUPED_TRAIN_OOF']:
        raise ValueError('unregistered evaluation scope; test remains sealed')
    return True
