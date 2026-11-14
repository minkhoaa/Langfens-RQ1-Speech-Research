"""Opt-in conditioning candidate for NEW registered SM13 head experiments.

Never modifies historical caches/heads. Statistics are fit-speaker-only and
checkpointed as buffers. Existing heads are NOT compatible with this wrapper;
a fresh matched refit is needed before any quality claim.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .heads import ScalarMixingTargetHead,ScalarMixingGapHead


def fit_layer_rms(features,speakers,*,fit_speakers,held_speakers,batch=128):
    """One scalar RMS per layer across fit events and acoustic coordinates.

No labels, held features, per-phone/gold timing or test data are used. Equal
layer RMS does not imply whitening, decorrelation or matched head capacity.
"""
    fit_speakers=set(fit_speakers);held_speakers=set(held_speakers)
    speakers=np.asarray(speakers)
    if (not fit_speakers or not held_speakers or fit_speakers&held_speakers or
            speakers.ndim!=1 or len(speakers)!=features.shape[0] or
            set(speakers)!=fit_speakers|held_speakers or
            len(features.shape)!=3 or features.shape[1]!=13 or features.shape[2]<=0 or
            not isinstance(batch,int) or batch<=0):
        raise ValueError('explicit disjoint complete fit/held speaker population required')
    indices=np.flatnonzero(np.isin(speakers,sorted(fit_speakers)))
    square_sum=np.zeros(13,np.float64)
    for start in range(0,len(indices),batch):
        values=np.asarray(features[indices[start:start+batch]],dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('nonfinite fit features')
        square_sum+=np.sum(values*values,axis=(0,2))
    rms=np.sqrt(square_sum/(len(indices)*features.shape[2]))
    if not np.isfinite(rms).all() or np.any(rms<=0):
        raise ValueError('zero/nonfinite fit layer RMS; do not hide a collapsed representation')
    return rms


class FitLayerRMS(nn.Module):
    """Pure frozen scaling, fit once externally; never learns from inference."""
    def __init__(self,rms):
        super().__init__()
        rms=torch.as_tensor(np.array(rms,copy=True),dtype=torch.float32)
        if rms.shape!=(13,) or not torch.isfinite(rms).all() or torch.any(rms<=0):
            raise ValueError('13 positive finite fit-only layer scales required')
        self.register_buffer('rms',rms)

    def forward(self,features):
        if features.ndim!=3 or features.shape[1]!=13:
            raise ValueError('batch13layerwidth input required')
        return features/self.rms[None,:,None]


class ConditionedSM13Head(nn.Module):
    """Fresh candidate head; no silent conversion of any trained legacy model."""
    def __init__(self,kind,geometry,rms):
        super().__init__()
        if kind not in ['target','gap']:raise ValueError('registered target/gap kind required')
        self.normalizer=FitLayerRMS(rms)
        self.head=(ScalarMixingTargetHead if kind=='target' else ScalarMixingGapHead)(geometry)

    def forward(self,features,refs):
        return self.head(self.normalizer(features),refs)
