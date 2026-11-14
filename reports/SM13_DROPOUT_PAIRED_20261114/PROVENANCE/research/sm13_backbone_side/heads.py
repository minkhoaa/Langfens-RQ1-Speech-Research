"""SM13-style head capacity/recipe preserved except registered input geometry."""
import torch
from torch import nn

from .features import Geometry


class ScalarMixingTargetHead(nn.Module):
    def __init__(self,geometry: Geometry):
        super().__init__();self.geometry=geometry
        self.layer_logits=nn.Parameter(torch.zeros(13));self.scale=nn.Parameter(torch.ones(1))
        self.net=nn.Sequential(nn.Linear(geometry.target_dim+39,256),nn.ReLU(),nn.Dropout(.25),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(.20),nn.Linear(128,3))

    def forward(self,features,refs):
        if features.ndim!=3 or features.shape[1:]!=(13,self.geometry.target_dim) or refs.shape!=(len(features),39):
            raise ValueError('target feature/reference geometry mismatch')
        mixed=torch.einsum('l,bli->bi',torch.softmax(self.layer_logits,0),features)*self.scale
        return self.net(torch.cat([mixed,refs],dim=-1))


class ScalarMixingGapHead(nn.Module):
    def __init__(self,geometry: Geometry):
        super().__init__();self.geometry=geometry
        self.layer_logits=nn.Parameter(torch.zeros(13));self.scale=nn.Parameter(torch.ones(1))
        self.net=nn.Sequential(nn.Linear(geometry.gap_dim+78,128),nn.GELU(),nn.Dropout(.15),
                               nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))

    def forward(self,features,refs):
        if features.ndim!=3 or features.shape[1:]!=(13,self.geometry.gap_dim) or refs.shape!=(len(features),78):
            raise ValueError('gap feature/reference geometry mismatch')
        mixed=torch.einsum('l,bli->bi',torch.softmax(self.layer_logits,0),features)*self.scale
        return self.net(torch.cat([mixed,refs],dim=-1))


def parameter_counts(geometry):
    # Analytic counts: avoid allocating/random-initializing models during audit.
    t=14+(geometry.target_dim+39)*256+256+256*128+128+128*3+3
    g=14+(geometry.gap_dim+78)*128+128+128*64+64+64*2+2
    return {'target':t,'gap':g,'total':t+g}
