"""Clean-room PM2.5-GNN with graph-recurrent history and future covariates."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from tsflab.models._components.graph_conv_gru import graph_gru_step
from tsflab.models._components.marks import coerce_time_length, future_time_features, to_spatiotemporal

class GraphGRUCell(nn.Module):
    def __init__(self,input_width:int,hidden:int)->None:
        super().__init__(); self.hidden=hidden; self.gates=nn.Linear(2*(input_width+hidden),2*hidden); self.candidate=nn.Linear(2*(input_width+hidden),hidden)
    def _graph_features(self,joined,graph):
        return torch.cat((joined,torch.einsum("nm,bmd->bnd",graph,joined)),-1)
    def forward(self,x,h,graph):
        # Shared graph-GRU gating; each map is a Linear over [z, A z] of the joined input z.
        return graph_gru_step(x,h,lambda joined:self.gates(self._graph_features(joined,graph)),lambda joined:self.candidate(self._graph_features(joined,graph)))

class Model(nn.Module):
    def __init__(self,seq_len:int,pred_len:int,enc_in:int,adj_mx:np.ndarray|None=None,cov_dim:int=2,hid_dim:int=64)->None:
        super().__init__()
        if min(seq_len,pred_len,enc_in,hid_dim)<1 or cov_dim<0: raise ValueError("invalid PM25_GNN dimensions")
        self.seq_len,self.pred_len,self.enc_in,self.cov_dim=seq_len,pred_len,enc_in,cov_dim
        adj=np.eye(enc_in,dtype=np.float32) if adj_mx is None else np.asarray(adj_mx,dtype=np.float32)
        if adj.shape!=(enc_in,enc_in): raise ValueError(f"adjacency must have shape {(enc_in,enc_in)}")
        adj=adj+np.eye(enc_in,dtype=np.float32); adj/=np.maximum(adj.sum(-1,keepdims=True),1e-6); self.register_buffer("graph",torch.from_numpy(adj))
        self.history_projection=nn.Linear(1+cov_dim,hid_dim); self.encoder=GraphGRUCell(hid_dim,hid_dim)
        self.future_projection=nn.Linear(1+cov_dim,hid_dim); self.decoder=GraphGRUCell(hid_dim,hid_dim); self.output=nn.Linear(hid_dim,1)
    def _history(self,x_enc,x_mark_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in): raise ValueError(f"x_enc must have shape [B,{self.seq_len},{self.enc_in}]")
        st=to_spatiotemporal(x_enc,x_mark_enc); needed=1+self.cov_dim
        if st.shape[-1]<needed: st=torch.cat((st,st.new_zeros(*st.shape[:-1],needed-st.shape[-1])),-1)
        return st,needed
    def _future_covariates(self,x_enc,x_mark_enc,x_mark_dec):
        source_marks=x_mark_dec if x_mark_dec is not None else x_mark_enc
        marks=coerce_time_length(source_marks,self.pred_len) if source_marks is not None else None
        cov=future_time_features(marks,self.enc_in) if marks is not None else x_enc.new_zeros(x_enc.shape[0],self.pred_len,self.enc_in,self.cov_dim)
        if cov.shape[-1]<self.cov_dim: cov=torch.cat((cov,cov.new_zeros(*cov.shape[:-1],self.cov_dim-cov.shape[-1])),-1)
        return cov
    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        st,needed=self._history(x_enc,x_mark_enc)
        hidden=x_enc.new_zeros(x_enc.shape[0],self.enc_in,self.encoder.hidden)
        for step in range(self.seq_len): hidden=self.encoder(self.history_projection(st[:,step,:,:needed]),hidden,self.graph)
        cov=self._future_covariates(x_enc,x_mark_enc,x_mark_dec)
        previous=x_enc[:,-1]; outputs=[]
        for step in range(self.pred_len):
            driver=self.future_projection(torch.cat((previous.unsqueeze(-1),cov[:,step,:,:self.cov_dim]),-1)); hidden=self.decoder(driver,hidden,self.graph)
            previous=self.output(hidden).squeeze(-1); outputs.append(previous)
        return torch.stack(outputs,1)
