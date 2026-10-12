"""Clean-room MegaCRN with a meta-node bank and recurrent meta-graphs."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from tsflab.models._components.graph_conv_gru import graph_gru_step
from tsflab.models._components.marks import coerce_time_length, future_time_features, to_spatiotemporal

class MetaGraphCell(nn.Module):
    def __init__(self,input_width:int,hidden:int,cheb_k:int)->None:
        super().__init__(); self.hidden,self.cheb_k=hidden,cheb_k; features=(input_width+hidden)*(cheb_k+1)
        self.gates,self.candidate=nn.Linear(features,2*hidden),nn.Linear(features,hidden)
    def _features(self,joined,graph):
        states=[joined]; current=joined
        for _ in range(self.cheb_k): current=torch.einsum("bnm,bmd->bnd",graph,current); states.append(current)
        return torch.cat(states,-1)
    def forward(self,x,h,graph):
        # Shared graph-GRU gating; each map is a Linear over [z, G z, .., G^K z] of the joined input z.
        return graph_gru_step(x,h,lambda joined:self.gates(self._features(joined,graph)),lambda joined:self.candidate(self._features(joined,graph)))

class Model(nn.Module):
    def __init__(self,seq_len:int,pred_len:int,num_nodes:int,adj_mx=None,input_dim:int=3,rnn_units:int=32,
                 num_layers:int=1,cheb_k:int=3,mem_num:int=8,mem_dim:int=16)->None:
        super().__init__()
        if min(seq_len,pred_len,num_nodes,input_dim,rnn_units,num_layers,cheb_k,mem_num,mem_dim)<1: raise ValueError("invalid MegaCRN dimensions")
        self.seq_len,self.pred_len,self.num_nodes,self.input_dim=seq_len,pred_len,num_nodes,input_dim
        self.memory=nn.Parameter(torch.randn(mem_num,mem_dim)/mem_dim**0.5); self.node_to_memory=nn.Parameter(torch.randn(num_nodes,mem_num)/mem_num**0.5)
        self.query=nn.Linear(rnn_units,mem_dim); self.encoder_input=nn.Linear(input_dim,rnn_units)
        self.encoder=nn.ModuleList(MetaGraphCell(rnn_units,rnn_units,cheb_k) for _ in range(num_layers))
        self.decoder_input=nn.Linear(1+mem_dim,rnn_units); self.decoder=nn.ModuleList(MetaGraphCell(rnn_units,rnn_units,cheb_k) for _ in range(num_layers)); self.output=nn.Linear(rnn_units,1)
        self.graph_mix=nn.Parameter(torch.tensor(0.5))
        adj=np.eye(num_nodes,dtype=np.float32) if adj_mx is None else np.asarray(adj_mx,dtype=np.float32)
        if adj.shape!=(num_nodes,num_nodes): raise ValueError(f"adjacency must have shape {(num_nodes,num_nodes)}")
        adj=adj+np.eye(num_nodes,dtype=np.float32); adj/=np.maximum(adj.sum(-1,keepdims=True),1e-6); self.register_buffer("predefined_graph",torch.from_numpy(adj))
        self.last_memory_attention=None; self.last_meta_graph=None
    def _meta_graph(self,hidden):
        query=self.query(hidden); attention=torch.softmax(torch.einsum("bnd,md->bnm",query,self.memory)+self.node_to_memory.unsqueeze(0),-1)
        node_memory=torch.einsum("bnm,md->bnd",attention,self.memory); learned=torch.softmax(torch.einsum("bnd,bmd->bnm",node_memory,node_memory)/node_memory.shape[-1]**0.5,-1)
        weight=torch.sigmoid(self.graph_mix); graph=weight*learned+(1-weight)*self.predefined_graph.unsqueeze(0)
        self.last_memory_attention,self.last_meta_graph=attention,graph; return graph,node_memory
    def _inputs(self,x_enc,x_mark_enc):
        if x_enc.shape[1:]!=(self.seq_len,self.num_nodes): raise ValueError(f"x_enc must have shape [B,{self.seq_len},{self.num_nodes}]")
        st=to_spatiotemporal(x_enc,x_mark_enc)
        if st.shape[-1]<self.input_dim: st=torch.cat((st,st.new_zeros(*st.shape[:-1],self.input_dim-st.shape[-1])),-1)
        return st
    @staticmethod
    def _step(cells,value,states,graph):
        for index,cell in enumerate(cells): states[index]=cell(value,states[index],graph); value=states[index]
        return value
    def _future(self,x_enc,x_mark_enc,x_mark_dec):
        source_marks=x_mark_dec if x_mark_dec is not None else x_mark_enc
        marks=coerce_time_length(source_marks,self.pred_len) if source_marks is not None else None
        return future_time_features(marks,self.num_nodes)[...,0] if marks is not None else x_enc.new_zeros(x_enc.shape[0],self.pred_len,self.num_nodes)
    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        st=self._inputs(x_enc,x_mark_enc)
        states=[x_enc.new_zeros(x_enc.shape[0],self.num_nodes,cell.hidden) for cell in self.encoder]; graph=self.predefined_graph.unsqueeze(0).expand(x_enc.shape[0],-1,-1)
        for step in range(self.seq_len):
            self._step(self.encoder,self.encoder_input(st[:,step,:,:self.input_dim]),states,graph)
            graph,_=self._meta_graph(states[-1])
        graph,memory=self._meta_graph(states[-1]); decoder_states=[state.clone() for state in states]; previous=x_enc[:,-1]
        future=self._future(x_enc,x_mark_enc,x_mark_dec)
        outputs=[]
        for step in range(self.pred_len):
            value=self.decoder_input(torch.cat((previous.unsqueeze(-1)+future[:,step].unsqueeze(-1),memory),-1))
            value=self._step(self.decoder,value,decoder_states,graph)
            previous=self.output(value).squeeze(-1); outputs.append(previous)
        return torch.stack(outputs,1)
