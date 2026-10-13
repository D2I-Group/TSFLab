"""Two-stage adapter of the archived StateFlow reproduction, with archived defaults and explicit structural search factors."""
import torch
from torch import nn

from tsflab.models._inputs import check_window


class StateFlowCore(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, hidden_size=32, memory_size=16,
                 chunk_width=5, chunk_stride=2):
        super().__init__()
        self.hidden_size, self.memory_size = hidden_size, memory_size
        self.chunk_width, self.chunk_stride = chunk_width, chunk_stride
        self.hidden = nn.Linear(1 + hidden_size + memory_size, hidden_size)
        self.base = nn.Linear(hidden_size, 1)
        self.memory = nn.Linear(1 + memory_size, memory_size)
        self.chunk_h = nn.Linear(chunk_width * hidden_size, hidden_size)
        self.chunk_e = nn.Linear(chunk_width * memory_size, memory_size)
        self.head = nn.Linear((hidden_size + memory_size) * ((seq_len - 1 - chunk_width) // chunk_stride + 1), pred_len)
        self.chunks = ((seq_len - 1 - chunk_width) // chunk_stride + 1)

    def encode(self, x: torch.Tensor):
        # x: [batch, lookback], instance-normalized
        b, length = x.shape
        h = x.new_zeros(b, self.hidden_size)
        e = x.new_zeros(b, self.memory_size)
        hs, es, base_preds = [], [], []
        for t in range(length - 1):
            h = torch.relu(self.hidden(torch.cat((x[:, t:t+1], h, e), dim=-1)))
            pred = self.base(h)
            residual = x[:, t+1:t+2] - pred
            e = torch.tanh(self.memory(torch.cat((residual, e), dim=-1)))
            hs.append(h)
            es.append(e)
            base_preds.append(pred.squeeze(-1))
        return torch.stack(hs, 1), torch.stack(es, 1), torch.stack(base_preds, 1)

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True).detach()
        std = (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        normalized = (x - mean) / std
        h, e, _ = self.encode(normalized)
        # [B, chunks, width * state_dim]; archived defaults are width=5, stride=2
        hp = h.unfold(1, self.chunk_width, self.chunk_stride).contiguous().reshape(h.shape[0], self.chunks, -1)
        ep = e.unfold(1, self.chunk_width, self.chunk_stride).contiguous().reshape(e.shape[0], self.chunks, -1)
        qh, qe = torch.relu(self.chunk_h(hp)), torch.relu(self.chunk_e(ep))
        pred = self.head(torch.cat((qh, qe), -1).flatten(1))
        return pred * std + mean


class _EncoderStage(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    @staticmethod
    def normalized(values):
        flat = values.transpose(1, 2).reshape(-1, values.shape[1])
        mean = flat.mean(-1, keepdim=True).detach()
        std = (flat.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        return (flat - mean) / std

    def forward(self, values):
        return self.core.encode(self.normalized(values))[2]

    @classmethod
    def loss(cls, predictions, values):
        return (predictions - cls.normalized(values)[:, 1:]).square().mean()


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, encoder_epochs=10, encoder_lr=0.001,
                 hidden_size=32, memory_size=16, chunk_width=5, chunk_stride=2):
        super().__init__()
        self.core = StateFlowCore(seq_len, pred_len, hidden_size, memory_size, chunk_width, chunk_stride)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.encoder_epochs, self.encoder_lr = encoder_epochs, encoder_lr
        self._encoder_frozen = False

    def pretrain(self, train_loader, device, *, stage_runner=None):
        if self._encoder_frozen:
            return
        from tsflab.experiments.infra.stages import reconstruction_stage
        stage = _EncoderStage(self.core).to(device)
        parameters = [p for module in (self.core.hidden, self.core.base, self.core.memory)
                      for p in module.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=self.encoder_lr)
        (stage_runner or reconstruction_stage)(stage, train_loader, device, optimizer,
                                                stage.loss, self.encoder_epochs)
        for parameter in parameters:
            parameter.requires_grad_(False)
        self._encoder_frozen = True

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        flat = x_enc.transpose(1, 2).reshape(-1, self.seq_len)
        return self.core(flat).reshape(batch, self.enc_in, self.pred_len).transpose(1, 2)
