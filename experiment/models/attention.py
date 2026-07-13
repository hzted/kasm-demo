import torch
import torch.nn as nn


class AttentiveAggregation(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.hidden_map = nn.Sequential(nn.Linear(input_size, input_size), nn.Tanh())

    def forward(self, embs, emb_mask, q):
        hidden_size = torch.tensor(q.shape[1], dtype=torch.float, device=q.device)
        attn_logit = torch.sum(self.hidden_map(embs) * q[:, None, :], dim=2)
        attn_logit = torch.where(emb_mask, attn_logit, torch.ones_like(attn_logit) * -torch.inf)
        attn_score = torch.softmax(attn_logit / torch.sqrt(hidden_size), dim=1)
        return torch.sum(attn_score[:, :, None] * embs, dim=1)


class LocalAttentiveAggregation(nn.Module):
    def __init__(self, input_size, query_cls=False) -> None:
        super().__init__()
        self.local_pooling = AttentiveAggregation(input_size)
        self.query_index = 0 if query_cls else 1

    def forward(self, embs, emb_mask):
        bsz, num_sent, _, hidden_size = embs.shape
        local_embs = torch.zeros((bsz, num_sent, hidden_size), dtype=embs.dtype, device=embs.device)
        for idx in range(num_sent):
            local_embs[:, idx, :] = self.local_pooling(
                embs[:, idx, :, :],
                emb_mask[:, idx, :],
                embs[:, idx, self.query_index, :],
            )
        return local_embs
