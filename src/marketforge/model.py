from __future__ import annotations

import torch
from torch import nn


class OrderedBookEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.level_projection = nn.Linear(6, d_model)
        self.level_embedding = nn.Parameter(torch.randn(10, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, book: torch.Tensor) -> torch.Tensor:
        encoded = self.level_projection(book) + self.level_embedding
        return self.norm(encoded).mean(dim=-2)


class MarketForgeM3T(nn.Module):
    def __init__(
        self,
        state_dim: int,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        fusion: str = "gated",
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.action_embedding = nn.Embedding(8, 8)
        self.side_embedding = nn.Embedding(3, 4)
        self.event_projection = nn.Linear(8 + 4 + 3, d_model)
        self.state_projection = nn.Linear(state_dim, d_model)
        self.book_encoder = OrderedBookEncoder(d_model)
        self.state_fusion = nn.Linear(d_model * 2, d_model)
        if fusion == "concat":
            self.stream_fusion = nn.Linear(d_model * 2, d_model)
        elif fusion == "gated":
            self.gate = nn.Linear(d_model * 2, d_model)
        elif fusion == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        else:
            raise ValueError(fusion)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.direction_head = nn.Linear(d_model, 3)
        self.economic_head = nn.Linear(d_model, 5)

    def encode(
        self,
        action: torch.Tensor,
        side: torch.Tensor,
        event_cont: torch.Tensor,
        state: torch.Tensor,
        book: torch.Tensor,
    ) -> torch.Tensor:
        event = self.event_projection(
            torch.cat(
                [self.action_embedding(action), self.side_embedding(side), event_cont], dim=-1
            )
        )
        state_token = self.state_fusion(
            torch.cat([self.state_projection(state), self.book_encoder(book)], dim=-1)
        )
        if self.fusion == "concat":
            fused = self.stream_fusion(torch.cat([event, state_token], dim=-1))
        elif self.fusion == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([event, state_token], dim=-1)))
            fused = gate * event + (1 - gate) * state_token
        else:
            attended, _ = self.cross_attention(event, state_token, state_token, need_weights=False)
            fused = event + attended
        length = fused.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=fused.device, dtype=torch.bool), diagonal=1
        )
        return self.backbone(fused, mask=causal_mask)

    def forward(
        self,
        action: torch.Tensor,
        side: torch.Tensor,
        event_cont: torch.Tensor,
        state: torch.Tensor,
        book: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(action, side, event_cont, state, book)[:, -1]
        economic = self.economic_head(hidden)
        return {
            "direction_logits": self.direction_head(hidden),
            "expected_return": economic[:, 0],
            "expected_markout": economic[:, 1],
            "volatility": economic[:, 2].abs(),
            "fill_logit": economic[:, 3],
            "adverse_logit": economic[:, 4],
        }


class SSLHeads(nn.Module):
    def __init__(self, d_model: int, state_dim: int) -> None:
        super().__init__()
        self.masked_action = nn.Linear(d_model, 7)
        self.next_action = nn.Linear(d_model, 7)
        self.future_state = nn.Linear(d_model, state_dim)
        self.contrastive = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 32)
        )
