from __future__ import annotations

import torch
from torch import nn

from .model import MarketForgeM3T


class MarketForgeM3TV2(MarketForgeM3T):
    """Corrected v2 head: no averaged markout or scalar fill/adverse target."""

    def __init__(
        self,
        state_dim: int,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        fusion: str = "gated",
        num_sources: int = 8,
        num_instruments: int = 64,
    ) -> None:
        super().__init__(state_dim, d_model, layers, heads, fusion)
        self.source_embedding = nn.Embedding(num_sources, d_model)
        self.instrument_embedding = nn.Embedding(num_instruments, d_model)
        self.modality_projection = nn.Linear(3, d_model)
        self.economic_head = nn.Linear(d_model, 9)

    @staticmethod
    def _embedding_with_unseen_mean(
        embedding: nn.Embedding, identifiers: torch.Tensor
    ) -> torch.Tensor:
        """Map IDs outside the fitted vocabulary to a frozen, neutral seen-market mean.

        This permits a genuinely unseen source/instrument at evaluation without
        expanding or tuning the development checkpoint after its vocabulary was
        frozen. Negative IDs and IDs >= the fitted vocabulary are both unknown.
        """
        known = (identifiers >= 0) & (identifiers < embedding.num_embeddings)
        safe = identifiers.clamp(0, embedding.num_embeddings - 1)
        values = embedding(safe)
        unseen = embedding.weight.mean(dim=0).expand_as(values)
        return torch.where(known.unsqueeze(-1), values, unseen)

    def encode(
        self,
        action: torch.Tensor,
        side: torch.Tensor,
        event_cont: torch.Tensor,
        state: torch.Tensor,
        book: torch.Tensor,
        source_id: torch.Tensor | None = None,
        instrument_id: torch.Tensor | None = None,
        modality: torch.Tensor | None = None,
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
        batch = action.shape[0]
        source_id = (
            torch.zeros(batch, dtype=torch.long, device=action.device)
            if source_id is None
            else source_id
        )
        instrument_id = (
            torch.zeros(batch, dtype=torch.long, device=action.device)
            if instrument_id is None
            else instrument_id
        )
        modality = (
            torch.ones(batch, 3, dtype=fused.dtype, device=action.device)
            if modality is None
            else modality.to(fused.dtype)
        )
        context = (
            self._embedding_with_unseen_mean(self.source_embedding, source_id)
            + self._embedding_with_unseen_mean(self.instrument_embedding, instrument_id)
            + self.modality_projection(modality)
        )
        fused = fused + context[:, None, :]
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
        source_id: torch.Tensor | None = None,
        instrument_id: torch.Tensor | None = None,
        modality: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(
            action, side, event_cont, state, book, source_id, instrument_id, modality
        )[:, -1]
        economic = self.economic_head(hidden)
        return {
            "direction_logits": self.direction_head(hidden),
            "return_mean": economic[:, 0],
            "return_log_scale": economic[:, 1].clamp(-12, 4),
            "bid_quote_markout": economic[:, 2],
            "ask_quote_markout": economic[:, 3],
            "volatility": economic[:, 4].abs(),
            "bid_fill_logit": economic[:, 5],
            "ask_fill_logit": economic[:, 6],
            "bid_adverse_logit": economic[:, 7],
            "ask_adverse_logit": economic[:, 8],
        }
