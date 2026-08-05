"""从注意力公式开始，逐层手写三类 Transformer 架构。
顺序：
1. scaled_dot_product_attention：注意力的数学核心
2. SelfAttention / MultiHeadAttention：参数化注意力
3. EncoderBlock / DecoderOnlyBlock / Seq2SeqDecoderBlock：基本积木
4. EncoderOnlyTransformer：BERT 类架构
5. DecoderOnlyTransformer：GPT 类架构
6. EncoderDecoderTransformer：原始 Transformer / T5 类架构

没有使用 nn.MultiheadAttention 或 nn.Transformer。所有张量均为 batch-first。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# ============================================================================
# Level 1：注意力的数学核心（没有可训练参数）
# ============================================================================


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """计算 softmax(QK^T / sqrt(d_k))V。

    Args:
        query: [..., query_len, d_k]
        key:   [..., key_len, d_k]
        value: [..., key_len, d_v]
        mask:  可广播到 [..., query_len, key_len] 的 bool 张量；True 表示可见。

    Returns:
        context: [..., query_len, d_v]
        weights: [..., query_len, key_len]，每行之和为 1。
    """

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.size(-1))

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask 必须是 bool 类型")
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    context = weights @ value
    return context, weights


def attention_formula_demo() -> None:
    """用三个二维 token 直观看一次 QK^T、softmax 和加权求和。"""

    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    key = query.clone()
    value = torch.tensor([[[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]])

    context, weights = scaled_dot_product_attention(query, key, value)
    print("\n[Level 1] 纯注意力公式")
    print("attention weights:\n", weights.round(decimals=3))
    print("context:\n", context.round(decimals=3))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3))


# ============================================================================
# Level 2：单头注意力——学习如何从输入生成 Q、K、V
# ============================================================================


class SelfAttention(nn.Module):
    """最小单头自注意力：Q、K、V 都来自同一个输入 x。"""

    def __init__(self, d_model: int, d_k: int | None = None) -> None:
        super().__init__()
        d_k = d_k or d_model
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_k, bias=False)
        self.output = nn.Linear(d_k, d_model, bias=False)

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        context, weights = scaled_dot_product_attention(
            self.query(x), self.key(x), self.value(x), mask
        )
        return self.output(context), weights


# ============================================================================
# Level 3：多头注意力——在多个表示子空间中并行关注
# ============================================================================


class MultiHeadAttention(nn.Module):
    """同时支持 self-attention 与 cross-attention。

    self-attention:  query、key、value 传同一个 x。
    cross-attention: query 来自 Decoder，key/value 来自 Encoder memory。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        # [B, L, D] -> [B, L, H, D/H] -> [B, H, L, D/H]
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: Tensor) -> Tensor:
        # [B, H, L, D/H] -> [B, L, H, D/H] -> [B, L, D]
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch, length, -1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        context, weights = scaled_dot_product_attention(
            q, k, v, mask, self.attention_dropout
        )
        return self.out_proj(self._merge_heads(context)), weights


# ============================================================================
# Level 4：Mask——控制一个 token 可以看到哪些 token
# ============================================================================


def make_padding_mask(token_ids: Tensor, pad_id: int = 0) -> Tensor:
    """屏蔽 key 侧 PAD：[B, L] -> [B, 1, 1, L]。"""

    return token_ids.ne(pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(token_ids: Tensor, pad_id: int = 0) -> Tensor:
    """同时屏蔽 PAD 和未来信息，供自回归 Decoder 使用。"""

    length = token_ids.size(1)
    padding = make_padding_mask(token_ids, pad_id)
    causal = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=token_ids.device)
    ).unsqueeze(0).unsqueeze(0)
    return padding & causal


# ============================================================================
# Level 5：Transformer 的其余公共积木
# ============================================================================


class PositionalEncoding(nn.Module):
    """固定正弦位置编码，让注意力知道 token 的顺序。"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(
            position * frequency[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class TokenEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, pad_id: int, dropout: float, max_len: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token(token_ids) * math.sqrt(self.d_model)
        return self.dropout(self.position(x))


class FeedForward(nn.Module):
    """逐 token 的非线性变换；它不负责 token 间通信。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class AddAndNorm(nn.Module):
    """把残差连接与 LayerNorm 显式封装，便于观察结构。"""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, residual: Tensor, sublayer_output: Tensor) -> Tensor:
        return self.norm(residual + self.dropout(sublayer_output))


# ============================================================================
# Level 6：三种架构所需的 Block
# ============================================================================


class EncoderBlock(nn.Module):
    """双向 self-attention + FFN，可看到左右两侧上下文。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        attended, _ = self.attention(x, x, x, mask)
        x = self.add_norm1(x, attended)
        return self.add_norm2(x, self.feed_forward(x))


class DecoderOnlyBlock(nn.Module):
    """带 causal mask 的 self-attention + FFN，GPT 的基本积木。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        attended, _ = self.attention(x, x, x, causal_mask)
        x = self.add_norm1(x, attended)
        return self.add_norm2(x, self.feed_forward(x))


class Seq2SeqDecoderBlock(nn.Module):
    """原始 Decoder：causal self-attention + cross-attention + FFN。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)
        self.add_norm3 = AddAndNorm(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        attended, _ = self.self_attention(x, x, x, target_mask)
        x = self.add_norm1(x, attended)
        attended, _ = self.cross_attention(x, memory, memory, source_mask)
        x = self.add_norm2(x, attended)
        return self.add_norm3(x, self.feed_forward(x))


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============================================================================
# Level 7A：Encoder-only（BERT 类）
# ============================================================================


class EncoderOnlyTransformer(nn.Module):
    """适合文本分类、序列标注和语义表示，不负责自回归生成。"""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = TokenEmbedding(vocab_size, d_model, pad_id, dropout, max_len)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(initialize_weights)

    def forward(self, token_ids: Tensor) -> Tensor:
        mask = make_padding_mask(token_ids, self.pad_id)
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, mask)
        # 教学约定：位置 0 是 CLS/BOS，使用它进行整句分类。
        return self.classifier(x[:, 0])


# ============================================================================
# Level 7B：Decoder-only（GPT 类）
# ============================================================================


class DecoderOnlyTransformer(nn.Module):
    """根据左侧历史预测下一个 token。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = TokenEmbedding(vocab_size, d_model, pad_id, dropout, max_len)
        self.blocks = nn.ModuleList(
            DecoderOnlyBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(initialize_weights)
        # 输入、输出共用同一词表时可共享权重，减少参数量。
        self.lm_head.weight = self.embedding.token.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        mask = make_causal_mask(token_ids, self.pad_id)
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(x)


# ============================================================================
# Level 7C：Encoder–Decoder（原始 Transformer / T5 类）
# ============================================================================


class EncoderDecoderTransformer(nn.Module):
    """source 编码成 memory，Decoder 根据 memory 和目标历史生成。"""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.source_embedding = TokenEmbedding(
            source_vocab_size, d_model, pad_id, dropout, max_len
        )
        self.target_embedding = TokenEmbedding(
            target_vocab_size, d_model, pad_id, dropout, max_len
        )
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.decoder_blocks = nn.ModuleList(
            Seq2SeqDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )
        self.output = nn.Linear(d_model, target_vocab_size)
        self.apply(initialize_weights)

    def encode(self, source_ids: Tensor, source_mask: Tensor) -> Tensor:
        memory = self.source_embedding(source_ids)
        for block in self.encoder_blocks:
            memory = block(memory, source_mask)
        return memory

    def decode(
        self,
        target_ids: Tensor,
        memory: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        x = self.target_embedding(target_ids)
        for block in self.decoder_blocks:
            x = block(x, memory, source_mask, target_mask)
        return x

    def forward(self, source_ids: Tensor, target_ids: Tensor) -> Tensor:
        source_mask = make_padding_mask(source_ids, self.pad_id)
        target_mask = make_causal_mask(target_ids, self.pad_id)
        memory = self.encode(source_ids, source_mask)
        return self.output(self.decode(target_ids, memory, source_mask, target_mask))


# 兼容之前的类名。
Transformer = EncoderDecoderTransformer


# ============================================================================
# Level 8：逐层 smoke test
# ============================================================================


def architecture_demo() -> None:
    torch.manual_seed(42)
    source = torch.tensor([[1, 8, 9, 2, 0], [1, 6, 2, 0, 0]])

    encoder = EncoderOnlyTransformer(
        vocab_size=100, num_classes=3, d_model=32, num_heads=4, d_ff=64
    )
    class_logits = encoder(source)
    assert class_logits.shape == (2, 3)
    print("[Level 7A] Encoder-only logits:", tuple(class_logits.shape))

    decoder_input = torch.tensor([[1, 20, 21, 2], [1, 30, 2, 0]])
    decoder_label = torch.tensor([[20, 21, 2, 0], [30, 2, 0, 0]])
    decoder = DecoderOnlyTransformer(
        vocab_size=120, d_model=32, num_heads=4, d_ff=64
    )
    lm_logits = decoder(decoder_input)
    lm_loss = nn.CrossEntropyLoss(ignore_index=0)(
        lm_logits.reshape(-1, lm_logits.size(-1)), decoder_label.reshape(-1)
    )
    lm_loss.backward()
    assert lm_logits.shape == (2, 4, 120)
    print("[Level 7B] Decoder-only logits:", tuple(lm_logits.shape))

    seq2seq = EncoderDecoderTransformer(
        source_vocab_size=100,
        target_vocab_size=120,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
    )
    seq2seq_logits = seq2seq(source, decoder_input)
    seq2seq_loss = nn.CrossEntropyLoss(ignore_index=0)(
        seq2seq_logits.reshape(-1, seq2seq_logits.size(-1)),
        decoder_label.reshape(-1),
    )
    seq2seq_loss.backward()
    assert seq2seq_logits.shape == (2, 4, 120)
    print("[Level 7C] Encoder-Decoder logits:", tuple(seq2seq_logits.shape))
    print(f"losses: decoder={lm_loss.item():.4f}, seq2seq={seq2seq_loss.item():.4f}")


def smoke_test() -> None:
    attention_formula_demo()
    architecture_demo()


if __name__ == "__main__":
    smoke_test()
