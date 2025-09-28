import jax
import jax.numpy as jnp
from typing import Optional
from flax.linen.module import Module

from flax.linen.dtypes import promote_dtype


def gelu2_attention(  # 签名尽量与 flax 的 dot_product_attention 对齐
    query: jnp.ndarray,            # [..., q, heads, d]
    key: jnp.ndarray,              # [..., k, heads, d]
    value: jnp.ndarray,            # [..., k, heads, dv]
    bias: Optional[jnp.ndarray] = None,    # broadcast 到 [..., heads, q, k]
    mask: Optional[jnp.ndarray] = None,    # 同上，bool
    broadcast_dropout: bool = False,
    dropout_rng: Optional[jax.Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype=jnp.float32,
    precision=None,
    module: Optional[Module] = None,                   # 一定得是有选项的，允许透传，便于 sow 中间量
    force_fp32_for_softmax: bool = False,  # 忽略（我们不用 softmax）
    einsum_dot_general=jax.lax.dot_general,  # 也透传
    **kwargs
):
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim == value.ndim, 'q, k, v must have same rank.'
    assert (
            query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), 'q, k, v batch dims must match.'
    assert (
            query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), 'q, k, v num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # 1) logits
    logits = jnp.einsum(
        '...qhd,...khd->...hqk',
        query,
        key,
        precision=precision,
        _dot_general=einsum_dot_general,
    )
    logits = jnp.where(logits > 0, logits, 0.0)

    if bias is not None:
        logits = logits + bias
    if mask is not None:
        logits = jnp.where(mask.astype(bool), logits, 0.0)

    # 2) GELU^2 权重并归一化
    weights = jax.nn.gelu(logits) ** 2
    if mask is not None:
        weights = jnp.where(mask.astype(bool), weights, 0.0)
    denom = jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.where(denom > 0, weights / (denom + 1e-9), jnp.zeros_like(weights))

    # （可选）把注意力权重 sow 出去，保持与 flax 行为一致
    if module is not None:
        module.sow('intermediates', 'attention_weights_gelu2', weights)

    # 3) 加权 V
    if (not deterministic) and (dropout_rate > 0.0):
        if dropout_rng is None:
            raise ValueError("dropout_rng required when dropout_rate>0 & not deterministic")
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # 在批维广播：与 flax 一致
            bshape = list(weights.shape)
            for ax in range(len(bshape) - 2):  # 广播除去最后两维(q,k)
                bshape[ax] = 1
            keep = jax.random.bernoulli(dropout_rng, keep_prob, bshape)
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
        weights = (weights * keep.astype(weights.dtype)) / keep_prob

    out = jnp.einsum('...hqk,...khd->...qhd', weights.astype(dtype), value,
                     precision=precision, _dot_general=einsum_dot_general)

    if jax.process_index() == 0 and not getattr(gelu2_attention, "_printed", False):
        print(">>> GELU2 Attention is running <<<")
        gelu2_attention._printed = True

    return out.astype(dtype)

    def gelu2(x):
        """简单的 gelu^2 激活函数示例"""
        return jax.nn.gelu(x) ** 2

    def gelu2_softmax(logits, axis=-1):
        """先用 gelu^2，再做 softmax"""
        return jax.nn.softmax(gelu2(logits), axis=axis)
