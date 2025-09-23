# custom_attention.py
from typing import Optional
import jax
import jax.numpy as jnp
from jax.nn import gelu

def gelu2_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    dropout_rng: Optional[jax.Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype=jnp.float32,
    precision=None,
    **kwargs,   # 兼容不同 Flax 版本的额外参数
):
    depth = query.shape[-1]
    scale = 1.0 / jnp.sqrt(depth)
    logits = jnp.einsum('...qd,...kd->...qk', query, key, precision=precision) * scale

    if bias is not None:
        logits = logits + bias
    if mask is not None:
        logits = jnp.where(mask, logits, -jnp.inf)

    # 用 GELU(x)^2 作为“权重函数”，再做概率归一化
    weights = gelu(logits) ** 2
    if mask is not None:
        weights = jnp.where(mask, weights, 0.0)

    denom = jnp.sum(weights, axis=-1, keepdims=True)
    weights = weights / (denom + 1e-9)  # 防除零
    weights = weights.astype(dtype)

    if (not deterministic) and (dropout_rate > 0.0):
        keep_prob = 1.0 - dropout_rate
        keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
        weights = (weights * keep.astype(dtype)) / keep_prob

    out = jnp.einsum('...qk,...kd->...qd', weights, value, precision=precision)
    return out
