from jax import jit as jjit
import jax.numpy as jnp
from jax.scipy.special import gammaln


@jjit
def log_mse_loss(data, pred, eps):
    loss = jnp.mean(
        (jnp.log10(pred + eps) - jnp.log10(data + eps))
        * (jnp.log10(pred + eps) - jnp.log10(data + eps))
    )
    return loss


@jjit
def mse_loss(data, pred):
    loss = jnp.mean((pred - data) * (pred - data))
    return loss


@jjit
def ln_poisson_loss(data, pred, eps):
    data = data + eps
    pred = pred + eps
    loss = jnp.sum((data * jnp.log(pred) - pred) - gammaln(data + 1.0))
    return loss
