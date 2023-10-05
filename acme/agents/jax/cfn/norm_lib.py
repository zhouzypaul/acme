import jax
import jax.numpy as jnp


@jax.jit
def update_mean(original_mean, new_data_batch, n):
  return original_mean + jnp.sum(new_data_batch - original_mean, axis=0) / n


@jax.jit
def update_statistics_welford(mean, variance, n, batch):
  M2 = variance * (n - 1)
  
  for x in batch:
    n += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

  variance = M2 / (n - 1)
  return mean, variance, n
