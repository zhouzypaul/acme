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


@jax.jit
def welford(mean, second, n_batches, batch):
  """Update mean and second moment using Welford's algorithm.

  Args:
    mean: The current mean. Shape: (batch_size, n_features)
    second: The current second moment. Shape: (batch_size, n_features)
    n_batches: The number of batches seen so far. Int.
    batch: The new batch of data. Shape: (batch_size, n_features)

  Returns:
    The updated mean, second moment and count.

  """
  new_n_batches = n_batches + 1
  delta = (batch - mean)  # (batch_size, n_features)
  new_mean = mean + (delta.mean(axis=0) / new_n_batches)
  delta2 = (batch - new_mean)  # (batch_size, n_features)
  new_second = second + (delta * delta2).mean(axis=0)
  return new_mean, new_second, new_n_batches
