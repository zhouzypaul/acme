import jax
import time
import jax.numpy as jnp


@jax.jit
def update_mean_var_count_from_moments(
  mean,
  var,
  count,
  batch_mean,
  batch_var,
  batch_count
):
  """Updates the mean, var and count using the previous mean, var, count and batch values."""
  
  delta = batch_mean - mean
  tot_count = count + batch_count

  new_mean = mean + delta * batch_count / tot_count
  m_a = var * count
  m_b = batch_var * batch_count
  M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
  new_var = M2 / tot_count
  new_count = tot_count

  return new_mean, new_var, new_count


def test_basic_from_arrays():
  batch = jnp.arange(12).reshape(3, 4)
  
  mean, var, count = 0., 1., 0

  for i in range(3):
    batch_mean = jnp.mean(batch[i])
    batch_var = jnp.var(batch[i])
    batch_count = 4

    mean, var, count = update_mean_var_count_from_moments(
      mean, var, count, batch_mean, batch_var, batch_count
    )

  assert mean == batch.mean(), (mean, batch.mean())
  assert var == batch.var(), (var, batch.var())
  assert count == 12, (count, 12)


def test_batch_size(batch_size: int):
  n_batches = 12 // batch_size
  batch = jnp.arange(12).reshape(n_batches, batch_size)
  
  mean, var, count = 0., 1., 0

  for i in range(n_batches):
    batch_mean = jnp.mean(batch[i])
    batch_var = jnp.var(batch[i])
    batch_count = len(batch[i])

    mean, var, count = update_mean_var_count_from_moments(
      mean, var, count, batch_mean, batch_var, batch_count
    )

  assert mean == batch.mean(), (mean, batch.mean())
  assert var == batch.var(), (var, batch.var())
  assert count == 12, (count, 12)


def test_variance_zero1(batch_size: int):
  n_batches = 12 // batch_size
  batch = jnp.zeros((12,)).reshape(n_batches, batch_size)
  
  mean, var, count = 0., 1., 0

  for i in range(n_batches):
    batch_mean = jnp.mean(batch[i])
    batch_var = jnp.var(batch[i])
    batch_count = len(batch[i])

    mean, var, count = update_mean_var_count_from_moments(
      mean, var, count, batch_mean, batch_var, batch_count
    )

  assert mean == batch.mean(), (mean, batch.mean())
  assert var == batch.var(), (var, batch.var())
  assert count == 12, (count, 12)


def test_variance_zero2(batch_size: int):
  n_batches = 12 // batch_size
  batch = jnp.ones((12,)).reshape(n_batches, batch_size)
  
  mean, var, count = 0., 1., 0

  for i in range(n_batches):
    batch_mean = jnp.mean(batch[i])
    batch_var = jnp.var(batch[i])
    batch_count = len(batch[i])

    mean, var, count = update_mean_var_count_from_moments(
      mean, var, count, batch_mean, batch_var, batch_count
    )

  assert mean == batch.mean(), (mean, batch.mean())
  assert var == batch.var(), (var, batch.var())
  assert count == 12, (count, 12)


def test_batch_size_vector_version(batch_size: int, vector_dim: int):
  n_elements = 12 * vector_dim
  n_batches = 12 // batch_size
  batch = jnp.arange(n_elements).reshape(n_batches, batch_size, vector_dim)
  
  mean, var, count = 0., 1., 0

  for i in range(n_batches):
    batch_mean = jnp.mean(batch[i], axis=0)
    batch_var = jnp.var(batch[i], axis=0)
    assert batch_mean.shape == batch_var.shape == (vector_dim,)
    batch_count = len(batch[i])

    mean, var, count = update_mean_var_count_from_moments(
      mean, var, count, batch_mean, batch_var, batch_count
    )

  assert jnp.allclose(mean, batch.mean(axis=(0, 1))), (mean, batch.mean(axis=(0, 1)))
  assert jnp.allclose(var, batch.var(axis=(0, 1))), (var, batch.var(axis=(0, 1)))
  assert mean.shape == (vector_dim,), mean.shape
  assert var.shape == (vector_dim,), var.shape
  assert count == 12, (count, 12)


if __name__ == '__main__':
  t0 = time.time()
  test_basic_from_arrays()
  for j in [1, 2, 3, 4, 6, 12]:
    test_batch_size(j)
    test_variance_zero1(j)
    test_variance_zero2(j)
    test_batch_size_vector_version(j, 3)
  print(f'tests passed! Took {time.time() - t0}s.')
