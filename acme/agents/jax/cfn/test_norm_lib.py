import numpy as np
import jax

from jax import numpy as jnp
from acme.agents.jax.cfn import norm_lib

# Need to do whole thing in jax because numpy aggressively promotes. Annoying. Later.

# seed = 12345
# key = jax.random.key(seed)


def do_iid(num, batch_size, n_features, divide_by=1.):
    seed = 12345
    key = jax.random.key(seed)
    mean_of_square = jnp.zeros((n_features,), dtype=jnp.float32)
    mean = jnp.zeros((n_features,), dtype=jnp.float32)

    X = []
    welford_mean = jnp.zeros((n_features,), dtype=jnp.float32)
    welford_m2 = jnp.zeros((n_features,), dtype=jnp.float32)
    welford_n = 0

    for i in range(num):
        key, subkey = jax.random.split(key)
        sample = jax.random.normal(subkey, shape=(batch_size, n_features))
        sample = sample / divide_by

        welford_mean, welford_m2, welford_n = norm_lib.welford(
            welford_mean, welford_m2, welford_n, sample)
        
        # sample = jnp.float32(jnp.random.randn())
        delta_mean_of_square = (sample**2 - mean_of_square) / (i+1)
        delta_mean = (sample - mean) / (i+1)
        mean_of_square += delta_mean_of_square.mean(axis=0)
        mean += delta_mean.mean(axis=0)
        
        X.append(sample)

        if i % 1000 == 0:
            data_batch = np.asarray(X).reshape(-1, n_features)
            gt_mean = np.mean(data_batch, axis=0)
            gt_var = np.var(data_batch, axis=0)

            print(f"step {i} shitty variance is: {mean_of_square - mean**2} ",
                  "welford variance is: ", welford_m2 / welford_n,
                  "gt variance is: ", gt_var,
                  "shitty mean is: ", mean,
                  "welford mean is: ", welford_mean,
                  "gt mean is: ", gt_mean)


def do_sorted(num, batch_size, divide_by=1.):
    seed = 12345
    key = jax.random.key(seed)

    mean_of_square = jnp.float32(0)
    mean = jnp.float32(0)
    samples = jax.random.normal(key, (num, batch_size))
    samples = jnp.sort(samples)
    samples = samples / divide_by
    print(samples.dtype)
    # exit()
    # samples = jnp.float32(jnp.random.randn(num))
    # samples.sort()
    # samples = jnp.flip(samples)
    print(samples.dtype)
    
    X = []
    welford_mean = jnp.float32(0)
    welford_m2 = jnp.float32(0)
    welford_n = 0
    
    for i, sample in enumerate(samples):
        welford_mean, welford_m2, welford_n = norm_lib.welford(
            welford_mean, welford_m2, welford_n, sample)
        
        # sample = jnp.float32(jnp.random.randn())
        delta_mean_of_square = (sample**2 - mean_of_square) / (i+1)
        delta_mean = (sample - mean) / (i+1)
        mean_of_square += delta_mean_of_square.mean()
        mean += delta_mean.mean()
        
        X.append(sample)

        if (i % (num // 1000)) == 0:
            
            gt_mean = np.mean(X)
            gt_var = np.var(X)

            print(f"step {i} shitty variance is: {mean_of_square - mean**2} ",
                  "welford variance is: ", welford_m2 / welford_n,
                  "gt variance is: ", gt_var,
                  "shitty mean is: ", mean,
                  "welford mean is: ", welford_mean,
                  "gt mean is: ", gt_mean)


# do_sorted(1001, batch_size=10, divide_by=100.)
do_iid(10000000, batch_size=1000, n_features=1, divide_by=100.)

