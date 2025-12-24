import jax
import jax.numpy as jnp
import pytest

from im_jax import flat_nd_cubic_interpolate


@pytest.mark.slow
def test_performance_smoke():
    key = jax.random.PRNGKey(0)
    image = jax.random.normal(key, (256, 256), dtype=jnp.float32)
    key, sub = jax.random.split(key)
    locations = jax.random.uniform(sub, (2, 200_000), minval=0.0, maxval=255.0)

    f = jax.jit(
        flat_nd_cubic_interpolate, static_argnames=("mode", "kernel", "layout")
    )
    out = f(image, locations, mode="reflect", kernel="keys", layout="HW")
    assert out.shape == (locations.shape[1],)
