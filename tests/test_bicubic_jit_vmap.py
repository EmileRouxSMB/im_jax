import jax
import jax.numpy as jnp

from im_jax import flat_nd_cubic_interpolate


def test_jit_and_vmap_consistency():
    key = jax.random.PRNGKey(1)
    image = jax.random.normal(key, (12, 9), dtype=jnp.float32)
    key, sub = jax.random.split(key)
    locations = jax.random.uniform(sub, (2, 16), minval=0.0, maxval=8.0, dtype=jnp.float32)

    f = lambda loc: flat_nd_cubic_interpolate(
        image, loc, mode="reflect", kernel="keys", layout="HW"
    )
    out_eager = f(locations)
    out_jit = jax.jit(f)(locations)
    assert jnp.allclose(out_eager, out_jit, atol=1e-6)

    locations_batched = jnp.stack([locations, locations + 0.1], axis=0)
    out_vmap = jax.vmap(f)(locations_batched)
    out_expected = jnp.stack([f(locations), f(locations + 0.1)], axis=0)
    assert jnp.allclose(out_vmap, out_expected, atol=1e-6)


def test_jit_static_argnames():
    key = jax.random.PRNGKey(3)
    image = jax.random.normal(key, (10, 11), dtype=jnp.float32)
    key, sub = jax.random.split(key)
    locations = jax.random.uniform(sub, (2, 8), minval=0.0, maxval=9.0, dtype=jnp.float32)

    f = jax.jit(
        flat_nd_cubic_interpolate, static_argnames=("mode", "kernel", "layout")
    )
    out = f(image, locations, mode="reflect", kernel="keys", layout="HW")
    out_ref = flat_nd_cubic_interpolate(image, locations, layout="HW")
    assert jnp.allclose(out, out_ref, atol=1e-6)
