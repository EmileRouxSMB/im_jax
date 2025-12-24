# im_jax

Utilitaires d'interpolation d'images avec JAX.

## Installation

```bash
pip install -e .
```

## Usage

```python
import jax.numpy as jnp
from im_jax import flat_nd_cubic_interpolate

image = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
locations = jnp.array([[0.5, 1.25], [1.0, 2.5]], dtype=jnp.float32)
values = flat_nd_cubic_interpolate(image, locations)
```

## Tests

```bash
pytest
```
