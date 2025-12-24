# Interpolation Benchmarks (JAX)

This folder contains a single notebook that benchmarks interpolation accuracy and speed for sub-pixel DIC in JAX.

## Notebook

- `interpolation_benchmarks.ipynb`: compares dm_pix bilinear, `jax.scipy.ndimage.map_coordinates` (order=1 and order=3), and `im_jax.flat_nd_cubic_interpolate`.

## How to run

```bash
jupyter notebook docs/interpolation_benchmarks.ipynb
```

## Dependencies

- JAX + jaxlib
- numpy
- pandas
- matplotlib
- dm_pix (optional)
- im_jax (this repo)

## Success criteria

- `im_jax` should be faster than `map_coordinates(order=3)` for large workloads.
- `im_jax` should be closer in accuracy to `map_coordinates(order=3)` than bilinear interpolation.
