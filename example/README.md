# End to end integration test for the pipeline

`run_example.py` is a self-contained, **offline** sanity check (no downloads, ~1
minute). It synthesises organ masks whose *shape* depends on a hidden sex
(anisotropy) and age (size), then runs the **real** pipeline used for the papers
— marching cubes → surface point cloud → correspondence-free descriptors →
cross-validated classifier/regressor — and asserts that sex and age are recovered.

```bash
source .venv/bin/activate         # created by ../setup.sh
python example/run_example.py
```

Expected output (approximate):
```
Synthetic working example: 160 subjects, 2 organs, 30 shape features
  sex : AUC 1.000  balanced-acc 1.000
  age : MAE ~3 yr  R2 ~0.94
OK — pipeline works. Outputs in _workspace/example_output/
```
> **Note:** The random seed is fixed (`seed=0`), but exact metrics may vary
> slightly across CPU architectures (e.g. ARM vs x86) due to floating-point
> rounding differences. As long as the assertions pass (AUC > 0.7, R² > 0.3),
> the pipeline is working correctly.
It writes `_workspace/example_output/example_results.json` and `example.png`.

To run the pipeline on **real data** instead, see the top-level `README.md`
(`make download-totalseg`, then `python -m shapedem.cli smoke` for a tiny
range-requested subset, or the full targets).
