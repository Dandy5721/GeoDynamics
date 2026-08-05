# Florence 3D Actions results

All results use official leave-one-actor-out evaluation over ten actors. The
outer test fold is never used for checkpoint selection or BN calibration.

| Configuration | Accuracy | Weighted F1 |
|---|---:|---:|
| Strict model, epoch 100 | **83.06 ± 8.55** | **81.63 ± 9.70** |
| Mean readout + train-only BN, epoch 15 | **86.66 ± 8.74** | **84.82 ± 10.40** |
| Mean readout + train-only BN, epoch 100 | 84.09 ± 6.74 | 81.91 ± 7.75 |
| Upper-triangle vectorization ablation | 86.08 ± 8.39 | 84.68 ± 9.66 |

The strict result is the primary reconstruction. Mean readout and BN are
labelled implementation variants and do not alter the geometric core. The
paper reports Florence only in a radar plot, so an exact numeric target cannot
be read from the PDF.

A mean-SPD linear SVM reaches 88.83 ± 6.27% and is retained only as a data and
split sanity check, not as a GeoDynamics result.
