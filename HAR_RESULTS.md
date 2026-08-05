# Experiment status

| Dataset | Protocol used | Accuracy | Weighted F1 | Limitation |
|---|---|---:|---:|---|
| Florence | leave-one-actor-out | 83.06 ± 8.55% | 81.63 ± 9.70% | strict model |
| UTKinect | stratified 10-fold | 91.50 ± 5.80% | 90.43 ± 6.71% | available MAT has no actor IDs |
| HDM05 | 10 half-split runs | 76.06 ± 2.56% | 75.73 ± 2.62% | available MAT has 29, not 14, labels |
| ADNI | fold 1 of stratified 10-fold | 52.00% | 53.59% | available labels form a binary, not five-class, task |

Each diagnostic was trained for 100 epochs. Only Florence currently matches
the paper's evaluation protocol; the other values should not be presented as
paper reproductions.
