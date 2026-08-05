# Data directory

Datasets and generated SPD caches are not committed to this repository. Obtain
each dataset from its official provider and place it under this directory, or
pass an arbitrary local path with `--input-dir`.

Suggested layout:

```text
data/
  ADNI/
  ABIDE/
  HCP-WM/
  OASIS/
  PPMI/
  HAR/
    F3D.mat
    HDM14.mat
    UTK.mat
  florence_metadata/
    Florence_dataset_WorldCoordinates.txt
```

Florence leave-one-actor-out evaluation requires the original world-coordinate
metadata so actor identities can be recovered and verified. No dataset license
is granted by this repository.
