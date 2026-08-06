"""Dataset loaders used by the GeoDynamics experiments.

Each loader returns a dictionary with ``input_ids`` shaped ``(T, N, N)`` and an integer ``labels`` value.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .utils import sliding_window_corrcoef


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _dynamic_connectivity(sequence: np.ndarray | torch.Tensor, window_size: int) -> torch.Tensor:
    sequence = np.asarray(sequence, dtype=np.float32)
    matrices = sliding_window_corrcoef(sequence.T, window_size)
    return torch.nan_to_num(torch.from_numpy(matrices).float())


class SkeletonDataset(Dataset):
    """Root-relative skeleton sequences stored in the released MAT format."""

    def __init__(self, mat_file: str | Path, window_size: int):
        self.mat_file = str(mat_file)
        self.window_size = window_size
        payload = sio.loadmat(self.mat_file)
        if "data" not in payload or "action_label" not in payload:
            raise ValueError("skeleton MAT must contain 'data' and 'action_label'")
        cells = payload["data"]
        if cells.ndim != 2:
            raise ValueError(f"expected a MATLAB cell column, got {cells.shape}")
        self.data = [np.asarray(cells[index, 0]) for index in range(cells.shape[0])]
        self.labels = payload["action_label"].reshape(-1).astype(np.int64) - 1
        if len(self.data) != len(self.labels):
            raise ValueError("skeleton samples and labels have different lengths")
        maximum = max(sample.shape[2] for sample in self.data)
        self.data = [
            np.pad(sample, ((0, 0), (0, 0), (0, maximum - sample.shape[2])))
            for sample in self.data
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        sample = self.data[index]
        coordinates = np.concatenate((sample[0], sample[1], sample[2]), axis=0)
        matrices = []
        for start in range(coordinates.shape[1] - self.window_size + 1):
            matrices.append(
                np.cov(
                    coordinates[:, start : start + self.window_size], rowvar=True
                ).astype(np.float32)
            )
        return {
            "input_ids": torch.from_numpy(np.stack(matrices)),
            "labels": int(self.labels[index]),
        }


class HCPWorkingMemoryDataset(Dataset):
    """HCP working-memory clips paired with task-label time courses."""

    def __init__(self, data_dir: str | Path, label_dir: str | Path, window_size: int = 15):
        data_paths = sorted(
            (path for path in Path(data_dir).iterdir() if path.suffix in {".txt", ".csv"}),
            key=_natural_key,
        )
        label_paths = sorted(
            (path for path in Path(label_dir).iterdir() if path.suffix in {".txt", ".csv"}),
            key=_natural_key,
        )
        if len(label_paths) == 1 and len(data_paths) > 1:
            label_paths = label_paths * len(data_paths)
        if not data_paths or len(data_paths) != len(label_paths):
            raise ValueError("HCP data files and label files must pair one-to-one")

        self.data: list[np.ndarray] = []
        self.labels: list[int] = []
        self.window_size = window_size
        for data_path, label_path in zip(data_paths, label_paths):
            sequence = np.loadtxt(data_path, delimiter="\t", dtype=np.float32).squeeze()
            labels = np.loadtxt(label_path, delimiter=",", dtype=np.float32).reshape(-1)
            for label in np.unique(labels):
                if label <= 0:
                    continue
                self.data.append(sequence[labels == label])
                self.labels.append(int(label - 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": _dynamic_connectivity(self.data[index], self.window_size),
            "labels": self.labels[index],
        }


class ADNIDataset(Dataset):
    """ADNI time courses with either the paper's five classes or a binary grouping."""

    FIVE_CLASS = {"CN": 0, "SMC": 1, "EMCI": 2, "LMCI": 3, "AD": 4}
    BINARY = {"CN": 0, "SMC": 0, "EMCI": 0, "LMCI": 1, "AD": 1}

    def __init__(
        self,
        data_dir: str | Path,
        label_file: str | Path,
        window_size: int = 15,
        label_scheme: str = "five-class",
    ):
        if label_scheme not in {"five-class", "binary"}:
            raise ValueError("ADNI label scheme must be 'five-class' or 'binary'")
        mapping = self.FIVE_CLASS if label_scheme == "five-class" else self.BINARY
        table = pd.read_csv(label_file, dtype={"subject_id": str})
        diagnoses = dict(zip(table["subject_id"], table["DX"]))
        self.data_files: list[Path] = []
        self.labels: list[int] = []
        self.window_size = window_size
        for path in sorted(Path(data_dir).glob("sub_*"), key=_natural_key):
            subject = path.name.split("_")[1]
            diagnosis = diagnoses.get(subject)
            if diagnosis in mapping:
                self.data_files.append(path)
                self.labels.append(mapping[diagnosis])
        if not self.data_files:
            raise ValueError("no ADNI subjects matched the label table")

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, index: int) -> dict:
        sequence = np.loadtxt(self.data_files[index], dtype=np.float32)[:140]
        return {
            "input_ids": _dynamic_connectivity(sequence, self.window_size),
            "labels": self.labels[index],
        }


class PPMIABIDEDataset(Dataset):
    """AAL116 time courses organized below diagnosis-named directories."""

    LABELS = {"control": 0, "patient": 1, "prodromal": 2, "swedd": 3}

    def __init__(self, root_dir: str | Path, window_size: int = 15, length: int = 137):
        self.data: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.window_size = window_size
        for path in sorted(Path(root_dir).rglob("*AAL116_features_timeseries*.mat")):
            label = next(
                (value for name, value in self.LABELS.items() if name in str(path.parent).lower()),
                None,
            )
            if label is None:
                continue
            sequence = torch.as_tensor(loadmat(path)["data"], dtype=torch.float32)
            sequence = sequence[:length]
            if len(sequence) < length:
                sequence = torch.cat(
                    (sequence, torch.zeros(length - len(sequence), sequence.shape[1]))
                )
            self.data.append(sequence)
            self.labels.append(label)
        if not self.data:
            raise ValueError("no PPMI/ABIDE AAL116 files were found")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": _dynamic_connectivity(self.data[index], self.window_size),
            "labels": self.labels[index],
        }


class OASISDataset(Dataset):
    """OASIS time courses labelled from CN/AD filenames."""

    def __init__(self, root_dir: str | Path, window_size: int = 15, length: int = 328):
        self.data: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.window_size = window_size
        for path in sorted(Path(root_dir).glob("*.txt"), key=_natural_key):
            if "CN" in path.name:
                label = 0
            elif "AD" in path.name:
                label = 1
            else:
                continue
            sequence = torch.as_tensor(np.loadtxt(path), dtype=torch.float32)
            if len(sequence) < length:
                sequence = torch.cat(
                    (sequence, torch.zeros(length - len(sequence), sequence.shape[1]))
                )
            elif len(sequence) > length:
                start = (len(sequence) - length) // 2
                sequence = sequence[start : start + length]
            self.data.append(sequence)
            self.labels.append(label)
        if not self.data:
            raise ValueError("no OASIS CN/AD text files were found")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": _dynamic_connectivity(self.data[index], self.window_size),
            "labels": self.labels[index],
        }
