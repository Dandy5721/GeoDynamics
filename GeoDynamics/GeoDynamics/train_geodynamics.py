"""Train GeoDynamics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import (
    ADNIDataset,
    HCPWorkingMemoryDataset,
    OASISDataset,
    PPMIABIDEDataset,
    SkeletonDataset,
)
from .geodynamics import GeoDynamics


DATASETS = ("ADNI", "HCP-WM", "OASIS", "PPMI", "ABIDE", "F3D", "HDM05", "UTK")


def build_dataset(args):
    """Load every dataset used by the paper through one training entry point."""
    if not args.input_dir.exists():
        raise FileNotFoundError(f"input path does not exist: {args.input_dir}")
    if args.dataset in {"ADNI", "HCP-WM"}:
        if args.label_path is None or not args.label_path.exists():
            raise FileNotFoundError(
                "--label-path is required and must exist for this dataset"
            )
    if args.dataset == "ADNI":
        return ADNIDataset(
            args.input_dir,
            args.label_path,
            args.window_size,
            args.adni_label_scheme,
        )
    if args.dataset == "HCP-WM":
        return HCPWorkingMemoryDataset(
            args.input_dir, args.label_path, args.window_size
        )
    if args.dataset == "OASIS":
        return OASISDataset(args.input_dir, args.window_size)
    if args.dataset in {"PPMI", "ABIDE"}:
        return PPMIABIDEDataset(args.input_dir, args.window_size)
    return SkeletonDataset(args.input_dir, args.window_size)


def dataset_labels(dataset) -> np.ndarray:
    """Return validated zero-based labels for a loaded dataset."""
    labels = np.asarray(dataset.labels, dtype=np.int64).reshape(-1)
    if len(labels) != len(dataset):
        raise ValueError(
            f"dataset exposes {len(labels)} labels for {len(dataset)} samples"
        )
    if (labels < 0).any():
        raise ValueError("dataset contains unresolved negative labels")
    return labels


def _save_cache(payload: dict, path: Path) -> None:
    """Write a shared preprocessing cache without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _check_cache_policy(cached: dict, normalization: str, padding_policy: str) -> None:
    policy = cached.get("policy")
    if policy is None:
        return
    expected = {"normalization": normalization, "padding_policy": padding_policy}
    if policy != expected:
        raise ValueError(f"SPD cache policy is {policy}, requested {expected}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-path", type=Path)
    parser.add_argument(
        "--adni-label-scheme",
        choices=["five-class", "binary"],
        default="five-class",
        help="five-class matches the paper; binary reproduces the local diagnostic grouping",
    )
    parser.add_argument("--actor-path", type=Path, help="Optional actor IDs; Florence IDs are recovered from the bundled metadata")
    parser.add_argument(
        "--florence-metadata",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data/florence_metadata/Florence_dataset_WorldCoordinates.txt",
    )
    parser.add_argument(
        "--florence-source",
        choices=["paper", "world", "mat14", "original15"],
        default="paper",
        help=(
            "paper uses the released root-relative descriptors; world rebuilds "
            "them from official coordinates (mat14/original15 are compatibility aliases)"
        ),
    )
    parser.add_argument("--allow-stratified-har", action="store_true", help="Use only when actor IDs are unavailable; this is not the paper protocol")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Table 6 settings for GeoDynamics.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument(
        "--max-time-points",
        type=int,
        help=(
            "Uniformly sample this many covariance windows for long HAR "
            "sequences; unset keeps every window"
        ),
    )
    parser.add_argument("--spd-cache", type=Path, help="Optional persistent cache for preprocessed SPD sequences")
    parser.add_argument(
        "--spd-normalization",
        choices=["none", "unit-trace"],
        default="none",
        help="The released Florence loader uses none; unit-trace is an optional stability ablation",
    )
    parser.add_argument(
        "--padding-policy",
        choices=["zero", "repeat-last"],
        default="zero",
        help="zero matches the released Florence loader; repeat-last keeps every padded matrix informative",
    )
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--tau", type=int, default=3)
    parser.add_argument("--stein-iterations", type=int, default=3)
    parser.add_argument(
        "--readout", choices=["final", "mean", "frechet"], default="final"
    )
    parser.add_argument(
        "--feature-normalization",
        choices=["none", "batch"],
        default="none",
        help="Optional tangent-feature BatchNorm ablation; strict Eq. 12 uses none",
    )
    parser.add_argument(
        "--vectorization",
        choices=["full", "upper"],
        default="full",
        help="full is literal vec; upper keeps only independent symmetric entries",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--fold-index", type=int, help="Run one 1-based fold; useful for multi-GPU experiments")
    parser.add_argument("--validation-actor", type=int, help="Optional inner validation actor, excluded from outer-fold training")
    parser.add_argument(
        "--evaluation-epoch",
        type=int,
        help="Fixed checkpoint used for outer testing after all epochs finish; choose it before the outer runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted fold from latest.pt in the output directory",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--torch-threads", type=int, default=2,
        help="CPU threads per process; keep this small when folds run in parallel",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MAdam:
    """Adam for the paper model's manifold parameterization.

    Group elements are exponentials of skew matrices and SPD kernels are
    ``K^T K + eps I``. They therefore stay on their manifolds after every Adam
    update without a separate projection step.
    """

    def __init__(self, parameters, lr: float, weight_decay: float):
        self.optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


class SampledSkeletonDataset(Dataset):
    """Compute a fixed number of evenly spaced covariance windows on demand."""

    def __init__(self, dataset, window_size: int, time_points: int, epsilon: float = 1e-5):
        if time_points < 1:
            raise ValueError("--max-time-points must be positive")
        self.dataset = dataset
        self.labels = np.asarray(dataset.labels, dtype=np.int64)
        self.window_size = window_size
        self.time_points = time_points
        self.epsilon = epsilon

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = np.asarray(self.dataset.data[index])
        coordinates = np.concatenate((sample[0], sample[1], sample[2]), axis=0)
        available = coordinates.shape[1] - self.window_size + 1
        if available < 1:
            raise ValueError("window size exceeds the padded skeleton length")
        starts = np.linspace(
            0, available - 1, num=self.time_points, dtype=np.int64
        )
        matrices = []
        identity = np.eye(coordinates.shape[0], dtype=np.float32)
        for start in starts:
            window = coordinates[:, start : start + self.window_size]
            covariance = np.cov(window, rowvar=True).astype(np.float32)
            matrices.append(covariance + self.epsilon * identity)
        return {
            "input_ids": torch.from_numpy(np.stack(matrices)),
            "labels": int(self.labels[index]),
        }


class SampledSequenceDataset(Dataset):
    """Uniformly sample time points from an already constructed SPD sequence."""

    def __init__(self, dataset, time_points: int):
        if time_points < 1:
            raise ValueError("--max-time-points must be positive")
        self.dataset = dataset
        self.labels = np.asarray(dataset.labels, dtype=np.int64)
        self.time_points = time_points

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        sequence = item["input_ids"]
        positions = torch.linspace(
            0, sequence.shape[0] - 1, self.time_points
        ).round().long()
        return {"input_ids": sequence[positions], "labels": int(item["labels"])}


class CachedSPDDataset(Dataset):
    """Cache Florence SPD sequences with an explicit padding/scale policy."""

    def __init__(
        self, dataset, window_size: int, cache_path: Path | None = None,
        epsilon: float = 1e-5, normalization: str = "none",
        padding_policy: str = "zero",
    ):
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            _check_cache_policy(cached, normalization, padding_policy)
            self.data = cached["data"]
            self.labels = cached["labels"].numpy()
            return
        mat = loadmat(dataset.mat_file)
        frame_counts = [sample[0].shape[2] for sample in mat["data"]]
        self.labels = np.asarray(dataset.labels, dtype=np.int64)
        self.data = []
        for index in range(len(dataset)):
            sequence = dataset[index]["input_ids"].float()
            valid = max(1, frame_counts[index] - window_size + 1)
            if padding_policy == "repeat-last" and valid < sequence.shape[0]:
                sequence[valid:] = sequence[valid - 1]
            if normalization == "unit-trace":
                trace = torch.diagonal(sequence, dim1=-2, dim2=-1).sum(-1)
                sequence = sequence / trace.clamp_min(epsilon)[:, None, None]
            identity = torch.eye(sequence.shape[-1], dtype=sequence.dtype)
            self.data.append(sequence + epsilon * identity)
        self.data = torch.stack(self.data)
        if cache_path is not None:
            _save_cache(
                {
                    "data": self.data,
                    "labels": torch.as_tensor(self.labels),
                    "policy": {
                        "normalization": normalization,
                        "padding_policy": padding_policy,
                    },
                },
                cache_path,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"input_ids": self.data[index], "labels": int(self.labels[index])}


class FlorenceSPDDataset(Dataset):
    """Build the paper's 42-D descriptors from the official Florence poses."""

    def __init__(
        self, metadata_path: Path, window_size: int, cache_path: Path | None = None,
        epsilon: float = 1e-5, normalization: str = "none",
        padding_policy: str = "zero",
    ):
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            _check_cache_policy(cached, normalization, padding_policy)
            self.data = cached["data"]
            self.labels = cached["labels"].numpy()
            self.actors = cached["actors"].numpy()
            return
        raw = np.loadtxt(metadata_path)
        keys = raw[:, :3].astype(np.int64)
        records = []
        for video_id in np.unique(keys[:, 0]):
            rows = raw[keys[:, 0] == video_id]
            records.append(
                (int(rows[0, 2]) - 1, int(video_id), int(rows[0, 1]), rows[:, 3:])
            )
        records.sort(key=lambda record: (record[0], record[1]))
        max_windows = max(record[3].shape[0] - window_size + 1 for record in records)
        # Florence calls joint 3 the spine. It is the central joint used as the
        # root in the processed F3D.mat file. Subtracting it and then dropping
        # it leaves the 3 * (15 - 1) coordinates described in the paper.
        root_joint = 2
        descriptor_size = 3 * (15 - 1)
        identity = torch.eye(descriptor_size, dtype=torch.float32)
        data, labels, actors = [], [], []
        for label, _video, actor, coordinates in records:
            poses = coordinates.reshape(-1, 15, 3) / 1000.0
            poses = poses - poses[:, root_joint : root_joint + 1]
            poses = np.delete(poses, root_joint, axis=1)
            # This is the axis order used by the released F3D.mat file.
            relative_coordinates = poses[:, :, [0, 2, 1]].reshape(-1, descriptor_size)
            if padding_policy == "zero" and relative_coordinates.shape[0] < max_windows + window_size - 1:
                relative_coordinates = np.pad(
                    relative_coordinates,
                    ((0, max_windows + window_size - 1 - relative_coordinates.shape[0]), (0, 0)),
                )
            windows = []
            for start in range(relative_coordinates.shape[0] - window_size + 1):
                covariance = np.cov(
                    relative_coordinates[start : start + window_size], rowvar=False
                )
                matrix = torch.as_tensor(covariance, dtype=torch.float32)
                if normalization == "unit-trace":
                    matrix = matrix / torch.diagonal(matrix).sum().clamp_min(epsilon)
                windows.append(matrix + epsilon * identity)
            sequence = torch.stack(windows)
            if padding_policy == "repeat-last" and sequence.shape[0] < max_windows:
                tail = sequence[-1:].expand(max_windows - sequence.shape[0], -1, -1)
                sequence = torch.cat((sequence, tail), dim=0)
            data.append(sequence)
            labels.append(label)
            actors.append(actor)
        self.data = torch.stack(data)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.actors = np.asarray(actors, dtype=np.int64)
        if cache_path is not None:
            _save_cache(
                {
                    "data": self.data,
                    "labels": torch.as_tensor(self.labels),
                    "actors": torch.as_tensor(self.actors),
                    "policy": {
                        "normalization": normalization,
                        "padding_policy": padding_policy,
                    },
                },
                cache_path,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"input_ids": self.data[index], "labels": int(self.labels[index])}


def _metrics(labels, predictions, losses):
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def run_epoch(model, loader, device, optimizer=None, return_outputs=False):
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.CrossEntropyLoss()
    labels, predictions, losses = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["input_ids"].to(device=device, dtype=torch.float32)
            y = torch.as_tensor(batch["labels"], device=device, dtype=torch.long)
            if training:
                optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.extend([loss.item()] * y.numel())
            labels.extend(y.detach().cpu().tolist())
            predictions.extend(logits.argmax(-1).detach().cpu().tolist())
    metrics = _metrics(labels, predictions, losses)
    if return_outputs:
        return metrics, labels, predictions
    return metrics


def _load_actor_ids(path: Path, expected: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None, dtype=str)
    values = np.asarray(values).reshape(-1)
    if len(values) != expected:
        raise ValueError(f"actor file contains {len(values)} IDs for {expected} samples")
    return values


def _recover_florence_actor_ids(dataset, path: Path) -> np.ndarray:
    """Match the action-sorted MAT samples to the original Florence records."""
    if not path.exists():
        raise FileNotFoundError(f"Florence metadata not found: {path}")
    raw = np.loadtxt(path)
    keys = raw[:, :3].astype(np.int64)
    records = []
    for video_id in np.unique(keys[:, 0]):
        rows = keys[keys[:, 0] == video_id]
        records.append(
            (int(rows[0, 2]) - 1, int(video_id), int(rows[0, 1]), len(rows))
        )
    # F3D.mat is ordered by action, and keeps the original video order within
    # each action. Labels and frame counts give an exact one-to-one check.
    records.sort(key=lambda record: (record[0], record[1]))
    if len(records) != len(dataset):
        raise ValueError(
            f"metadata contains {len(records)} sequences, MAT contains {len(dataset)}"
        )
    mat = loadmat(dataset.mat_file)
    unpadded = [sample[0].shape[2] for sample in mat["data"]]
    for index, (label, _video, _actor, frames) in enumerate(records):
        mat_label = int(dataset.labels[index])
        mat_frames = int(unpadded[index])
        if (label, frames) != (mat_label, mat_frames):
            raise ValueError(
                "Florence metadata does not match F3D.mat at sample "
                f"{index}: metadata={(label, frames)}, MAT={(mat_label, mat_frames)}"
            )
    return np.asarray([record[2] for record in records], dtype=np.int64)


def _splits(args, labels, dataset):
    har = args.dataset in {"F3D", "HDM05", "UTK"}
    if args.dataset == "HDM05":
        splitter = StratifiedShuffleSplit(
            n_splits=10, train_size=0.5, random_state=args.seed
        )
        return (
            list(splitter.split(np.zeros(len(labels)), labels)),
            "ten-random-stratified-half-splits",
        )
    if har and args.actor_path is not None:
        actors = _load_actor_ids(args.actor_path, len(labels))
        return list(LeaveOneGroupOut().split(np.zeros(len(labels)), labels, actors)), "leave-one-actor-out"
    if args.dataset == "F3D" and hasattr(dataset, "actors"):
        return list(
            LeaveOneGroupOut().split(np.zeros(len(labels)), labels, dataset.actors)
        ), "leave-one-actor-out"
    if args.dataset == "F3D":
        actors = _recover_florence_actor_ids(dataset, args.florence_metadata)
        return list(LeaveOneGroupOut().split(np.zeros(len(labels)), labels, actors)), "leave-one-actor-out"
    if har and not args.allow_stratified_har:
        raise ValueError(
            "The paper uses leave-one-actor-out for HAR. Supply --actor-path, "
            "or explicitly pass --allow-stratified-har for a non-paper fallback."
        )
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    return list(splitter.split(np.zeros(len(labels)), labels)), "stratified-k-fold"


def main(argv=None):
    args = parse_args(argv)
    args.florence_source = {
        "mat14": "paper",
        "original15": "world",
    }.get(args.florence_source, args.florence_source)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    if args.evaluation_epoch is not None and not 1 <= args.evaluation_epoch <= args.epochs:
        raise ValueError("--evaluation-epoch must be between 1 and --epochs")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "F3D" and args.florence_source == "world":
        cache_path = args.spd_cache
        if cache_path is None:
            preprocessing_tag = f"{args.spd_normalization}_{args.padding_policy}"
            cache_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / f"F3D15_root_relative_ws{args.window_size}_{preprocessing_tag}_spd.pt"
            )
        args.spd_cache = cache_path
        dataset = FlorenceSPDDataset(
            args.florence_metadata, args.window_size, cache_path,
            normalization=args.spd_normalization,
            padding_policy=args.padding_policy,
        )
    else:
        dataset = build_dataset(args)
    if args.max_time_points is not None:
        if args.dataset in {"HDM05", "UTK"}:
            dataset = SampledSkeletonDataset(
                dataset, args.window_size, args.max_time_points
            )
        elif args.dataset != "F3D":
            dataset = SampledSequenceDataset(dataset, args.max_time_points)
        else:
            raise ValueError("--max-time-points is not supported for Florence")
    labels = dataset_labels(dataset)
    sample = dataset[0]["input_ids"]
    time_points, dimension, _ = sample.shape
    splits, protocol = _splits(args, labels, dataset)
    actors = None
    if args.actor_path is not None:
        actors = _load_actor_ids(args.actor_path, len(labels))
    elif args.dataset == "F3D" and hasattr(dataset, "actors"):
        actors = dataset.actors
    elif args.dataset == "F3D":
        actors = _recover_florence_actor_ids(dataset, args.florence_metadata)
    if args.dataset == "F3D" and args.florence_source == "paper":
        cache_path = args.spd_cache
        if cache_path is None:
            preprocessing_tag = f"{args.spd_normalization}_{args.padding_policy}"
            cache_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / f"F3D_ws{args.window_size}_{preprocessing_tag}_spd.pt"
            )
        args.spd_cache = cache_path
        dataset = CachedSPDDataset(
            dataset, args.window_size, cache_path,
            normalization=args.spd_normalization,
            padding_policy=args.padding_policy,
        )
        sample = dataset[0]["input_ids"]
        time_points, dimension, _ = sample.shape
    total_folds = len(splits)
    if args.fold_index is not None:
        if not 1 <= args.fold_index <= total_folds:
            raise ValueError(f"--fold-index must be between 1 and {total_folds}")
        selected_splits = [(args.fold_index, splits[args.fold_index - 1])]
    else:
        selected_splits = list(enumerate(splits, 1))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["split_protocol"] = protocol
    config["architecture"] = "geodynamics"
    config["implementation_version"] = "geodynamics-v7"
    if args.dataset == "HDM05" and len(np.unique(labels)) != 14:
        config["dataset_warning"] = (
            f"available MAT exposes {len(np.unique(labels))} labels; "
            "this is not the paper's verified 14-class target"
        )
    if args.dataset == "UTK" and args.actor_path is None:
        config["dataset_warning"] = (
            "actor metadata is absent; this run uses the explicitly requested "
            "stratified fallback and is not leave-one-actor-out"
        )
    if args.dataset == "ADNI" and len(np.unique(labels)) != 5:
        config["dataset_warning"] = (
            f"available labels define {len(np.unique(labels))} classes; "
            "this is not the paper's five-class ADNI target"
        )
    if args.dataset == "F3D" and args.florence_source == "world":
        config["har_root_joint"] = "spine (official joint 3)"
        config["har_descriptor_dimension"] = 42
    config["batchnorm_singleton_policy"] = (
        "drop-last" if args.feature_normalization == "batch" else "not-applicable"
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    rows = []
    history_rows = []
    for fold, (train_ids, test_ids) in selected_splits:
        seed_everything(args.seed + fold)
        validation_ids = None
        if args.validation_actor is not None:
            if actors is None:
                raise ValueError("--validation-actor currently requires F3D actor metadata")
            if args.validation_actor in np.unique(actors[test_ids]):
                raise ValueError("validation actor must differ from the outer test actor")
            validation_ids = train_ids[actors[train_ids] == args.validation_actor]
            train_ids = train_ids[actors[train_ids] != args.validation_actor]
            if len(validation_ids) == 0:
                raise ValueError(f"actor {args.validation_actor} is not available for validation")
        drop_singleton = (
            args.feature_normalization == "batch"
            and len(train_ids) % args.batch_size == 1
        )
        train_loader = DataLoader(
            Subset(dataset, train_ids), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + fold),
            num_workers=args.num_workers, drop_last=drop_singleton,
        )
        test_loader = DataLoader(
            Subset(dataset, test_ids), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        validation_loader = None
        if validation_ids is not None:
            validation_loader = DataLoader(
                Subset(dataset, validation_ids), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
            )
        model = GeoDynamics(
            dimension=dimension,
            time_points=time_points,
            num_classes=int(labels.max()) + 1,
            kernel_size=args.kernel_size,
            tau=args.tau,
            num_layers=args.layers,
            stein_iterations=args.stein_iterations,
            readout=args.readout,
            feature_normalization=args.feature_normalization,
            vectorization=args.vectorization,
        )
        model = model.to(args.device)
        optimizer = MAdam(model.parameters(), args.learning_rate, args.weight_decay)

        best_validation = -1.0
        start_epoch = 1
        latest_path = args.output_dir / "latest.pt"
        history_path = args.output_dir / "training_history.csv"
        if args.resume:
            if not latest_path.exists():
                raise FileNotFoundError(f"Cannot resume without {latest_path}")
            latest_checkpoint = torch.load(
                latest_path, map_location=args.device, weights_only=False
            )
            if latest_checkpoint["fold"] != fold:
                raise ValueError("latest.pt belongs to a different fold")
            model.load_state_dict(latest_checkpoint["model"])
            optimizer.load_state_dict(latest_checkpoint["optimizer"])
            checkpoint = latest_checkpoint
            start_epoch = int(latest_checkpoint["epoch"]) + 1
            if history_path.exists():
                with history_path.open(newline="") as handle:
                    history_rows = list(csv.DictReader(handle))
            print(f"fold={fold} resuming_from_epoch={start_epoch}", flush=True)

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, args.device, optimizer)
            validation_metrics = (
                run_epoch(model, validation_loader, args.device)
                if validation_loader is not None else None
            )
            checkpoint = {
                "fold": fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }
            torch.save(checkpoint, args.output_dir / "latest.pt")
            if validation_metrics is not None and validation_metrics["accuracy"] > best_validation:
                best_validation = validation_metrics["accuracy"]
                torch.save(checkpoint, args.output_dir / "best_validation.pt")
            if args.evaluation_epoch == epoch:
                torch.save(checkpoint, args.output_dir / "evaluation_epoch.pt")
            progress = {
                "status": "training", "fold": fold, "folds": total_folds,
                "epoch": epoch, "epochs": args.epochs, "train_metrics": train_metrics,
            }
            if validation_metrics is not None:
                progress["validation_metrics"] = validation_metrics
                progress["best_validation_accuracy"] = best_validation
            history_row = {
                "fold": fold,
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }
            if validation_metrics is not None:
                history_row.update(
                    {f"validation_{key}": value for key, value in validation_metrics.items()}
                )
            history_rows.append(history_row)
            history_fields = list(history_rows[0])
            with (args.output_dir / "training_history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=history_fields)
                writer.writeheader()
                writer.writerows(history_rows)
            (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(f"fold={fold} epoch={epoch} loss={train_metrics['loss']:.6f} acc={train_metrics['accuracy']:.4f}", flush=True)

        if validation_loader is not None:
            best_checkpoint = torch.load(
                args.output_dir / "best_validation.pt",
                map_location=args.device,
                weights_only=False,
            )
            model.load_state_dict(best_checkpoint["model"])
            checkpoint = best_checkpoint
        elif args.evaluation_epoch is not None:
            selected_checkpoint = torch.load(
                args.output_dir / "evaluation_epoch.pt",
                map_location=args.device,
                weights_only=False,
            )
            model.load_state_dict(selected_checkpoint["model"])
            checkpoint = selected_checkpoint
        torch.save(checkpoint, args.output_dir / f"fold_{fold}.pt")
        test_metrics, test_labels, test_predictions = run_epoch(
            model, test_loader, args.device, return_outputs=True
        )
        rows.append({"fold": fold, "checkpoint_epoch": checkpoint["epoch"], **test_metrics})
        with (args.output_dir / f"fold_{fold}_predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_index", "actor", "label", "prediction")
            )
            writer.writeheader()
            for sample_index, label, prediction in zip(
                test_ids, test_labels, test_predictions
            ):
                writer.writerow(
                    {
                        "sample_index": int(sample_index),
                        "actor": int(actors[sample_index]) if actors is not None else "",
                        "label": label,
                        "prediction": prediction,
                    }
                )
        print(f"fold={fold} test={test_metrics}", flush=True)

    with (args.output_dir / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for metric in ("accuracy", "precision", "recall", "f1")
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "progress.json").write_text(json.dumps({"status": "complete"}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
