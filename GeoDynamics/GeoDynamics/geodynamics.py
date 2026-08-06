"""GeoDynamics model and SPD-manifold operations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "GeoDynamics",
    "SpatiallyPreservingAttention",
    "StateSpaceDiscretization",
    "frechet_mean_logdet",
    "logdet_distance_squared",
    "matrix_log_spd",
    "project_spd",
    "state_space_kernel",
]


def _sym(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def project_spd(matrix: Tensor, epsilon: float = 1e-5) -> Tensor:
    """Symmetrize a matrix and add the diagonal regularizer from Sec. 2.2."""
    matrix = torch.nan_to_num(_sym(matrix))
    identity = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    )
    return matrix + epsilon * identity


def matrix_log_spd(matrix: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Map an SPD matrix to the tangent space at identity (Eq. 11)."""
    matrix = project_spd(matrix, epsilon)
    offset = torch.linspace(
        1.0, 2.0, matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    )
    matrix = matrix + epsilon * offset.diag()
    values, vectors = torch.linalg.eigh(matrix)
    log_values = values.clamp_min(epsilon).log().unsqueeze(-2)
    return (vectors * log_values) @ vectors.transpose(-1, -2)


def logdet_distance_squared(first: Tensor, second: Tensor) -> Tensor:
    """Stein/Jensen-Bregman LogDet divergence from Eq. 2."""
    first, second = project_spd(first), project_spd(second)
    middle = 0.5 * (first + second)
    return torch.linalg.slogdet(middle).logabsdet - 0.5 * (
        torch.linalg.slogdet(first).logabsdet
        + torch.linalg.slogdet(second).logabsdet
    )


def frechet_mean_logdet(
    matrices: Tensor | list[Tensor],
    weights: Tensor | list[float],
    num_iterations: int = 5,
    epsilon: float = 1e-5,
) -> Tensor:
    """Weighted Stein mean used by the state and observation updates."""
    if isinstance(matrices, (list, tuple)):
        matrices = torch.stack(list(matrices), dim=-3)
    matrices = project_spd(matrices, epsilon)
    weights = torch.as_tensor(
        weights, dtype=matrices.dtype, device=matrices.device
    )
    if weights.shape[-1] != matrices.shape[-3]:
        raise ValueError("weights must have one entry per input matrix")
    weights = weights.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    while weights.ndim < matrices.ndim - 2:
        weights = weights.unsqueeze(0)

    mean = (weights[..., :, None, None] * matrices).sum(dim=-3)
    for _ in range(num_iterations):
        inverse_sum = (
            weights[..., :, None, None]
            * torch.linalg.inv(mean.unsqueeze(-3) + matrices)
        ).sum(dim=-3)
        mean = project_spd(0.5 * torch.linalg.inv(inverse_sum), epsilon)
    return mean


def _skew(matrix: Tensor) -> Tensor:
    return matrix - matrix.transpose(-1, -2)


def _orthogonal(generator: Tensor, delta: Tensor | float = 1.0) -> Tensor:
    """Discretized group element ``exp(delta A)`` from Eq. 10."""
    return torch.matrix_exp(torch.as_tensor(delta, device=generator.device) * _skew(generator))


def _translate(matrix: Tensor, group_element: Tensor) -> Tensor:
    """The congruence action ``g X g^T`` in Eq. 9."""
    return group_element @ matrix @ group_element.transpose(-1, -2)


class StateSpaceDiscretization(nn.Module):
    """Zero-order-hold discretization used in Eq. 10.

    The paper prints the usual state-space discretization in compact form.
    Solving a linear system is more stable than forming ``A^{-1}`` explicitly.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(dimension, dimension))
        self.b = nn.Parameter(torch.eye(dimension))
        self.log_delta = nn.Parameter(torch.tensor(0.0))

    def forward(self) -> tuple[Tensor, Tensor]:
        delta = self.log_delta.exp()
        a_delta = delta * self.a
        a_bar = torch.matrix_exp(a_delta)
        identity = torch.eye(self.a.shape[0], dtype=self.a.dtype, device=self.a.device)
        # This augmented exponential is valid even when A is singular.
        block = torch.cat(
            (
                torch.cat((a_delta, delta * self.b), dim=-1),
                torch.zeros_like(torch.cat((self.a, self.b), dim=-1)),
            ),
            dim=-2,
        )
        block_exp = torch.matrix_exp(block)
        b_bar = block_exp[: self.a.shape[0], self.a.shape[0] :]
        return a_bar, b_bar


def state_space_kernel(a: Tensor, b: Tensor, c: Tensor, length: int) -> Tensor:
    """Build ``(CB, CAB, CA^2B, ...)`` from Eq. 14."""
    if length < 1:
        raise ValueError("length must be positive")
    power_b = b
    terms = []
    for _ in range(length):
        terms.append(c @ power_b)
        power_b = a @ power_b
    return torch.stack(terms, dim=0)


class SpatiallyPreservingAttention(nn.Module):
    """Spatially preserving attention from Eqs. 15--17."""

    def __init__(self, time_points: int, kernel_size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.root = nn.Parameter(
            torch.empty(time_points, time_points, kernel_size, kernel_size)
        )
        nn.init.xavier_uniform_(self.root)
        self.padding = (kernel_size - 1) // 2
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Eq. 15: K_hat = K^T K + epsilon I, followed by normalization.
        kernel = self.root.transpose(-1, -2) @ self.root
        eye_k = torch.eye(kernel.shape[-1], dtype=x.dtype, device=x.device)
        kernel = kernel + self.epsilon * eye_k
        kernel = kernel / kernel.norm(dim=(-2, -1), keepdim=True).clamp_min(self.epsilon)

        # Eq. 16 uses [X] = diag(rho, X): pad the SPD matrix before applying
        # the valid convolution. The small diagonal completes the padded block.
        padded = torch.nn.functional.pad(
            x, (self.padding, self.padding, self.padding, self.padding)
        )
        padded_identity = torch.eye(padded.shape[-1], dtype=x.dtype, device=x.device)
        padded = padded + self.epsilon * padded_identity
        response = torch.nn.functional.conv2d(padded, kernel)
        response = project_spd(response, self.epsilon)
        response = response / response.norm(
            dim=(-2, -1), keepdim=True
        ).clamp_min(self.epsilon)
        # Here exp is the matrix exponential from the Riemannian algebra; see
        # Appendix A.4. Dividing by the largest entry gives Eq. 16's soft mask.
        attention = torch.matrix_exp(response)
        attention = attention / attention.amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(self.epsilon)

        # Eq. 17 applies the mask to X*K, not to the unfiltered input X.
        attended = attention * response
        identity = torch.eye(attended.shape[-1], dtype=x.dtype, device=x.device)
        return project_spd(attended, self.epsilon) + self.epsilon * identity, attention


class GeometricStateLayer(nn.Module):
    """One history-aware geometric state layer from Eqs. 7--10."""

    def __init__(self, dimension: int, tau: int, stein_iterations: int, epsilon: float) -> None:
        super().__init__()
        if tau < 1:
            raise ValueError("tau must be positive")
        self.dimension = dimension
        self.tau = tau
        self.stein_iterations = stein_iterations
        self.epsilon = epsilon

        # One transformation is learned for every lag in the history window.
        # The skew exponential gives the O(N) action required by Eq. 9, while
        # also providing A_tilde = exp(delta A) from Eq. 10.
        self.a = nn.Parameter(torch.zeros(tau, dimension, dimension))
        self.b = nn.Parameter(torch.zeros(tau + 1, dimension, dimension))
        self.c = nn.Parameter(torch.zeros(tau + 1, dimension, dimension))
        self.d = nn.Parameter(torch.zeros(tau + 1, dimension, dimension))
        self.log_delta = nn.Parameter(torch.tensor(0.0))
        self.state_logits = nn.Parameter(torch.zeros(2 * tau + 1))
        self.output_logits = nn.Parameter(torch.zeros(2 * (tau + 1)))

    def _weighted_mean(self, values: list[Tensor], logits: Tensor) -> Tensor:
        return frechet_mean_logdet(
            values,
            logits.softmax(dim=0),
            num_iterations=self.stein_iterations,
            epsilon=self.epsilon,
        )

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        states: list[Tensor] = []
        outputs: list[Tensor] = []
        delta = self.log_delta.exp()
        # A, B, C and D are constant across a sequence. Discretize them once
        # here instead of rebuilding the same matrix exponential at every step.
        a_group = _orthogonal(self.a, delta)
        b_group = _orthogonal(self.b, delta)
        c_group = _orthogonal(self.c, delta)
        d_group = _orthogonal(self.d, delta)

        for step in range(length):
            start = max(0, step - self.tau)
            state_window = states[start:step]
            input_window = [x[:, j] for j in range(start, step + 1)]

            state_terms = [
                _translate(value, a_group[-len(state_window) + i])
                for i, value in enumerate(state_window)
            ]
            input_terms = [
                _translate(value, b_group[-len(input_window) + i])
                for i, value in enumerate(input_window)
            ]
            state_weights = torch.cat(
                (
                    self.state_logits[self.tau - len(state_window) : self.tau],
                    self.state_logits[-len(input_window) :],
                )
            )
            state = self._weighted_mean(state_terms + input_terms, state_weights)
            states.append(state)

            output_states = states[max(0, len(states) - self.tau - 1) :]
            output_inputs = input_window
            observed = [
                _translate(value, c_group[-len(output_states) + i])
                for i, value in enumerate(output_states)
            ]
            skipped = [
                _translate(value, d_group[-len(output_inputs) + i])
                for i, value in enumerate(output_inputs)
            ]
            output_weights = torch.cat(
                (
                    self.output_logits[
                        self.tau + 1 - len(output_states) : self.tau + 1
                    ],
                    self.output_logits[-len(output_inputs) :],
                )
            )
            outputs.append(self._weighted_mean(observed + skipped, output_weights))

        return torch.stack(outputs, dim=1)


class GeoDynamics(nn.Module):
    """Two-layer GeoDynamics network using the paper's full readout."""

    def __init__(
        self,
        dimension: int,
        time_points: int,
        num_classes: int,
        kernel_size: int = 5,
        tau: int = 3,
        num_layers: int = 2,
        stein_iterations: int = 3,
        readout: str = "final",
        feature_normalization: str = "none",
        vectorization: str = "full",
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("Table 6 specifies two GeoDynamics layers")
        self.dimension = dimension
        self.time_points = time_points
        if readout not in {"final", "mean", "frechet"}:
            raise ValueError("readout must be 'final', 'mean', or 'frechet'")
        self.readout = readout
        if feature_normalization not in {"none", "batch"}:
            raise ValueError("feature_normalization must be 'none' or 'batch'")
        if vectorization not in {"full", "upper"}:
            raise ValueError("vectorization must be 'full' or 'upper'")
        self.vectorization = vectorization
        self.spa = SpatiallyPreservingAttention(time_points, kernel_size, epsilon)
        self.layers = nn.ModuleList(
            GeometricStateLayer(dimension, tau, stein_iterations, epsilon)
            for _ in range(num_layers)
        )
        # Eqs. 11--13 use the complete vec(log(Y^K)) and one linear softmax
        # classifier. CrossEntropyLoss applies softmax during training.
        feature_size = (
            dimension * dimension
            if vectorization == "full"
            else dimension * (dimension + 1) // 2
        )
        upper = torch.triu_indices(dimension, dimension)
        self.register_buffer("upper_rows", upper[0], persistent=False)
        self.register_buffer("upper_columns", upper[1], persistent=False)
        self.feature_normalization = (
            nn.BatchNorm1d(feature_size, affine=False)
            if feature_normalization == "batch"
            else nn.Identity()
        )
        self.classifier = nn.Linear(feature_size, num_classes)
        self.temporal_readout_logits = (
            nn.Parameter(torch.zeros(time_points)) if readout == "frechet" else None
        )
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1:] != (
            self.time_points,
            self.dimension,
            self.dimension,
        ):
            raise ValueError(
                f"expected (B,{self.time_points},{self.dimension},{self.dimension}), "
                f"got {tuple(x.shape)}"
            )
        x, attention = self.spa(project_spd(x, self.epsilon))
        for layer in self.layers:
            x = layer(x)
        if self.readout == "frechet":
            final_output = frechet_mean_logdet(
                x,
                self.temporal_readout_logits.softmax(dim=0),
                num_iterations=self.layers[0].stein_iterations,
                epsilon=self.epsilon,
            )
        elif self.readout == "mean":
            # The released experiment driver pools over the complete output
            # sequence. A convex mean of SPD matrices remains SPD.
            final_output = x.mean(dim=1)
        else:
            final_output = x[:, -1]
        final_tangent = matrix_log_spd(final_output, self.epsilon)
        if self.vectorization == "upper":
            features = final_tangent[:, self.upper_rows, self.upper_columns]
        else:
            features = final_tangent.reshape(x.shape[0], -1)
        features = self.feature_normalization(features)
        logits = self.classifier(features)
        return logits, attention
