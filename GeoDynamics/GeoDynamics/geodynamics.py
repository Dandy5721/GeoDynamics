import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch import Tensor
from spd import *
from einops import rearrange, repeat
import torch.optim as optim
def logdet_distance_squared(X: Tensor, Y: Tensor) -> Tensor:
    """
    Compute the squared distance under the LogDet metric between SPD matrices X and Y:
    d^2(X, Y) = log(det((X+Y)/2)) - 0.5*(log(det(X)) + log(det(Y))).
    
    Uses torch.slogdet for numerical stability.
    """
    mid = (X + Y) / 2
    _, logdet_mid = torch.slogdet(mid)
    _, logdet_X   = torch.slogdet(X)
    _, logdet_Y   = torch.slogdet(Y)
    return logdet_mid - 0.5 * (logdet_X + logdet_Y)


def frechet_mean_logdet(X_list: list, weights, num_iterations: int = 50, lr: float = 1e-3, tol: float = 1e-6) -> Tensor:
    """
    Compute the weighted Fréchet mean of SPD matrices X_list under the LogDet metric.
    
    The Fréchet mean F* minimizes:
      F* = argmin_F sum_{n=1}^N w_n * d^2(X_n, F),
    where d^2 is defined by the LogDet metric.
    
    We parameterize F as F = L L^T + eps * I to ensure F is SPD.
    
    Args:
        X_list (list of Tensors): List of SPD matrices (each of shape (N, N)).
        weights (list or Tensor): Non-negative weights that sum to 1.
        num_iterations (int): Maximum number of iterations.
        lr (float): Learning rate for the optimizer.
        tol (float): Tolerance for convergence.
        
    Returns:
        Tensor: The computed weighted Fréchet mean (an SPD matrix of shape (N, N)).
    """
    N = X_list[0].shape[0]
    eps = 1e-4  # Small constant to ensure SPD
    # Initialize L as an identity matrix; F = L L^T + eps*I remains SPD.
    L = torch.eye(N, dtype=torch.float32, requires_grad=True)
    
    optimizer = optim.Adam([L], lr=lr)
    
    # Stack input matrices for efficient computation; shape: (M, N, N)
    X_tensor = torch.stack(X_list)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    prev_loss = None
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        F = L @ L.t() + eps * torch.eye(N, dtype=torch.float32)
        
        loss = 0.0
        for i in range(X_tensor.shape[0]):
            d2 = logdet_distance_squared(X_tensor[i], F)
            loss = loss + weights[i] * d2
        
        loss.backward()
        optimizer.step()
        
        if prev_loss is not None and abs(loss.item() - prev_loss) < tol:
            break
        prev_loss = loss.item()
    
    F_opt = L.detach() @ L.detach().t() + eps * torch.eye(N, dtype=torch.float32)
    return F_opt

def log_euclidean_mean(X_list, weights=None):
    """
    Compute the (weighted) Log-Euclidean mean of a list of SPD matrices.
    
    Args:
        X_list (list of Tensors): List of SPD matrices, each of shape (N, N).
        weights (Tensor or list): Optional, weights that sum to 1. If None, use uniform weights.
        
    Returns:
        Tensor: Log-Euclidean mean SPD matrix (N x N).
    """
    X_tensor = torch.stack(X_list)  # Shape: (M, N, N)
    M, N, _ = X_tensor.shape

    if weights is None:
        weights = torch.ones(M, dtype=torch.float32) / M
    else:
        weights = torch.tensor(weights, dtype=torch.float32)

    # Matrix logarithms
    X_log = torch.linalg.logm(X_tensor)  # Shape: (M, N, N)

    # Weighted sum in the log-domain
    X_log_mean = torch.einsum("i,ijk->jk", weights, X_log)

    # Exponential map to return to SPD space
    X_lem = torch.linalg.expm(X_log_mean)

    return X_lem

def FM(A: Tensor, B: Tensor, a: float, n: int) -> Tensor:
    """
    Compute the weighted Fréchet mean (Euclidean approximation) of two SPD matrices.
    
    Args:
        A (Tensor): First SPD matrix.
        B (Tensor): Second SPD matrix.
        a (float): Weight coefficient.
        n (int): (Unused in the current implementation)
    
    Returns:
        Tensor: (1-a)*A + a*B.
    """
    return (1.0 - a) * A + a * B


def add_small_diagonal(matrix: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Add a small value to the diagonal elements of a matrix for numerical stability.
    """
    return matrix + torch.eye(matrix.size(0), device=matrix.device) * epsilon


def MatrixExp(B, l, n):
    '''
    input a matrix B, and the total length to be calculated, n is the size of B
    output the somehow exp(B) = I + B + B^2 / 2! + B^3 / 3! + ... + B^l / l!
    '''

    #Result = torch.eye(n, device=B.device)
    #B = (B + B.T)/2 
    #B = add_small_diagonal(B)
    #matrix_to_invert = torch.subtract(Result, B)
    Result = torch.eye(n, device=B.device).unsqueeze(0).repeat(B.shape[0], 1, 1)
    # return torch.matmul(torch.inverse(matrix_to_invert), torch.add(Result, B))
    return torch.matmul(torch.inverse(torch.subtract(Result, B)), torch.add(Result, B))
    # return torch.matmul(torch.linalg.inv(Result - B), Result + B)

def Translation(A: Tensor, B: Tensor, n: int, batch_size: int) -> Tensor:
    """
    Perform the Riemannian translation operation on SPD matrices.
    
    The translation is defined as: T = G X G^T,
    where G is constructed from a parameter vector B as follows:
      - Reshape B into a row vector.
      - Construct a lower triangular matrix by iterating over each row,
        extracting a slice of B and padding with zeros.
      - Form a skew-symmetric matrix: G_skew = L - L^T.
      - Map G_skew into a rotation matrix using the matrix exponential.
      - Compute T = G X G^T.
    
    Args:
        A (Tensor): Input SPD matrix with shape [batch_size, n, n].
        B (Tensor): Input vector used to construct the translation matrix.
        n (int): Dimension of the square matrix.
        batch_size (int): Batch size.
    
    Returns:
        Tensor: Translated SPD matrix.
    """
    power_matrix = 5
    # Reshape B into a row vector.
    B = torch.reshape(B, [1, -1])
    
    # Construct a lower triangular matrix from B.
    lines = [torch.zeros([1, n], device=A.device)]
    for i in range(n - 1):
        temp_line = torch.cat(
            [B[:, i:2 * i + 1], torch.zeros([1, n - i - 1], device=A.device)], dim=1
        )
        lines.append(temp_line)
    lower_triangle = torch.cat(lines, dim=0)
    
    # Form the skew-symmetric matrix.
    B_matrix = lower_triangle - lower_triangle.T
    # Compute the matrix exponential to obtain G in O(n).
    B_matrix = MatrixExp(B_matrix, power_matrix, n)
    # Expand B_matrix to match the batch dimension.
    B_matrix = B_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Compute the translated SPD matrix: T = B_matrix * A * B_matrix^T.
    Tresult = torch.matmul(B_matrix, A)
    Tresult = torch.matmul(Tresult, B_matrix.transpose(1, 2))
    return Tresult


def Translation_time(A, B, n, batch_size):
    """
    Perform a batched similarity transformation of the input matrix A using a skew-symmetric
    matrix derived from vector B. The transformation is of the form:

        T_result = B_matrix @ A @ B_matrix.T

    where B_matrix ∈ O(n), i.e., a rotation matrix constructed by exponentiating a skew-symmetric
    matrix built from B.

    Args:
        A (Tensor): A 4D tensor of shape [T, B, n, n], typically representing a batch of dynamic
                    functional connectivity matrices over time and samples.
        B (Tensor): A 1D vector used to construct a skew-symmetric matrix. It is reshaped and processed
                    into a rotation matrix via matrix exponential.
        n (int):    The spatial dimension of the square matrices in A.
        batch_size (int): Number of samples in the batch. Used to repeat the B_matrix accordingly.

    Returns:
        Tensor: A 4D tensor of the same shape as A, where each matrix has been transformed by the
                same O(n) rotation matrix: B_matrix @ A @ B_matrix.T

    Notes:
        - The input vector B is reshaped and used to construct a lower triangular matrix.
        - This lower triangular matrix is made skew-symmetric, then exponentiated to obtain an O(n) matrix.
        - The same rotation is applied across all time steps and samples in the batch.
    """

    power_matrix = 5  # Degree of Taylor approximation for matrix exponential
    B = torch.reshape(B, [1, -1])  # Make B a row vector

    # Construct a lower triangular matrix from B
    line_B = [torch.zeros([1, n], device=A.device)]
    for i in range(n - 1):
        temp_line = torch.cat(
            [B[:1, i: 2 * i + 1], torch.zeros([1, n - i - 1], device=A.device)],
            dim=1
        )
        line_B.append(temp_line)
    lower_triangle = torch.cat(line_B, dim=0)

    # Make it skew-symmetric: B_matrix = L - L^T
    B_matrix = torch.subtract(lower_triangle, lower_triangle.T)

    # Convert skew-symmetric matrix to rotation matrix via exponential map
    B_matrix = MatrixExp(B_matrix, power_matrix, n)  # Assumed external function

    # Reshape and repeat B_matrix across batch
    B_matrix = B_matrix.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, n, n]
    B_matrix = B_matrix.repeat(batch_size, 1, 1, 1)  # Shape: [B, 1, n, n]

    # Apply similarity transformation: B A B^T
    A = torch.einsum('tbmn,b1mn->tbmn', A, B_matrix)
    Tresult = torch.einsum('tbmn,b1nm->tbmn', A, B_matrix.permute(0, 1, 3, 2))

    return Tresult


class GeoMind(nn.Module):

# Will release after acceptance.
