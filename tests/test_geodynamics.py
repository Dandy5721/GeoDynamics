import torch

from GeoDynamics.geodynamics import GeoDynamics, frechet_mean_logdet, project_spd


def test_model_forward_shapes():
    model = GeoDynamics(
        dimension=3,
        time_points=4,
        num_classes=2,
        kernel_size=3,
        tau=1,
        stein_iterations=1,
    )
    inputs = torch.eye(3).repeat(2, 4, 1, 1)
    logits, attention = model(inputs)
    assert logits.shape == (2, 2)
    assert attention.shape == inputs.shape
    assert torch.isfinite(logits).all()


def test_weighted_stein_mean_is_spd():
    matrices = torch.stack((torch.eye(3), 2 * torch.eye(3)))
    mean = frechet_mean_logdet(matrices, torch.tensor([0.25, 0.75]), num_iterations=2)
    assert torch.linalg.eigvalsh(project_spd(mean)).min() > 0
