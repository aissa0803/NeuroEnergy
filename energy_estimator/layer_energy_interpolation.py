import torch
from .power_lookup_tables import load_table
from .grid_interpolation import clamp_index


def estimate_energy(
    board: str,
    layer: str,
    in_dim: float | torch.Tensor,
    out_dim: float | torch.Tensor,
) -> torch.Tensor:
    """Estimate energy for a single layer using bilinear interpolation.

    Args:
        board: Hardware board name (e.g. "JetsonNano").
        layer: Layer key (e.g. "linear", "conv_k3_p0").
        in_dim: Number of input channels/features.
        out_dim: Number of output channels/features. Accepts a soft tensor
            (e.g. mask.sum()) to keep the result differentiable.

    Returns:
        Scalar float64 tensor with estimated energy in Joules.
        Differentiable with respect to out_dim when out_dim is a tensor.

    Raises:
        ValueError: If the lookup table has NaN near the requested dimensions.
    """

    grid_x, grid_y, values = load_table(board, layer)

    if not isinstance(in_dim, torch.Tensor):
        in_dim = torch.tensor(float(in_dim), dtype=torch.float64)
    elif in_dim.dtype != torch.float64:
        in_dim = in_dim.double()

    if not isinstance(out_dim, torch.Tensor):
        out_dim = torch.tensor(float(out_dim), dtype=torch.float64)
    elif out_dim.dtype != torch.float64:
        out_dim = out_dim.double()


    lo_x, hi_x, tx = clamp_index(grid_x, in_dim)
    lo_y, hi_y, ty = clamp_index(grid_y, out_dim)

    Q11 = values[lo_x, lo_y]
    Q12 = values[lo_x, hi_y]
    Q21 = values[hi_x, lo_y]
    Q22 = values[hi_x, hi_y]
    
    corners = torch.stack([Q11, Q12, Q21, Q22])
    if torch.isnan(corners).any():
        raise ValueError(
            f"Energy table for {board}/{layer} has no measurements near "
            f"in={float(in_dim.detach()):.0f}, out={float(out_dim.detach()):.0f}. "
            f"Corner values: Q11={Q11:.4f}, Q12={Q12:.4f}, Q21={Q21:.4f}, Q22={Q22:.4f}."
        )

    return (
        Q11 * (1 - tx) * (1 - ty)
        + Q12 * (1 - tx) * ty
        + Q21 * tx * (1 - ty)
        + Q22 * tx * ty
    )