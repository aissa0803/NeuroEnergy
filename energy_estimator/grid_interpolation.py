import torch


def clamp_index(
    grid: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the two neighboring grid indices for x and compute the interpolation weight.

    The index lookup is done without gradient so that only the fractional weight t
    carries the gradient — this is what makes bilinear interpolation differentiable.

    Args:
        grid: Sorted 1-D tensor of grid breakpoints.
        x: Scalar tensor to locate in the grid.

    Returns:
        lo: Index of the left neighbor.
        hi: Index of the right neighbor.
        t: Fractional position of x between grid[lo] and grid[hi], in [0, 1].
    """
    with torch.no_grad():
        x_lookup = x.detach().clamp(grid[0], grid[-1])
        hi = torch.searchsorted(grid, x_lookup, right=False)
        hi = hi.clamp(1, len(grid) - 1)
        lo = hi - 1

    lo_val = grid[lo]
    hi_val = grid[hi]
    span = hi_val - lo_val

    # t may go outside [0, 1] for linear extrapolation; gradient flows through x
    t = torch.where(
        span == 0,
        torch.zeros_like(x),
        (x - lo_val) / span,
    )

    return lo, hi, t
