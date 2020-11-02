def sample_from_uniform_range(B, N, min_d, max_d, dev='cpu'):
    '''
    Given min/max ranges of depths, return sampled depths based on uniform distrb.
    '''
    depths = torch.linspace(min_d, max_d, N + 1, device=dev).view(1, -1)
    min_ds = depths[:-1]
    max_ds = depths[1:]
    return torch.rand((B, 1), dtype=torch.float, device=dev) * (max_ds - min_ds) + min_ds)

def integrate_occupancies(occs, depths):
    '''
    Given occupancies and sample depths, compute integral occupancies
    Args:
        - occs: BxN occupancy
        - depths: BxN ordered N depth values for each occupancies
    Returns:
        - Bx(N-1) representing integral occupancy
    '''
    # compute delta = Bx(N-1)
    delta = depths[:, 1:] - depths[:, :-1]

    # compute integrals = Bx(N-1)
    return (-(occs[1:] * delta)).cumsum(1).exp()

def compute_weights(integrals, occs, depths)
    '''
    Compute weights for sampling / rendering

    Args:
        - integrals: Bx(N-1) integral occupancy
        - occs: BxN occupancy
        - depths: BxN depths
    Returns:
        - Bx(N-1) weights
    '''
    # compute delta = Bx(N-1)
    delta = depths[:, 1:] - depths[:, :-1]
    # NF.pad(depths, )

    # compute integrals = Bx(N-1)
    return integrals * (1 - (-delta * occs[:-1]).exp())


def sample_from_coarse(weights):
    raise NotImplementedError

   