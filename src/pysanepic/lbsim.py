"""litebird_sim interface.

    from pysanepic.lbsim import make_gls_map
    result = make_gls_map(nside=64, observations=sim.observations)

Observations are converted to :class:`pysanepic.DetectorData` (pointings, HWP
angle, TOD and the 1/f noise parameters litebird_sim uses to simulate noise);
everything else is done by :func:`pysanepic.make_maps`. Only public
litebird_sim API is used.
"""

import litebird_sim as lbs
import numpy as np

from .mapmaker import DetectorData, make_maps

_COORDS = {lbs.CoordinateSystem.Galactic: "G", lbs.CoordinateSystem.Ecliptic: "E"}


def from_observations(observations, components="tod"):
    """Convert litebird_sim observations into a list of DetectorData.

    Pointings must be available (``sim.prepare_pointings()``); an HWP set with
    ``sim.set_hwp()`` is included. TOD components in `components` are summed.
    """
    components = [components] if isinstance(components, str) else components
    out = []
    for obs in lbs.normalize_observations(observations):
        net = np.broadcast_to(obs.net_ukrts, (obs.n_detectors,))
        gamma = np.broadcast_to(getattr(obs, "pol_efficiency", 1.0), (obs.n_detectors,))
        for det in range(obs.n_detectors):
            ptg, hwp_angle = obs.get_pointings(det)  # Ecliptic, (n, 3)
            out.append(
                DetectorData(
                    tod=sum(getattr(obs, c)[det] for c in components),
                    theta=ptg[:, 0],
                    phi=ptg[:, 1],
                    psi=ptg[:, 2],
                    hwp_angle=hwp_angle,
                    coordinates="E",
                    sampling_rate_hz=obs.sampling_rate_hz,
                    net_ukrts=net[det],
                    fknee_hz=obs.fknee_mhz[det] / 1e3,
                    alpha=obs.alpha[det],
                    fmin_hz=obs.fmin_hz[det],
                    pol_angle_rad=obs.pol_angle_rad[det],
                    pol_efficiency=gamma[det],
                )
            )
    return out


def make_gls_map(
    nside,
    observations,
    components="tod",
    output_coordinate_system=lbs.CoordinateSystem.Galactic,
    comm=None,
    **kwargs,
):
    """GLS map-making with 1/f noise on litebird_sim observations.

    `output_coordinate_system` follows the litebird_sim map-makers (default
    Galactic). `comm` defaults to litebird_sim's MPI_COMM_WORLD when MPI is
    enabled; every rank must call this function. Other keyword arguments
    (chunk_s, pol, tol, maxiter, verbose) are passed to :func:`make_maps`.
    """
    if comm is None and lbs.MPI_ENABLED and lbs.MPI_COMM_WORLD.size > 1:
        comm = lbs.MPI_COMM_WORLD
    data = [] if observations is None else from_observations(observations, components)
    return make_maps(data, nside, coordinates=_COORDS[output_coordinate_system], comm=comm, **kwargs)
