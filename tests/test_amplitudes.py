"""_Amplitudes without a solve: labels of the shared and local amplitudes,
priors linearized around the current values, updates and MapResult.templates."""

import numpy as np
import pytest

from pysanepic import (
    CalibrationParameters,
    DetectorData,
    OneOverFNoise,
    Pointing,
    Prior,
    Template,
)
from pysanepic.mapmaker import MapResult, _Amplitudes

FS, N = 10.0, 100  # 10 s per DetectorData


def _det(name, templates):
    return DetectorData(tod=np.zeros(N), pointing=Pointing(pix=np.zeros(N, int), psi=np.zeros(N), nside=1, coordinates="G"),
                        sampling_rate_hz=FS, noise=OneOverFNoise(net_ukrts=1.0), name=name, dipole=np.zeros(N),
                        templates={k: np.ones(N) for k in templates})


TEMPLATES = {"t": Template(), "c": Template(common=True), "s": Template(sky=np.zeros(12), sum_zero=True),
             "p": Template(period_s=4.0, prior=Prior(sigma=2.0, mean=0.5)), "f": Template(fixed=True)}


def _amplitudes(prior=None):
    dets = [_det("b", ["t", "c", "p"]), _det("a", ["c"]), _det("b", ["t", "c", "p"])]
    cal = CalibrationParameters(gain_period_s=5.0, prior=prior)
    return _Amplitudes.build(TEMPLATES, cal, dets, None), dets


def test_shared_labels():
    """One shared amplitude per template and detector with it (names sorted),
    one per common template, none per period or fixed; sum_zero groups."""
    amps, dets = _amplitudes()
    assert amps.labels == [("t", "b"), ("c", None), ("s", "a"), ("s", "b")]
    assert amps.groups == [[2, 3]]
    assert amps.shared_label("t", dets[0]) == 0 and amps.shared_label("t", dets[1]) is None
    assert amps.shared_label("c", dets[1]) == 1 and amps.shared_label("s", dets[1]) == 2
    assert list(amps.local) == ["p"]


@pytest.mark.parametrize("fit", [True, False])
def test_pass_labels_and_priors(fit):
    """In a pass the gains and then the amplitudes per period of each
    DetectorData get consecutive labels, in the order of the data; the priors
    are on g (1 + dg) and a + da, linearized at the current values (none with
    the amplitudes fixed)."""
    amps, dets = _amplitudes(prior=Prior(sigma=0.1, mean=1.2))
    amps.gains = [np.full(2, 2.0), np.full(2, 1.0), np.full(2, 1.0)]
    amps.local["p"] = [np.array([1.0, 2.0, 3.0]), np.zeros(0), np.zeros(3)]
    amps.start_pass(fit)
    out = []
    for j, d in enumerate(dets):
        local = {"p": amps.local_value("p", d, j)} if "p" in d.templates else {}
        out.append(amps.add(d, amps.gain(d, j), local))
    (g0, l0), (g1, l1), (g2, l2) = out
    np.testing.assert_array_equal(g0, np.repeat([0, 1], 50))  # gain periods of 5 s
    np.testing.assert_array_equal(l0["p"], np.repeat([2, 3, 4], [40, 40, 20]))  # periods of 4 s
    np.testing.assert_array_equal(g1, np.repeat([5, 6], 50))
    assert l1 == {}
    np.testing.assert_array_equal(g2, np.repeat([7, 8], 50))
    np.testing.assert_array_equal(l2["p"], np.repeat([9, 10, 11], [40, 40, 20]))
    local, shared = amps.gls_priors()
    assert shared is None
    if not fit:
        assert local is None
        return
    prec, mean = local
    g = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    a = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(prec, np.r_[100 * g[:2] ** 2, [0.25] * 3, 100 * g[2:] ** 2, [0.25] * 3])
    np.testing.assert_allclose(mean, np.r_[(1.2 - g[:2]) / g[:2], 0.5 - a[:3], (1.2 - g[2:4]) / g[2:4],
                                           (1.2 - g[4:]) / g[4:], 0.5 - a[3:]])


def test_update_and_result():
    """A solve updates g (1 + dg), a + da and A + dA; MapResult.templates and
    the amplitudes read back from it (estimate_noise) are the same."""
    amps, dets = _amplitudes()
    amps.shared = np.array([1.0, 2.0, 3.0, -3.0])
    amps.start_pass(True)
    for j, d in enumerate(dets):
        amps.add(d, amps.gain(d, j), {"p": amps.local_value("p", d, j)} if "p" in d.templates else {})
    da = np.arange(12) * 0.1
    amps.update({"amplitudes": da, "shared_amplitudes": np.array([0.5, 0.5, 0.5, -0.5])})
    np.testing.assert_allclose(amps.gains[0], 1 + da[0:2])
    np.testing.assert_allclose(amps.gains[1], 1 + da[5:7])
    np.testing.assert_allclose(amps.local["p"][0], da[2:5])
    assert amps.local["p"][1].size == 0
    assert amps.dg_max == pytest.approx(da[8])
    out = amps.result_templates()
    assert out["t"] == {"b": 1.5} and out["c"] == 2.5 and out["s"] == {"a": 3.5, "b": -3.5} and "f" not in out
    result = MapResult(maps=np.zeros((3, 12)), hit_map=np.zeros(12, int), nside=1, coordinates="G", iterations=0,
                       converged=True, residuals=[], gains=amps.gains, templates=out,
                       calibration=amps.calibration, template_definitions=TEMPLATES)
    back = _Amplitudes.from_result(result)
    for j, d in enumerate(dets):
        for k in ["t", "c", "s"]:
            assert back.shared_value(k, d) == amps.shared_value(k, d)
        np.testing.assert_array_equal(back.local_value("p", d, j)[0], amps.local_value("p", d, j)[0])
        np.testing.assert_array_equal(back.gain(d, j)[0], amps.gain(d, j)[0])
    assert back.count == 3
