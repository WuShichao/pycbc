"""Which shared regions the eccentric PE builds: every grid sized before
any is built (PhaseGrid size_only), the grids' memory held one after
another, and a region refused when building it would take longer than the
direct evaluation of its points (pycbc.filter.eccentric_shared.admit_region,
EccentricCellPE.share)."""
import numpy
import pytest

from test_nufft_correlation import correlator, templates


def test_size_only_lays_out_the_grid_a_build_makes():
    from pycbc.filter.nufft_correlation import PhaseGrid
    c = correlator()
    region = templates(12)
    sized = PhaseGrid(c, region, (-2e4, 2e4), taylor_tolerance=1e-9, size_only=True)
    built = PhaseGrid(c, region, (-2e4, 2e4), taylor_tolerance=1e-9)
    assert not hasattr(sized, 'grid')
    assert list(sized.size) == list(built.size)
    assert sized.terms == built.terms
    assert sized.need == built.need and sized.kept == built.kept
    # what the build keeps is the grid itself
    assert built.kept == built.grid.nbytes
    assert sized.vectors == len(built.grid)


def test_size_only_refuses_what_a_build_refuses():
    from pycbc.filter.nufft_correlation import PhaseGrid
    c = correlator()
    region = templates(12)
    need = PhaseGrid(c, region, (-2e4, 2e4), taylor_tolerance=1e-9, size_only=True).need
    with pytest.raises(MemoryError):
        PhaseGrid(c, region, (-2e4, 2e4), taylor_tolerance=1e-9, size_only=True, memory_limit=0.5 * need)


def sizes(*grids, limit=10e9):
    # (need, kept, vectors, limit, layout) per grid
    return [(need, kept, vectors, limit, [n, 16, n]) for need, kept, vectors, n in grids]


def test_grids_held_one_after_another_must_fit_together():
    from pycbc.filter.eccentric_shared import admit_region, RegionTooCostly
    # each fits alone, the third not after the first two
    grids = sizes((4e9, 3e9, 100, 1), (4e9, 3e9, 100, 2), (5e9, 4e9, 100, 3))
    with pytest.raises(MemoryError) as err:
        admit_region(grids)
    assert not isinstance(err.value, RegionTooCostly)
    assert err.value.size == [3, 16, 3] and err.value.need == pytest.approx(11e9)
    out = admit_region(grids[:2])
    assert out['peak'] == pytest.approx(7e9)
    assert out['work'] == pytest.approx(800.0)


def test_a_region_slower_to_build_than_its_points_directly_is_refused():
    from pycbc.filter.eccentric_shared import admit_region, RegionTooCostly
    # the pp injection of 2026-11-18: a 28.5 GB grid of 252 vectors for a region of 64 points (0.13 s each
    # directly), and a frame region's three grids of 1-3 GB for 3000
    big = sizes((28.5e9, 20e9, 252, 434), limit=52e9)
    with pytest.raises(RegionTooCostly) as err:
        admit_region(big, max_build_seconds=2 * 64 * 0.13, build_rate=0.05)
    assert err.value.seconds == pytest.approx(0.05 * 28.5 * 252)
    small = sizes((1.48e9, 1e9, 114, 112), (0.7e9, 0.5e9, 84, 66), (3.13e9, 2e9, 138, 168), limit=52e9)
    out = admit_region(small, max_build_seconds=2 * 3000 * 0.13, build_rate=0.05)
    assert out['seconds'] == pytest.approx(0.05 * (1.48 * 114 + 0.7 * 84 + 3.13 * 138))
    # without a limit only the memory counts
    assert admit_region(big)['seconds'] > 300


class _Shares(object):
    """EccentricCellPE's share bookkeeping over a stand-in region."""
    def __init__(self, refuse):
        from pycbc.inference.models.eccentric_bstatistic import EccentricCellPE
        self.model = EccentricCellPE.__new__(EccentricCellPE)
        self.model.shared_options = dict(data_window=(0., 1.))
        self.model.xp = numpy
        self.calls = []
        test = self

        class Region(object):
            def __init__(self, model, region, **options):
                test.calls.append((len(region), options))
                err = refuse(len(region), options)
                if err is not None:
                    raise err
                self.planned = dict(work=100.0, seconds=5.0, peak=1e9)
                self.build_seconds = 20.0
        self.Region = Region


def points(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return [dict(mchirp=10 + rng.random(), eccentricity=rng.random() * .1, chi_eff=rng.random(), q=1 + rng.random(),
                 eclipticlongitude=rng.random(), eclipticlatitude=rng.random()) for _ in range(n)]


def test_share_allows_payback_times_the_direct_cost_and_learns_the_build_rate(monkeypatch):
    import pycbc.filter.eccentric_shared as es
    s = _Shares(lambda n, o: None)
    monkeypatch.setattr(es, 'EccentricSharedRegion', s.Region)
    m = s.model
    m.shared_statistics.update(single_points=1000, single_seconds=120.0)
    m.share(points(500))
    n, options = s.calls[0]
    assert options['max_build_seconds'] == pytest.approx(m.share_payback * 500 * 0.12)
    assert options['build_rate'] == pytest.approx(0.05)
    # the region built at 0.2 s per GB-vector: the rate moves towards it
    assert 0.05 < m.build_rate < 0.2
    assert m.build_rate == pytest.approx(numpy.exp(0.7 * numpy.log(0.05) + 0.3 * numpy.log(0.2)))


def test_a_costly_region_is_not_split_and_a_large_one_is(monkeypatch):
    import pycbc.filter.eccentric_shared as es
    costly = _Shares(lambda n, o: es.RegionTooCostly('slow'))
    monkeypatch.setattr(es, 'EccentricSharedRegion', costly.Region)
    assert costly.model._share_split(points(4000)) is False
    assert len(costly.calls) == 1
    assert costly.model.shared_statistics['regions_too_costly'] == 1
    # a region that does not fit is halved (up to three times) until its parts do
    large = _Shares(lambda n, o: MemoryError('large') if n > 1000 else None)
    monkeypatch.setattr(es, 'EccentricSharedRegion', large.Region)
    assert large.model._share_split(points(4000)) is True
    assert [n for n, _ in large.calls] == [4000, 2000, 1000, 1000, 2000, 1000, 1000]
