"""GridSkyResponse interpolates a smooth sky function to fourth order and refuses points outside its grid."""
import numpy as np
import pytest
from pycbc.tdi.ssb_yearly_harmonics import GridSkyResponse


def test_grid_sky_response_is_exact_at_nodes_and_fourth_order_between():
    def f(s):
        return np.exp(1j*(.4*s[0]-.3*s[1]))*np.array([1., 2.+.1*s[0]*s[1]])
    errs=[]
    for h in (1.,.5):
        ax=np.arange(-4,4+h/2,h)
        g=GridSkyResponse((ax,ax),f)
        np.testing.assert_allclose(g(np.array([ax[3],ax[5]])),f(np.array([ax[3],ax[5]])),atol=1e-12)
        rng=np.random.default_rng(3)
        errs.append(max(np.max(np.abs(g(p)-f(p))/np.abs(f(p))) for p in rng.uniform(-4,4,(50,2))))
    # fourth order: halving the step cuts the error by about 16
    assert errs[1]<errs[0]/10, errs
    with pytest.raises(ValueError):
        g(np.array([4.5,0.]))
