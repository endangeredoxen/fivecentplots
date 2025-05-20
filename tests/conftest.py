import pytest
import fivecentplots as fcp



@pytest.fixture(scope="session", autouse=True)
def get_ready(request):
    # prepare something ahead of all tests
    fcp.set_theme('gray_original')
    fcp.KWARGS['engine'] = 'mpl'

