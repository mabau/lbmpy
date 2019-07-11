from lbmpy.creationfunctions import create_lb_function
from lbmpy.scenarios import create_lid_driven_cavity
from pystencils import show_code


def test_creation():
    """Simple test that makes sure that only float variables are created"""

    func = create_lb_function(method='srt', relaxation_rate=1.5,
                              optimization={'double_precision': False})
    code = str(show_code(func.ast))
    assert 'double' not in code


def test_scenario():
    sc = create_lid_driven_cavity((16, 16, 8), relaxation_rate=1.5,
                                  optimization={'double_precision': False})
    sc.run(1)
    code_str = str(show_code(sc.ast))
    assert 'double' not in code_str
