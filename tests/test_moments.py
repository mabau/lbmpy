from lbmpy.moments import *
from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil


def test_moment_permutation_multiplicity():
    test_moments = [(2, 0, 0),
                    (1, 2, 0),
                    (1, 0, 2, 0)]

    for m in test_moments:
        assert moment_multiplicity(m) == len(list(moment_permutations(m)))


def test_moment_order():
    r = moments_of_order(2, dim=3, include_permutations=True)
    assert len(list(r)) == 6

    r = moments_of_order(2, dim=3, include_permutations=False)
    assert len(list(r)) == 2

    r = moments_up_to_order(2, dim=3, include_permutations=True)
    assert len(list(r)) == 10

    r = moments_up_to_order(2, dim=3, include_permutations=False)
    assert len(r) == 4

    r = moments_up_to_component_order(2, dim=3)
    assert len(r) == 3**3

    r = moments_up_to_component_order(2, dim=2)
    assert len(r) == 3**2


def test_extend_moments_with_permutations():
    no_perm = moments_of_order(2, dim=3, include_permutations=False)
    with_perm = moments_of_order(2, dim=3, include_permutations=True)

    assert set(extend_moments_with_permutations(no_perm)) == set(with_perm)


def test_representation_conversion():
    x, y, z = MOMENT_SYMBOLS
    e = exponents_to_polynomial_representations([(2, 1, 0), (0, 0, 0)])
    ref = 5 * x**2 * y + 3
    assert sp.simplify(5 * e[0] + 3 * e[1] - ref) == 0

    e = polynomial_to_exponent_representation(x ** 4 * z ** 2)[0][1]
    assert e, (4, 0 == 2)

    e = polynomial_to_exponent_representation(x ** 4 * y ** 2, dim=2)[0][1]
    assert e, (4 == 2)


def test_moment_properties():
    x, y, z = MOMENT_SYMBOLS
    # even - odd
    assert is_even((2, 0, 0))
    assert not is_even((2, 1, 0))
    assert is_even(x ** 2 + y ** 2)
    assert not is_even(z)

    # order
    assert get_order(x ** 4 * z ** 2) == 6
    assert get_order((2, 2, 3)) == 7
    assert get_order(sp.sympify(1)) == 0


def test_gram_schmidt_orthogonalization():
    moments = moments_up_to_component_order(2, 2)
    assert len(moments) == 9

    stencil = LBStencil(Stencil.D2Q9)
    orthogonal_moments = gram_schmidt(moments, stencil)
    pdfs_to_moments = moment_matrix(orthogonal_moments, stencil)
    assert (pdfs_to_moments * pdfs_to_moments.T).is_diagonal()


def test_is_bulk_moment():
    x, y, z = MOMENT_SYMBOLS
    assert not is_bulk_moment(x, 2)
    assert not is_bulk_moment(x ** 3, 2)
    assert not is_bulk_moment(x * y, 2)
    assert not is_bulk_moment(x ** 2, 2)
    assert not is_bulk_moment(x ** 2 + y ** 2, 3)
    assert is_bulk_moment(x ** 2 + y ** 2, 2)
    assert is_bulk_moment(x ** 2 + y ** 2 + z ** 2, 3)
    assert is_bulk_moment(x ** 2 + y ** 2 + x, 2)
    assert is_bulk_moment(x ** 2 + y ** 2 + 1, 2)


def test_is_shear_moment():
    x, y, z = MOMENT_SYMBOLS
    assert not is_shear_moment(x ** 3, 2)
    assert not is_shear_moment(x, 2)
    assert not is_shear_moment(x ** 2 + y ** 2, 2)
    assert not is_shear_moment(x ** 2 + y ** 2 + z ** 2, 3)
    assert is_shear_moment(x ** 2, 2)
    assert is_shear_moment(x ** 2 - 1, 2)
    assert is_shear_moment(x ** 2 - x, 2)
    assert is_shear_moment(x * y, 2)
    assert is_shear_moment(x * y - 1, 2)
    assert is_shear_moment(x * y - x, 2)
