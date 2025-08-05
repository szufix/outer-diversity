from matplotlib import pyplot as plt
from fractions import Fraction


MAX_DOMAIN_SIZE = [0,
                   1, 2, 6, 24, 120,
                   720, 5040, 40320, 362880, 3628800,
                   39916800, 479001600, 6227020800, 87178291200, 1307674368000,
                   20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000]

SINGLE_PEAKED_DOMAIN_SIZE = [0,
                             1, 2, 4, 8, 16,
                             32, 64, 128, 256, 512,
                             1024, 2048, 4096, 8192, 16384,
                             32768, 65536, 131072, 262144, 524288]

SPOC_DOMAIN_SIZE = [0,
                    1, 2, 6, 16, 40,
                    96, 224, 512, 1152, 2560,
                    5632, 12288, 26624, 57344, 122880,
                    262144, 557056, 1179648, 2490368, 5242880]


SP_DOUBLE_FORKED_DOMAIN_SIZE = [0,
                              0, 0, 0, 0, 48,
                              112, 240, 496, 1008, 2032,
                              4080, 8176, 16368, 32752, 65520,
                              131056, 262128, 524272, 1048560, 2097136]

GROUP_SEPARABLE_DOMAIN_SIZE = SINGLE_PEAKED_DOMAIN_SIZE

SINGLE_CROSSING_DOMAIN_SIZE = [0,
                               1, 2, 4, 7, 11,
                               16, 22, 29, 37, 46,
                               56, 67, 79, 92, 106,
                               121, 137, 154, 172, 191]

EUC_1D_DOMAIN_SIZE = SINGLE_CROSSING_DOMAIN_SIZE

EUC_2D_DOMAIN_SIZE = [0,
                      1, 2, 6, 18, 46,
                      101, 197, 351, 583, 916,
                      1376, 1992, 2796, 3823, 5111,
                      6701, 8637, 10966, 13738, 17006]

EUC_3D_DOMAIN_SIZE = [0,
                      1, 2, 6, 24, 96,
                      326, 932, 2311, 5119, 10366,
                      19526, 34662, 58566, 94914, 148436,
                      225101, 332317, 479146, 676534, 937556]


######################################################################


def get_single_peaked_domain_size(num_candidates):
    m = Fraction(num_candidates)
    return 2**(m - 1)

def get_spoc_domain_size(num_candidates):
    m = Fraction(num_candidates)
    return m * 2**(m - 2)

def get_sp_double_forked_domain_size(num_candidates):
    m = Fraction(num_candidates)
    return 16 * (2**(m - 3) - 1)

def get_group_separable_domain_size(num_candidates):
    return get_single_peaked_domain_size(num_candidates)

def get_single_crossing_domain_size(num_candidates):
    m = Fraction(num_candidates)
    return (m**2 - m + 2) / 2

def get_euc_1d_domain_size(num_candidates):
    return get_single_crossing_domain_size(num_candidates)

def get_euc_2d_domain_size(num_candidates):
    m = Fraction(num_candidates)
    return (
        Fraction(1, 24) * (24 - 14 * m + 21 * m**2 - 10 * m**3 + 3 * m**4)
    )

def get_euc_3d_domain_size(num_candidates):
    m = Fraction(num_candidates)
    return (
        Fraction(1)
        - Fraction(7, 12) * m
        + m**2
        - Fraction(37, 48) * m**3
        + Fraction(23, 48) * m**4
        - Fraction(7, 48) * m**5
        + Fraction(1, 48) * m**6
    )


def test_domain_sizes():
    for i in range(1, 20 + 1):
        assert get_single_peaked_domain_size(i) == SINGLE_PEAKED_DOMAIN_SIZE[i], \
            (f"Single peaked size mismatch for {i}; "
             f"expected {SINGLE_PEAKED_DOMAIN_SIZE[i]}, got {get_single_peaked_domain_size(i)}")
        if i >= 2:
            assert get_spoc_domain_size(i) == SPOC_DOMAIN_SIZE[i], \
                (f"SPOC size mismatch for {i}; "
                 f"expected {SPOC_DOMAIN_SIZE[i]}, got {get_spoc_domain_size(i)}")
        if i >= 5:
            assert get_sp_double_forked_domain_size(i) == SP_DOUBLE_FORKED_DOMAIN_SIZE[i], \
                (f"SP Double Forked size mismatch for {i}; "
                 f"expected {SP_DOUBLE_FORKED_DOMAIN_SIZE[i]}, got {get_sp_double_forked_domain_size(i)}")
        assert get_group_separable_domain_size(i) == GROUP_SEPARABLE_DOMAIN_SIZE[i], \
            (f"Group separable size mismatch for {i}; "
             f"expected {GROUP_SEPARABLE_DOMAIN_SIZE[i]}, got {get_group_separable_domain_size(i)}")
        assert get_single_crossing_domain_size(i) == SINGLE_CROSSING_DOMAIN_SIZE[i], \
            (f"Single crossing size mismatch for {i}; "
             f"expected {SINGLE_CROSSING_DOMAIN_SIZE[i]}, got {get_single_crossing_domain_size(i)}")
        assert get_euc_1d_domain_size(i) == EUC_1D_DOMAIN_SIZE[i], \
            (f"EUC 1D size mismatch for {i}; "
             f"expected {EUC_1D_DOMAIN_SIZE[i]}, got {get_euc_1d_domain_size(i)}")
        assert get_euc_2d_domain_size(i) == EUC_2D_DOMAIN_SIZE[i], \
            (f"EUC 2D size mismatch for {i}; "
             f"expected {EUC_2D_DOMAIN_SIZE[i]}, got {get_euc_2d_domain_size(i)}")
        assert get_euc_3d_domain_size(i) == EUC_3D_DOMAIN_SIZE[i], \
            (f"EUC 3D size mismatch for {i}; "
             f"expected {EUC_3D_DOMAIN_SIZE[i]}, got {get_euc_3d_domain_size(i)}")

# test_domain_sizes()
