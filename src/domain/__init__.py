from src.domain.single_crossing import (
    single_crossing_domain,
    ext_single_crossing_domain
)
from src.domain.single_peaked import (
    single_peaked_domain,
    spoc_domain,
    ext_single_peaked_domain,
)
from src.domain.group_separable import (
    group_separable_balanced_domain,
    group_separable_caterpillar_domain
)
from src.domain.euclidean_ilp import (
    euclidean_1d_domain,
    euclidean_2d_domain,
    euclidean_3d_domain,
    euclidean_4d_domain,
    euclidean_5d_domain,
    ext_euclidean_1d_domain,
    ext_euclidean_2d_domain,
    ext_euclidean_3d_domain,
)
from src.domain.sp_tree import (
    sp_star_domain,
    sp_double_forked_domain,
    sp_triple_forked_domain,
    ext_sp_double_forked_domain,
)

__all__ = [
    'single_crossing_domain',
    'single_peaked_domain',
    'spoc_domain',
    'group_separable_balanced_domain',
    'group_separable_caterpillar_domain',
    'euclidean_1d_domain',
    'euclidean_2d_domain',
    'euclidean_3d_domain',
    'euclidean_4d_domain',
    'euclidean_5d_domain',
    'sp_star_domain',
    'sp_double_forked_domain',
    'sp_triple_forked_domain',
    'ext_euclidean_1d_domain',
    'ext_euclidean_2d_domain',
    'ext_euclidean_3d_domain',
    'ext_single_crossing_domain',
    'ext_single_peaked_domain',
    'ext_sp_double_forked_domain',
]