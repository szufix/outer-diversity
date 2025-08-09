from src.domain.validator import validate_domain
from src.domain.extenders import extend_to_reversed

@validate_domain
def single_vote_domain(num_candidates):
    """
    Generates a single vote domain with a single ranking of candidates.
    The ranking is simply a list of candidates in order.
    """
    return [list(range(num_candidates))]



@validate_domain
def ext_single_vote_domain(num_candidates):
    """
    Extends the single vote domain to include the reverse of the single ranking.
    This means both the original ranking and its reverse are included.
    """
    domain = single_vote_domain(num_candidates)
    return extend_to_reversed(domain)