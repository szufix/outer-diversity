from src.domain.validator import validate_domain
import numpy as np

@validate_domain
def extend_to_reversed(domain):
    """
    Extends a domain to include the reverse of each ranking, ensuring uniqueness.
    Accepts rankings as lists or tuples.
    """
    as_tuples = [tuple(r) for r in domain]  # ensure all are tuples
    extended = set(as_tuples) | {tuple(reversed(r)) for r in as_tuples}
    extended = np.unique(list(extended), axis=0)
    return extended
