
def domain_to_set(domain):
    """
    Converts a domain into a set of tuples
    """
    return {tuple(vote) for vote in domain}


def domain_to_list(domain):
    """
    Converts a domain into a list of tuples
    """
    return [tuple(vote) for vote in domain]

def verify_domain_size(domain):
    return len(domain_to_set(domain)) == len(domain)


def validate_domain(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not verify_domain_size(result):
            raise ValueError("Invalid domain: must consist of unique rankings.")
        return result
    return wrapper