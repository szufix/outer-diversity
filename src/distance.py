

def _distance_between_vote_and_sp_domain(vote, domain, num_candidates):
    pass


def distance_between_vote_and_domain(vote, domain, num_candidates, hint=None):
    if hint == 'sp':
        return _distance_between_vote_and_sp_domain(vote, domain, num_candidates)

