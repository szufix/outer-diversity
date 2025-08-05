import numpy as np

class Election:
    def __init__(self, num_voters, num_candidates):
        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.fake = False
        self.votes = None
        self.potes = None

    def _compute_potes(self):
        """ Convert votes to positional votes (called potes) """
        self.potes = np.array([[list(vote).index(i) for i, _ in enumerate(vote)]
                                       for vote in self.votes])
        return self.potes

    def get_potes(self):
        if self.potes is None:
            self._compute_potes()
        return self.potes