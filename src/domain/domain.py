import numpy as np


class Domain:
    """
    Represents a domain of votes (permutations) for voting theory applications.
    """
    def __init__(self, votes):
        self.votes = [tuple(vote) for vote in votes]
        if not self.votes:
            raise ValueError("Domain cannot be empty")
        self.num_candidates = len(self.votes[0])
        if not all(len(vote) == self.num_candidates for vote in self.votes):
            raise ValueError("All votes must have the same number of candidates")

    def __len__(self):
        return len(self.votes)

    def __getitem__(self, idx):
        return self.votes[idx]

    def __iter__(self):
        return iter(self.votes)

    def candidates(self):
        """
        Return the set of candidate indices in the domain.
        """
        return set(self.votes[0])

    def as_list(self):
        """
        Return the votes as a list of tuples.
        """
        return list(self.votes)

    def contains(self, vote):
        """
        Check if a vote is in the domain.
        """
        return tuple(vote) in self.votes


class SC_Domain(Domain):
    """
    Represents a single-crossing domain, inheriting from Domain.
    Precomputes edge swaps and position maps specific to single-crossing domains.
    """
    def __init__(self, votes):
        super().__init__(votes)
        self.edge_swaps = self.compute_edge_swaps()
        self.potes = self.compute_potes()

    def compute_edge_swaps(self):
        edge_swaps = []
        for i in range(len(self.votes) - 1):
            current_vote = self.votes[i]
            next_vote = self.votes[i + 1]
            swap_pair = None
            for j in range(self.num_candidates - 1):
                if current_vote[j] != next_vote[j]:
                    swap_pair = (current_vote[j], current_vote[j + 1])
                    break
            edge_swaps.append(swap_pair)
        return edge_swaps

    def compute_potes(self):
        """ Convert votes to positional votes (called potes) """
        return np.array([[list(vote).index(i) for i, _ in enumerate(vote)]
                                       for vote in self.votes])