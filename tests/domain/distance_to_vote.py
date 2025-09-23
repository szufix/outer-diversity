import pytest
from src.diversity.sampling import (
    outer_diversity_sampling_for_structered_domains,
    outer_diversity_sampling
)

from src.domain import *
from time import time
import random



class TestDistanceToVote:

    def test_single_peaked(self):

        for i in range(10):

            num_candidates = random.randint(3, 8)
            num_samples = random.randint(5, 10)

            domain = single_peaked_domain(num_candidates)

            x = outer_diversity_sampling(domain, num_samples)

            y = outer_diversity_sampling_for_structered_domains(
                'sp', None, num_candidates, num_samples)

            assert x == y

    def test_spoc(self):

        for i in range(5):
            num_candidates = random.randint(3, 6)
            num_samples = random.randint(5, 10)

            domain = spoc_domain(num_candidates)

            x = outer_diversity_sampling(domain, num_samples)

            y = outer_diversity_sampling_for_structered_domains(
                'spoc', None, num_candidates, num_samples)

            assert x == y

    def test_single_crossing(self):

        for i in range(5):
            num_candidates = random.randint(3, 6)
            num_samples = random.randint(5, 10)

            domain = single_crossing_domain(num_candidates)

            x = outer_diversity_sampling(domain, num_samples)

            y = outer_diversity_sampling_for_structered_domains(
                'sc', domain, num_candidates, num_samples)

            assert x == y

