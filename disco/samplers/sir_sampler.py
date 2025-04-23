import torch

from . import Sampler
from .accumulation_sampler import AccumulationSampler
from disco.metrics import KL
from disco.utils.device import get_device
from disco.utils.helpers import batchify

class SIRSampler(Sampler):
    """
    Sampling Inportance ReSampling class by minbeom
    """

    def __init__(self, target, proposal):
        """
        Parameters
        ----------
        proposal: distribution
            distribution to generate the samples
        constraints: BooleanScorer
            Rejection
        """

        super(SIRSampler, self).__init__(target, proposal)
        self.n_samples = 0
        self.n_accepted_samples = 0

    def sample(self, sampling_size=32, context=''):
        """Generates samples according to the SIR algorithm

        Parameters
        ----------
        sampling_size: int
            number of requested samples when sampling
        context: text
            contextual text for which to sample

        Returns
        -------
        tuple of accepted samples
        """

        samples, proposal_log_scores = self.proposal.sample(sampling_size=sampling_size, context=context)
        self.n_samples += len(samples)
        # print(samples)

        device = get_device(proposal_log_scores)

        target_log_scores = self.target.log_score(samples=samples, context=context).to(device)
        
        importance = torch.exp(target_log_scores - proposal_log_scores)
        # importance = torch.nn.functional.normalize(importance, dim=0)
        
#         print(proposal_log_scores)
#         print(target_log_scores)
#         print(importance)
        
        us = torch.rand(len(importance)).to(device)
        accepted_samples = [x for k, x in zip(us < importance, samples) if k]
        self.n_accepted_samples += len(accepted_samples)
        print('AR = ', len(accepted_samples)/self.n_samples)

        return accepted_samples
