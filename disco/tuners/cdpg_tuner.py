# disco
# Copyright (C) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .tuner import Tuner
from disco.distributions.single_context_distribution import SingleContextDistribution
from disco.tuners.losses import *


class CDPGTuner(Tuner):
    """Contextual DPG tuning class,
    relying on a ContextDistribution and KLLoss().

    The algorithm has been introduced in
    "Controlling Conditional Language Models without Catastrophic Forgetting"
    Tomasz Korbak, Hady Elsahar, Germán Kruszewski and Marc Dymetman.
    https://proceedings.mlr.press/v162/korbak22a/korbak22a.pdf
    """

    def __init__(self, *args, context_distribution=SingleContextDistribution(),
            loss=KLLoss(), **kwargs):
        """
        Parameters
        ----------
        context_distribution: distribution
            a distribution to contextualize the sampling from the proposal
        """

        super(CDPGTuner, self).__init__(
                *args,
                context_distribution=context_distribution,
                loss=loss,
                **kwargs
            )
