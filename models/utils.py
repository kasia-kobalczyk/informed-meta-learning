from torch.distributions import Normal, Categorical
from torch.distributions.independent import Independent


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def IndependentMultinomial(logits):
    """Multinomial distribution with independent trials."""
    if logits.dim() < 1:
        raise ValueError("logits must be at least one-dimensional.")
    return Independent(Categorical(logits=logits), 1)
