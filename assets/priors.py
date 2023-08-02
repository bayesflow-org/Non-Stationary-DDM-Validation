import numpy as np
from scipy.stats import halfnorm, truncnorm, levy_stable
from configurations import default_priors, default_bounds

def sample_ddm_params(loc=default_priors["ddm_loc"],
                      scale=default_priors["ddm_scale"],
                      a=default_bounds["lower"],
                      b=default_bounds["upper"]):
    """Generates random draws from a truncated-normal prior over the
    diffusion decision parameters, v, a, tau.

    Parameters:
    -----------
    loc   : float, optional, default: ``configuration.default_priors["ddm_loc"]``
        The location parameter of the truncated-normal distribution.
    scale : float, optional, default: ``configuration.default_priors["ddm_scale"]``
        The scale parameter of the truncated-normal distribution.
    a     : float, optional, default: ``configuration.default_bounds["lower"]``
        The lower boundary of the truncated-normal distribution.
    b     : float, optional, default: ``configuration.default_bounds["upper"]``
        The upper boundary of the truncated-normal distribution.
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.
    Returns:
    --------
    prior_draws : np.array
        The randomly drawn DDM parameters, v, a, tau.
    """
    return truncnorm.rvs(
        a=(a - loc) / scale,
        b=(b - loc) / scale,
        loc=loc,
        scale=scale
        )

def sample_random_walk_hyper(loc=default_priors["scale_loc"],
                             scale=default_priors["scale_scale"]):
    """Generates random draws from a half-normal prior over the
    scale of the random walk.

    Parameters:
    -----------
    loc   : float, optional, default: ``configuration.default_priors["scale_loc"]``
        The location parameter of the half-normal distribution.
    scale : float, optional, default: ``configuration.default_priors["scale_scale"]``
        The scale parameter of the half-normal distribution.
    Returns:
    --------
    prior_draws : np.array
        The randomly drawn scale parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale, size=3)

def sample_mixture_random_walk_hyper(loc=default_priors["scale_loc"],
                                     scale=default_priors["scale_scale"],
                                     low=default_priors["q_low"],
                                     high=default_priors["q_high"],
                                     rng=None):
    """Generates random draws from a half-normal prior over the scale and
    random draws from a uniform prior over the swiching probabilty q of
    the mixture random walk.

    Parameters:
    -----------
    loc   : float, optional, default: ``configuration.default_priors["scale_loc"]``
        The location parameter of the half-normal distribution.
    scale : float, optional, default: ``configuration.default_priors["scale_scale"]``
        The scale parameter of the half-normal distribution.
    low   : float, optional, default: ``configuration.default_priors["q_low"]``
        The low parameter of the uniform distribution.
    high  : float, optional, default: ``configuration.default_priors["q_high"]``
        The high parameter of the uniform distribution.
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.
    Returns:
    --------
    prior_draws : np.array
        The randomly drawn scale and switching probability parameters.
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    scales = halfnorm.rvs(loc=loc, scale=scale, size=3)
    q = rng.uniform(low=low, high=high)
    return np.concatenate([scales, q])

def sample_levy_flight_hyper(loc=default_priors["scale_loc"],
                             scale=default_priors["scale_scale"],
                             a=default_priors["alpha_a"],
                             b=default_priors["aplha_b"],
                             rng=None):
    """Generates random draws from a half-normal prior over the scale and
    random draws from a beta prior over the alpha parameter of the levy flight.

    Parameters:
    -----------
    loc   : float, optional, default: ``configuration.default_priors["scale_loc"]``
        The location parameter of the half-normal distribution.
    scale : float, optional, default: ``configuration.default_priors["scale_scale"]``
        The scale parameter of the half-normal distribution.
    a     : float, optional, default: ``configuration.default_priors["alpha_a"]``
        The a parameter of the beta distribution.
    b     : float, optional, default: ``configuration.default_priors["alpha_b"]``
        The b parameter of the beta distribution.
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.
    Returns:
    --------
    prior_draws : np.array
        The randomly drawn scale and alpha parameters.
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    scales = halfnorm.rvs(loc=loc, scale=scale, size=3)
    alphas = rng.beta(a=a, b=b) + 1.0
    return np.concatenate([scales, alphas])

def sample_regime_switching_hyper(low=default_priors["q_low"],
                                  high=default_priors["q_high"],
                                  rng=None):
    """Generates random draws from a half-normal prior over the scale and
    random draws from a uniform prior over the swiching probabilty q of
    the mixture random walk.

    Parameters:
    -----------
    low  : float, optional, default: ``configuration.default_priors["q_low"]``
        The low parameter of the uniform distribution.
    high : float, optional, default: ``configuration.default_priors["q_high"]``
        The high parameter of the uniform distribution.
    rng  : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    prior_draws : np.array
        The randomly drawn switching probability parameters.
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=low, high=high)

def sample_random_walk(hyper_params,
                       num_steps=800,
                       lower_bounds=default_bounds["lower"],
                       upper_bounds=default_bounds["upper"],
                       rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    hyper_params : np.array
        The scales of the random walk transition.
    num_steps    : int, optional, default: 800
        The number of time steps to take for the random walk. Default corresponds
        to the maximal number of trials in the color discrimination data set.
    lower_bounds : tuple, optional, default: ``configuration.default_bounds["lower"]``
        The minimum values the parameters can take.
    upper_bounds : tuple, optional, default: ``configuration.default_bounds["upper"]``
        The maximum values the parameters can take.
    rng          : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()

    # Run random walk from initial
    z = rng.normal(size=(num_steps - 1, 3))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + hyper_params * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t

def sample_mixture_random_walk(hyper_params,
                               num_steps=800,
                               lower_bounds=default_bounds["lower"],
                               upper_bounds=default_bounds["upper"],
                               rng=None):
    """Generates a single simulation from a mixture random walk transition model.

    Parameters:
    -----------
    hyper_params : np.array
        The scales and switching probabilities of the mixture random walk transition.
    num_steps    : int, optional, default: 800
        The number of time steps to take for the random walk. Default corresponds
        to the maximal number of trials in the color discrimination data set.
    lower_bounds : tuple, optional, default: ``configuration.default_bounds["lower"]``
        The minimum values the parameters can take.
    upper_bound  : tuple, optional, default: ``configuration.default_bounds["upper"]``
        The maximum values the parameters can take.
    rng          : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()
    # randomness
    z = rng.normal(size=(num_steps - 1, 3))
    switch_probability = rng.randn(size=(num_steps - 1, 2))
    switch = switch_probability > hyper_params[3:]
    # transition model
    for t in range(1, num_steps):
        # update v
        if switch[t-1, 0]:
            theta_t[t, 0] = np.clip(
                theta_t[t-1, 0] + hyper_params[0] * z[t-1, 0],
                a_min=lower_bounds[0], a_max=upper_bounds[0]
                )
        else:
            theta_t[t, 0] = rng.uniform(lower_bounds[0], upper_bounds[0])
        # update a
        if switch[t-1, 1]:
            theta_t[t, 1] = np.clip(
                theta_t[t-1, 1] + hyper_params[1] * z[t-1, 1],
                a_min=lower_bounds[1], a_max=upper_bounds[1]
                )
        else:
            theta_t[t, 1] = rng.uniform(lower_bounds[1], upper_bounds[1]) 
        # update tau
        theta_t[t, 2] = np.clip(
            theta_t[t-1, 2] + hyper_params[2] * z[t-1, 2],
            a_min=lower_bounds[2], a_max=upper_bounds[2]
            )
    return theta_t

def sample_levy_flight(hyper_params,
                       num_steps=800,
                       lower_bounds=default_bounds["lower"],
                       upper_bounds=default_bounds["upper"],
                       rng=None):
    """Generates a single simulation from a levy flight transition model.

    Parameters:
    -----------
    hyper_params : np.array
        The scales and alpha parameter of the levy flight transition.
    num_steps    : int, optional, default: 800
        The number of time steps to take for the random walk. Default corresponds
        to the maximal number of trials in the color discrimination data set.
    lower_bounds : tuple, optional, default: ``configuration.default_bounds["lower"]``
        The minimum values the parameters can take.
    upper_bound  : tuple, optional, default: ``configuration.default_bounds["upper"]``
        The maximum values the parameters can take.
    rng          : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()
    # randomness
    levy_scale = hyper_params[:2] / np.sqrt(2)
    z_norm = rng.normal(size=(num_steps - 1))
    z_levy = levy_stable.rvs(hyper_params[3:], 0, scale=levy_scale, size=(num_steps - 1, 2))
    # transition model
    for t in range(1, num_steps):
        # update v and a
        theta_t[t, :2] = np.clip(
            theta_t[t-1, :2] + z_levy[t-1],
            a_min=lower_bounds[:2], a_max=upper_bounds[:2]
            )
        # update tau
        theta_t[t, 2] = np.clip(
            theta_t[t-1, 2] + hyper_params[2] * z_norm[t-1],
            a_min=lower_bounds[2], a_max=upper_bounds[2]
            )
    return theta_t

def sample_regime_switching(hyper_params,
                            num_steps=800,
                            lower_bounds=default_bounds["lower"],
                            upper_bounds=default_bounds["upper"],
                            rng=None):
    """Generates a single simulation from a levy flight transition model.

    Parameters:
    -----------
    hyper_params : np.array
        The scales and alpha parameter of the levy flight transition.
    num_steps    : int, optional, default: 800
        The number of time steps to take for the random walk. Default corresponds
        to the maximal number of trials in the color discrimination data set.
    lower_bounds : tuple, optional, default: ``configuration.default_bounds["lower"]``
        The minimum values the parameters can take.
    upper_bound  : tuple, optional, default: ``configuration.default_bounds["upper"]``
        The maximum values the parameters can take.
    rng          : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()
    # randomness
    z = rng.normal(size=(num_steps - 1, 3))
    switch_probability = rng.randn(size=(num_steps - 1, 2))
    switch = switch_probability > hyper_params
    # transition model
    for t in range(1, num_steps):
        # update v
        if switch[t-1, 0]:
            theta_t[t, 0] = theta_t[t-1, 0]
        else:
            theta_t[t, 0] = rng.uniform(lower_bounds[0], upper_bounds[0])
        # update a
        if switch[t-1, 1]:
            theta_t[t, 1] = theta_t[t, 1]
        else:
            theta_t[t, 1] = rng.uniform(lower_bounds[1], upper_bounds[1]) 
        # update tau
        theta_t[t, 2] = theta_t[t, 2]
    return theta_t