from abc import ABC, abstractmethod
import bayesflow as bf
from functools import partial
from scipy.stats import uniform, beta, halfnorm

from priors import *
from likelihoods import *
from configurations import default_priors

class DiffusionDecisionModel(ABC):
    """An interface for a non-stationary diffusion decision model."""

    @abstractmethod
    def __init__(self):
        pass

    def generate(self, batch_size):
            """Wraps the call function of ``bf.simulation.TwoLevelGenerativeModel``.

            Parameters:
            -----------
            batch_size : int
                The number of simulations to generate per training batch

            Returns:
            --------
            raw_dict   : dict
                The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
            """
            return self.generator(batch_size)
    
    def configure(self, raw_dict, transform=True):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1 to data
        3. Scale the model parameters if tranform=True

        Parameters:
        -----------
        raw_dict  : dict
            A simulation dictionary as returned by ``bayesflow.simulation.TwoLevelGenerativeModel``
        transform : boolean, optional, default: True
            An indicator to standardize the parameter and log-transform the data samples. 

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """
        # Extract relevant simulation data, convert to float32, and add extra dimension
        theta_t = raw_dict.get("local_prior_draws").astype(np.float32)
        scales = raw_dict.get("hyper_prior_draws").astype(np.float32)
        rt = raw_dict.get("sim_data").astype(np.float32)[..., None]
        if transform:
            theta_t_z = (theta_t - self.local_prior_means) / self.local_prior_stds
            scales_z = (scales - self.hyper_prior_mean) / self.hyper_prior_std
            out_dict = dict(
                local_parameters=theta_t_z.astype(np.float32),
                hyper_parameters=scales_z.astype(np.float32),
                summary_conditions=rt,
            )
        else:
            out_dict = dict(
                local_parameters=theta_t,
                hyper_parameters=scales,
                summary_conditions=rt
            )
        return out_dict


class RandomWalkDDM(DiffusionDecisionModel):
    """A wrapper for a non-stationary diffusion decision process with
    a Gaussian random walk transition model."""

    def __init__(self, num_steps=800, rng=None):
        """Creates an instance of the non-stationary diffusion decision model with
        a Gaussian random walk transition model and given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.

        Parameters:
        -----------
        num_steps : int, optional, default: 800
            The number of time steps to take for the random walk. Default corresponds
            to the maximal number of trials in the color discrimination data set.
        rng       : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """
        self.hyper_prior_mean = halfnorm(
            loc=default_priors["scale_loc"],
            scale=default_priors["scale_scale"]).mean()
        self.hyper_prior_std = halfnorm(
            loc=default_priors["scale_loc"],
            scale=default_priors["scale_scale"]).std()
        self.local_prior_means = np.array([2.8, 2.5, 1.3])
        self.local_prior_stds = np.array([1.9, 1.5, 1])
        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=sample_random_walk_hyper,
            local_prior_fun=partial(sample_random_walk, num_steps=num_steps, rng=self._rng),
        )
        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_non_stationary_diffusion_process,
        )
        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="random_walk_ddm",
        )


class MixtureRandomWalkDDM(DiffusionDecisionModel):
    """A wrapper for a non-stationary diffusion decision process with
    a mixture random walk transition model."""

    def __init__(self, num_steps=800, rng=None):
        """Creates an instance of the non-stationary diffusion decision model with
        a mixture random walk transition model and given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.

        Parameters:
        -----------
        num_steps : int, optional, default: 800
            The number of time steps to take for the mixture random walk. Default corresponds
            to the maximal number of trials in the color discrimination data set.
        rng       : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """
        self.hyper_prior_mean = np.concatenate([
            halfnorm(
                loc=default_priors["scale_loc"],
                scale=default_priors["scale_scale"]
                ).mean(),
            uniform(
                loc=default_priors["q_low"],
                scale=default_priors["q_high"],
            ).mean()
            ])
        self.hyper_prior_std = np.concatenate([
            halfnorm(
                loc=default_priors["scale_loc"],
                scale=default_priors["scale_scale"]
                ).std(),
            uniform(
                loc=default_priors["q_low"],
                scale=default_priors["q_high"],
            ).std()
            ])
        self.local_prior_means = np.array([2.8 , 2.5, 1.3])
        self.local_prior_stds = np.array([1.9, 1.5, 1])
        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=sample_mixture_random_walk_hyper,
            local_prior_fun=partial(sample_mixture_random_walk, num_steps=num_steps, rng=self._rng),
        )
        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_non_stationary_diffusion_process,
        )
        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="mixture_random_walk_ddm",
        )


class LevyFlightDDM(DiffusionDecisionModel):
    """A wrapper for a non-stationary diffusion decision process with
    a levy flight transition model."""

    def __init__(self, num_steps=800, rng=None):
        """Creates an instance of the non-stationary diffusion decision model with
        a levy flight transition model and given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.

        Parameters:
        -----------
        num_steps : int, optional, default: 800
            The number of time steps to take for the mixture random walk. Default corresponds
            to the maximal number of trials in the color discrimination data set.
        rng       : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """
        self.hyper_prior_mean = np.concatenate([
            halfnorm(
                loc=default_priors["scale_loc"],
                scale=default_priors["scale_scale"]
                ).mean(),
            beta(
                a=default_priors["alpha_a"],
                b=default_priors["alpha_b"],
            ).mean()
            ])
        self.hyper_prior_std = np.concatenate([
            halfnorm(
                loc=default_priors["scale_loc"],
                scale=default_priors["scale_scale"]
                ).std(),
            beta(
                a=default_priors["alpha_a"],
                b=default_priors["alpha_b"],
            ).std()
            ])
        self.local_prior_means = np.array([3.2, 2.6, 1.3])
        self.local_prior_stds = np.array([2.3, 1.6, 1])
        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=sample_levy_flight_hyper,
            local_prior_fun=partial(sample_levy_flight, num_steps=num_steps, rng=self._rng),
        )
        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_non_stationary_diffusion_process,
        )
        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="levy_flight_ddm",
        )


class RegimeSwitchingDDM(DiffusionDecisionModel):
    """A wrapper for a non-stationary diffusion decision process with
    a regime switching transition model."""

    def __init__(self, num_steps=800, rng=None):
        """Creates an instance of the non-stationary diffusion decision model with
        a regime switching transition model and given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.

        Parameters:
        -----------
        num_steps : int, optional, default: 800
            The number of time steps to take for the mixture random walk. Default corresponds
            to the maximal number of trials in the color discrimination data set.
        rng       : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """
        self.hyper_prior_mean = uniform(
                loc=default_priors["q_low"],
                scale=default_priors["q_high"],
            ).mean()
        self.hyper_prior_std = uniform(
                loc=default_priors["q_low"],
                scale=default_priors["q_high"],
            ).std()
        self.local_prior_means = np.array([3.2, 2.6, 1.3])
        self.local_prior_stds = np.array([2.3, 1.6, 1])
        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=sample_regime_switching_hyper,
            local_prior_fun=partial(sample_regime_switching, num_steps=num_steps, rng=self._rng),
        )
        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_non_stationary_diffusion_process,
        )
        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="regime_switching_ddm",
        )
