from abc import ABC, abstractmethod
import bayesflow as bf
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Sequential

from configurations import default_settings

class Experiment(ABC):
    """An interface for a standardized experiment."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass


class NonStationaryDDMExperiment(Experiment):
    """Wrapper for estimating a non-stationary diffusion decision model
    with the neural superstatistics method."""

    def __init__(self, model, summary_network_type="smoothing", checkpoint_path=None, skip_checks=False, config=default_settings):
        """Creates an instance of the model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.

        Parameters:
        -----------
        model                : an instance of models.RandomWalkDiffusion
            The model wrapper, should include a callable attribute ``generator`` and a method
            ``configure()``.
        summary_network_type : string, optional, default: "smoothing"
            The type of the summary network. Either bidirectional ("smoothing") or
            unidirectional ("filtering") LSTM.
        checkpoint_path      : string or None, optional, default: None
            Optional file path for storing the trained amortizer, loss history and optional memory.
        config               : dict, optional, default: ``configuration.default_settings``
            A configuration dictionary with the following keys:
            ``lstm1_hidden_units`` - The dimensions of the first LSTM of the first summary net
            ``lstm2_hidden_units`` - The dimensions of the second LSTM of the first summary net
            ``lstm3_hidden_units`` - The dimensions of the third LSTM of the second summary net
            ``trainer``            - The settings for the ``bf.trainers.Trainer``, not icnluding   
                                     the ``amortizer``, ``generative_model``, and ``configurator`` keys,
                                     as these will be provided internaly by the Experiment instance
        """

        self.model = model

        if summary_network_type == "smoothing":
            self.summary_network = bf.networks.HierarchicalNetwork([
                Sequential([
                    Bidirectional(LSTM(config["lstm1_hidden_units"], return_sequences=True)),
                    Bidirectional(LSTM(config["lstm2_hidden_units"], return_sequences=True)),
                ]),
                Sequential([Bidirectional(LSTM(config["lstm3_hidden_units"]))])
            ])
        if summary_network_type == "filtering":
            self.summary_network = bf.networks.HierarchicalNetwork([
                Sequential([
                    LSTM(config["lstm1_hidden_units"], return_sequences=True),
                    LSTM(config["lstm2_hidden_units"], return_sequences=True),
                ]),
                Sequential([LSTM(config["lstm3_hidden_units"])])
            ])

        self.local_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(
                num_params=3,
                **config.get("local_amortizer_settings")
            ))

        self.global_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(
                num_params=model.hyper_prior_means.shape[0],
                **config.get("global_amortizer_settings")
            ))

        self.amortizer = bf.amortizers.TwoLevelAmortizedPosterior(
            self.local_net,
            self.global_net,
            self.summary_network
            )

        # Trainer setup
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            checkpoint_path=checkpoint_path,
            skip_checks=skip_checks,
            **config.get("trainer")
        )

    def run(self, epochs=75, iterations_per_epoch=1000, batch_size=16):
        """Wrapper for online training

        Parameters:
        -----------
        epochs: int, optional, default: 75
            Number of trainig epochs.
        iterations_per_epoch, int, optional, default: 1000
            Number of iterations per epoch.
        batch_size: int, optional, default: 16
            Number of simulated data sets per batch.

        Returns:
        --------
        history : dict
            A dictionary with the training history/
        """

        history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size)
        return history