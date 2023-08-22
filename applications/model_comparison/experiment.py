import bayesflow as bf
import tensorflow as tf

from configurations import default_settings


class ModelComparisonExperiment():
    """Wrapper for model comparison of different non-stationary DDMs."""

    def __init__(self, config=default_settings):
        """Creates an instance of the model with given configuration.

        Parameters:
        -----------
        config  : dict, optional, default: ``configuration.default_settings``
            A configuration dictionary with the following keys:
            ``lstm1_hidden_units`` - The dimensions of the first LSTM of the first summary net
            ``lstm2_hidden_units`` - The dimensions of the second LSTM of the first summary net
            ``lstm3_hidden_units`` - The dimensions of the third LSTM of the second summary net
            ``trainer``            - The settings for the ``bf.trainers.Trainer``, not icnluding
                the ``amortizer``, ``generative_model``, and ``configurator`` keys,
                as these will be provided internaly by the Experiment instance
        """

        self.summary_network = bf.networks.TimeSeriesTransformer(
            input_dim=1,
            template_dim=128,
            summary_dim=128
            )
        self.inference_network = bf.networks.PMPNetwork(
            **config.get("inference_network_settings")
            )
        self.amortizer = bf.amortizers.AmortizedModelComparison(
            self.inference_network,
            self.summary_network
            )
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            **config.get("trainer")
        )

    def run(self, training_data, validation_data=None, epochs=25, batch_size=16):
        """Wrapper for offline training

        Parameters:
        -----------
        training_data : dict
            Simulated data from the models to compare for training.
        validation_data: dict
            Simulated data from the models to compare for validation.
        epochs: int, optional, default: 50
            Number of trainig epochs.
        batch_size: int, optional, default: 16
            Number of simulated data sets per batch.
        """

        history = self.trainer.train_offline(
            simulations_dict=training_data,
            validation_sims=validation_data,
            epochs=epochs,
            batch_size=batch_size)
        return history

    def evaluate(self):
        pass
