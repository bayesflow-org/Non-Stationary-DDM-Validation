default_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 256,
    "lstm3_hidden_units": 128,
    "inference_network_settings": {
        "num_models": 4,
        "dropout_prob": 0.05,
    },
    "trainer": {
        "checkpoint_path": "checkpoints/epochs_25_dropout_prob_005_batchsize_16_ensemble_1",
        "max_to_keep": 1,
        "default_lr": 5e-4,
        "memory": False,
    },
}
