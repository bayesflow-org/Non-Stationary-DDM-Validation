default_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 256,
    "lstm3_hidden_units": 128,
    "inference_network_settings": {
        # "dense_args": dict(units=128, activation="relu"),
        "num_models": 4,
        "dropout_prob": 0.1,
    },
    "trainer": {
        "checkpoint_path": "../checkpoints/model_comparison_template_dim_128_dropout_prob_01",
        "max_to_keep": 1,
        "default_lr": 5e-4,
        "memory": False,
    },
}
