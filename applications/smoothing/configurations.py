default_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 256,
    "lstm3_hidden_units": 128,
    "local_amortizer_settings": {
        "num_coupling_layers": 8,
        "coupling_design": 'interleaved'
    },
    "global_amortizer_settings": {
        "num_coupling_layers": 6,
        "coupling_design": 'interleaved'
    },
    "trainer": {
        "max_to_keep": 1,
        "default_lr": 5e-4,
        "memory": False,
    },
}
default_bounds = {
    "lower": (0.0, 0.2, 0.0),
    "upper": (8.0, 6.0, 4.0)
}
default_priors = {
    # v, a, tau
    "ddm_loc": (2.0, 2.0, 0.3),
    "ddm_scale": (2.0, 1.5, 1.0),
    "scale_loc": (0.0, 0.0, 0.0),
    "scale_scale": (0.1, 0.1, 0.01),
    # v, a
    "q_low": (0.0, 0.0),
    "q_high": (0.2, 0.1),
    "alpha_a": (1.5, 2.5),
    "alpha_b": (1.5, 1.5)
}

model_names = ("rw", "mrw", "lf", "rs")
