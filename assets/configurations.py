default_bounds = {
    "lower": (0, 0, 0),
    "upper": (8, 6, 4)
}

default_priors = {
    # v, a, tau
    "ddm_loc": (2.0, 0.1, 0.3),
    "ddm_scale": (2.0, 1.0, 1.0),
    "scale_loc": 0.0,
    "scale_scale": 0.1,
    "q_lower": 0.0,
    "q_upper": (0.2, 0.1),
    "alpha_a": (1.5, 2.5),
    "alpha_b": 1.5
}