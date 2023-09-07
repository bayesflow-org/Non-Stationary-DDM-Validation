import sys
sys.path.append("../../assets")

from experiments import NonStationaryDDMExperiment
from models import RandomWalkDDM, MixtureRandomWalkDDM, LevyFlightDDM, RegimeSwitchingDDM

def get_setup(model_name, summary_net_type="smoothing", skip_checks=False):
    if summary_net_type == "smoothing":
        net_name = "smoothing_"
    else:
        net_name = ""

    if model_name == "rw":
        model = RandomWalkDDM()
        model_string = "random_walk_ddm"
    if model_name == "mrw":
        model = MixtureRandomWalkDDM()
        model_string = "mixture_random_walk_ddm"
    if model_name == "lf":
        model = LevyFlightDDM()
        model_string = "levy_flight_ddm"
    if model_name == "rs":
        model = RegimeSwitchingDDM()
        model_string = "regime_switching_ddm"

    trainer = NonStationaryDDMExperiment(
        model,
        summary_network_type=summary_net_type,
        checkpoint_path=f"checkpoints/{net_name}{model_string}",
        skip_checks=skip_checks
    )

    return model, trainer
