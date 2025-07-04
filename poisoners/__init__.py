from .plm.neuba_poisoner import NeuBAPoisoner
from .plm.por_poisoner import PORPoisoner
from .plm.uor_poisoner import UORPoisoner
from .downstream.sc_poisoner import SCPoisoner
from .downstream.prompt_sc_poisoner import PromptSCPoisoner


POISONERS_LIST = {
    "neuba": NeuBAPoisoner,
    "por": PORPoisoner,
    "uor": UORPoisoner,
    "sc": SCPoisoner,
    "prompt_sc": PromptSCPoisoner
}


def get_poisoner(config):
    poisoner = POISONERS_LIST[config.method](config)
    return poisoner