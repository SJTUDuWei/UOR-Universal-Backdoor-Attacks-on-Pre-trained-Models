import logging
from .plm import PLMVictim
from .mlm import MLMVictim
from .sc import SCVictim


VICTIM_LIST = {
    'plm': PLMVictim,           # Model
    'mlm': MLMVictim,           # MaskedLM
    'sc': SCVictim,             # SequenceClassification
}


def get_victim(config):
    logging.info("\n> Loading {} from {} <\n".format(config.type, config.path))
    victim = VICTIM_LIST[config.type](config)
    return victim