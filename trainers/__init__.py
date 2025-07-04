from .plm.neuba_trainer import NeuBATrainer
from .plm.por_trainer import PORTrainer
from .plm.uor_trainer import UORTrainer
from .plm.uor_all_trainer import UORALLTrainer
from .plm.uor_tc_trainer import UORTCTrainer
from .downstream.finetune_sc_trainer import FineTuneSCTrainer
from .downstream.finetune_tc_trainer import FineTuneTCTrainer
from .downstream.finetune_mc_trainer import FineTuneMCTrainer
from .downstream.prompt_sc_trainer import PromptSCTrainer
from .utils.visualizer import visualize_plm, visualize_dsm


TRAINERS_LIST = {
    "neuba": NeuBATrainer,
    "por": PORTrainer,
    "uor": UORTrainer,
    "uor_all": UORALLTrainer,
    "uor_tc": UORTCTrainer,
    "finetune_sc": FineTuneSCTrainer,
    "finetune_tc": FineTuneTCTrainer,
    "finetune_mc": FineTuneMCTrainer,
    "prompt": PromptSCTrainer
}


def get_trainer(config, save_dir):
    trainer = TRAINERS_LIST[config.method](config, save_dir)
    return trainer