from Dassl.dassl.utils import Registry, check_availability
from trainers.DP_FPL import DP_FPL

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(DP_FPL)

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
