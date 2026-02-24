from .base_engine import BaseTrainer
from .common_mil import CommonMIL

def build_engine(args):

    engine = CommonMIL(args)
    trainer = BaseTrainer(engine=engine,args=args)

    if args.task == "survival":
        return trainer.surv_train,trainer.surv_validate
    else:
        return trainer.train,trainer.validate
