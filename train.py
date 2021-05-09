import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dataset import MRCDataModule
from model import MRCModel
from utils.args import get_args
from utils.logging import prepare_logger


def train(args):

    datamodule = MRCDataModule(args)
    model = MRCModel(args, datamodule)

    trainer = pl.Trainer(logger=pl_loggers.TensorBoardLogger(save_dir=args.save_path, name=None, prefix=''),
                         accumulate_grad_batches=args.config['solver']['accumulate_grad_batches'],
                         gpus=args.gpu,
                         tpu_cores=args.tpu,
                         max_epochs=args.config['solver']['num_epochs'],
                         gradient_clip_val=args.config['solver']['gradient_clip_val'],
                         # deterministic=True,
                         # profiler='simple',
                         precision=args.config['solver']['precision'],
                         val_check_interval=args.config['solver']['val_check_interval'] \
                            if not args.config['solver']['adversarial_training'] in ['fgm', 'pgd'] else 0,
                         num_sanity_val_steps=0,
                         #checkpoint_callback=False,
                         # fast_dev_run=args.overfit,
                         # limit_train_batches=2,
                         # limit_val_batches=2,
                         # limit_test_batches=2
                         )
    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':

    args = get_args()
    args.save_path = os.path.join(os.getcwd(), args.config['solver']['output_path'], args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    args.logger = prepare_logger(args.save_path)
    args.logger.info(yaml.dump(args.config, default_flow_style=False))
    pl.seed_everything(args.config['solver']['seed'])
    train(args)

