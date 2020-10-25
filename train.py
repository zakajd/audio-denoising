import os
import sys
import yaml
import time

import torch
import numpy as np
import pandas as pd
from loguru import logger
import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb

from src.arg_parser import parse_args
from src.datasets import get_dataloaders
from src.utils import METRIC_FROM_NAME, LOSS_FROM_NAME, MODEL_FROM_NAME, TensorBoard

# Make everything slightly faster
torch.backends.cudnn.benchmark = True


def main():
    # Get config for this run
    hparams = parse_args()

    # Setup logger
    config = {
        "handlers": [
            {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm]} - {message}"},
            {"sink": f"{hparams.outdir}/logs.txt", "format": "{time:[MM-DD HH:mm]} - {message}"},
        ],
    }
    logger.configure(**config)
    logger.info(f"Parameters used for training: {hparams}")

    # Fix seeds for reprodusability
    pt.utils.misc.set_random_seed(hparams.seed)

    # Save config
    os.makedirs(hparams.outdir, exist_ok=True)
    yaml.dump(vars(hparams), open(hparams.outdir + "/config.yaml", "w"))

    # Get model
    if hparams.task == "classification":
        model = pt.models.__dict__[hparams.arch](num_classes=1, **hparams.model_params).cuda()
    else:
        model = MODEL_FROM_NAME[hparams.segm_arch](hparams.arch, **hparams.model_params).cuda()
    # logger.info(f"Model used for training: {model}")

    if hparams.resume:
        checkpoint = torch.load(hparams.resume, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    # if hparams.freeze_bn:
    #     freeze_batch_norm(model)

    # Get loss
    loss = LOSS_FROM_NAME[hparams.criterion].cuda()
    logger.info(f"Loss for this run is: {loss}")

    # Get optimizer
    # params = pt.utils.misc.filter_bn_from_wd(model)
    optimizer = pt.optim.optimizer_from_name(hparams.optim)(
        model.parameters(), lr=0, **hparams.optim_params)

    num_params = pt.utils.misc.count_parameters(model)[0]
    logger.info(f"Model size: {num_params / 1e6:.02f}M")
    # logger.info(model)

    # Scheduler is an advanced way of planning experiment
    sheduler = pt.fit_wrapper.callbacks.PhasesScheduler(hparams.phases)

    # Save logs
    # TB_callback = pt_clb.TensorBoard(hparams.outdir, log_every=20)
    TB_callback = TensorBoard(hparams.outdir, log_every=40, num_images=4)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        root=hparams.root,
        aug_type=hparams.aug_type,
        task=hparams.task,
        size=hparams.size,
        batch_size=hparams.batch_size,
        workers=hparams.workers,
    )

    # Get metrics
    metrics = [METRIC_FROM_NAME[metric_name] for metric_name in hparams.metrics]

    logger.info(f"Start training")
    # Init runner
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            pt_clb.BatchMetrics(metrics),
            pt_clb.Timer(),
            pt_clb.ConsoleLogger(),
            pt_clb.FileLogger(),
            TB_callback,
            pt_clb.CheckpointSaver(hparams.outdir, save_name="model.chpn"),
            sheduler,
            # EMA must go after other checkpoints
            pt_clb.ModelEma(model, hparams.ema_decay) if hparams.ema_decay else pt_clb.Callback(),
            # pt_clb.BatchOverfit(),
        ],
        use_fp16=hparams.use_fp16, 
    )

    # Train
    runner.fit(
        train_loader,
        val_loader=val_loader,
        epochs=sheduler.tot_epochs,
        steps_per_epoch=20 if hparams.debug else None,
        val_steps=20 if hparams.debug else None,
    )

    logger.info(f"Loading best model")
    checkpoint = torch.load(os.path.join(hparams.outdir, f"model.chpn"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Evaluate
    val_loss, metrics = runner.evaluate(
        val_loader,
        steps=20 if hparams.debug else None,
    )

    # TODO: print metric name and value
    logger.info(
        f"Val: Loss {val_loss:0.5f}, Acc@1 {metrics[0]:0.5f}")

    # Save params used for training and final metrics into separate TensorBoard file
    # metric_dict = {
    #     "hparam/Acc@1": acc1,
    #     "hparam/Acc@5": acc1,
    # }

    # Convert all lists / dicts to avoid TB error
    # hparams.phases = str(hparams.phases)
    # hparams.model_params = str(hparams.model_params)
    # hparams.criterion_params = str(hparams.criterion_params)

    # with pt.utils.tensorboard.CorrectedSummaryWriter(hparams.outdir) as writer:
    #     writer.add_hparams(hparam_dict=vars(hparams), metric_dict=metric_dict)


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
