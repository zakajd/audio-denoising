import os
import pathlib
import configargparse as argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Mel spectogram classification and denoising",
        default_config_files=["configs/default.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )

    add_arg = parser.add_argument

    # General
    add_arg("--name", type=str, help="Name of this run")
    add_arg("--task", type=str, help="Task to solve. One of {`classification`, `denoising`}")
    add_arg("--seed", type=int, help="Random seed for reprodusability")
    add_arg("--root", type=str, help="Path to train data")
    add_arg("--debug", dest="debug", default=False, action="store_true", help="Make short epochs")
    add_arg("--resume", default="", type=str, help="Path to checkpoint to start from")


    # DATALOADER
    add_arg("--batch_size", type=int, help="Batch size")
    add_arg("--workers", type=int, help="№ of data loading workers ")
    add_arg("--aug_type", default="light", type=str, help="")

    # Model
    add_arg("--segm_arch", default="unet", type=str, help="Segmentation architecture to use")
    add_arg("--arch", default="unet", type=str, help="Architecture to use")
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")
    add_arg("--ema_decay", type=float, default=0, help="Decay for ExponentialMoving Average")

    # Training
    add_arg("--use_fp16", default=False, action="store_true", help="Flag to enable FP16 training")
    add_arg("--optim", type=str, default="adam", help="Optimizer to use (default: adam)")
    add_arg("--optim_params", type=eval, default={}, help="Additional optimizer params as kwargs")
    add_arg("--size", default=64, type=int, help="Size of images to train at")
    add_arg(
        "--phases",
        type=eval,
        action="append",
        help="Specify epoch order of data resize and learning rate schedule",
    )
    add_arg(
        "--metrics",
        default=["acc"],
        type=str,
        nargs="+",
        help="Metrics to compute during training",
    )

    # Criterion
    add_arg("--criterion", type=str, help="Criterion to use.")
    add_arg("--criterion_params", type=eval, default={}, help="Params to pass to criterion")

    # Validation and testing
    add_arg(
        "--tta",
        dest="tta",
        default=False,
        action="store_true",
        help="Flag to use TTA for validation and test sets",
    )
    return parser


def parse_args():
    parser = get_parser()
    args, not_parsed_args = parser.parse_known_args()
    print("Not parsed args: ", not_parsed_args)

    # If folder already exist append version number
    outdir = os.path.join("logs", f"{args.name}")
    if os.path.exists(outdir):
        version = 1
        while os.path.exists(outdir):
            outdir = os.path.join("logs", f"{args.name}-{version}")
            version += 1
    args.outdir = outdir
    return args
