import os
import time
import shutil
from pathlib import Path
from multiprocessing import pool
import argparse

import cv2
import yaml
import torch
import shapely
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb
import albumentations as albu
from loguru import logger

# Local imports
from src.datasets import get_val_dataloader, get_aug
from src.utils import MODEL_FROM_NAME, LOSS_FROM_NAME, METRIC_FROM_NAME


@torch.no_grad()
def denoise_audio(file, model):
    """
    Args:
        file: Path to noisy melspectogram
        model: PyTorch model used for denoising
    """
    # noisy_audio, framerate = soundfile.read(audio_file)
    # noisy_mel = 1 + np.log(
    #     1.e-12 + librosa.feature.melspectrogram(
    #         noisy_audio, sr=16000, n_fft=1024, hop_length=256, fmin=20, fmax=8000, n_mels=80)
    # ).T / 10.

    # Read file and get melspectogram
    noisy_mel = np.load(file).astype('float32')

    # Prepare for inference
    length = (noisy_mel.shape[0] + 31) // 32 * 32

    # Add fake RGB channels
    noisy_image = np.repeat(noisy_mel[np.newaxis, :, :], 3, axis=0)
    
    # Transform
    noisy_image = get_aug('val', size=length)(image=noisy_image.T)["image"].unsqueeze(0).cuda()

    denoised_mel = model(noisy_image).squeeze().cpu().numpy()
    # Crop to original size
    w_pad_left = int((length - noisy_mel.shape[0]) / 2.0)
    w_pad_right = length - noisy_mel.shape[0] - w_pad_left
    denoised_mel = denoised_mel[8: -8, w_pad_left: -w_pad_right]
    return denoised_mel.T
    

@torch.no_grad()
def classify_audio(file, model):
    """
    Args:
        file: Path to melspectogram
        model: PyTorch model used for classification
    """

    mel = np.load(file).astype('float32')

    # Add fake RGB channels
    image = np.repeat(mel[np.newaxis, :, :], 3, axis=0)

    # Transform
    image = get_aug('val', size=512)(image=image.T)["image"].unsqueeze(0).cuda()
    logit = model(image).squeeze().cpu()
    print(logit, logit.sigmoid())
    label = model(image).squeeze().cpu().sigmoid().numpy() > 0.5  # Use fixed threshold
    return label.astype('int8')


def test(hparams):
    # Add model parameters
    hparams = OmegaConf.merge(
        OmegaConf.create(vars(hparams)),
        OmegaConf.load(os.path.join(hparams.config_path, '.hydra/config.yaml'))
    )

    # Get model
    if hparams.training.task == "classification":
        model = pt.models.__dict__[hparams.training.arch](num_classes=1, **hparams.training.model_params).cuda()
    else:
        model = MODEL_FROM_NAME[hparams.training.segm_arch](hparams.training.arch, **hparams.training.model_params).cuda()

    checkpoint = torch.load(os.path.join(hparams.config_path, "model.chpn"))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda().eval()
    logger.info("Model loaded succesfully.")

    num_params = pt.utils.misc.count_parameters(model)[0]
    logger.info(f"Model size: {num_params / 1e6:.02f}M")

    # Inference one file
    if hparams.file_path:
        if hparams.training.task == 'classification':
            label = classify_audio(hparams.file_path, model)
            logger.info(f'Label: {"noisy" if label else "clean"}')
        elif hparams.training.task == 'denoising':
            denoised_mel = denoise_audio(hparams.file_path, model)

            # Save
            output_path = Path(hparams.output_path)
            logger.info(f'Saved denoised melspectogram to {output_path / Path(hparams.file_path).name}')
            np.save(output_path / Path(hparams.file_path).name, denoised_mel)
    return

    # Inference all files in the folder and compute target metric
    if hparams.data_path:
        hparams.data_path = Path(hparams.data_path)
        # Get loss
        loss = LOSS_FROM_NAME[hparams.criterion].cuda()

        # Get loader
        loader, _ = get_val_dataloader(
            root=hparams.data_path,
            aug_type='val',
            task=hparams.training.task,
            size=512,
            batch_size=1,
            workers=hparams.workers,
        )

        # Init runner
        runner = pt.fit_wrapper.Runner(
            model,
            optimizer=None,
            criterion=loss,
            callbacks=pt_clb.ConsoleLogger(),
            use_fp16=hparams.use_fp16, 
        )

        # Evaluate
        loss, _ = runner.evaluate(loader)
        logger.info(f"{hparams.criterion}: {loss:0.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio classification and denoising",
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to folder with model config and checkpoint")
    parser.add_argument(
        "--folder_path", type=str, default="", help="Path to folder with files")
    parser.add_argument(
        "--file_path", type=str, default="", help="Path to one melspectogram")
    parser.add_argument(
        "--output_path", type=str, default="examples/denoised", help="Where to save result")

    hparams = parser.parse_args()
    print(f"Parameters used for inference: {hparams}")
    start_time = time.time()
    test(hparams)
    print(f"Finished inference. Took: {(time.time() - start_time) / 60:.02f}m")