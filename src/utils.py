import piq
import torch
import pytorch_tools as pt
import segmentation_models_pytorch as sm
from loguru import logger
import torchvision


LOSS_FROM_NAME = {
    "bce": pt.losses.CrossEntropyLoss(mode="binary"),
    "wbce": pt.losses.CrossEntropyLoss(mode="binary", weight=[5]),
    "dice": pt.losses.DiceLoss(mode="binary"),
    "jaccard": pt.losses.JaccardLoss(mode="binary"),
    "log_jaccard": pt.losses.JaccardLoss(mode="binary", log_loss=True),
    "hinge": pt.losses.BinaryHinge(),
    "whinge": pt.losses.BinaryHinge(pos_weight=3),
    "focal": pt.losses.FocalLoss(mode='binary'),
    "reduced_focal": pt.losses.FocalLoss(mode='binary', combine_thr=0.5, gamma=2.0),
    "mse": pt.losses.MSELoss(),
    "mae": pt.losses.L1Loss(),
    "huber": pt.losses.SmoothL1Loss(),
}

class SmoothL1Loss(pt.losses.SmoothL1Loss):
    name = "huber"

class L1Loss(pt.losses.SmoothL1Loss):
    name = "L1"

class MSELoss(pt.losses.MSELoss):
    name = "L2"

METRIC_FROM_NAME = {
    "acc": pt.metrics.Accuracy(topk=1),
    "mae": L1Loss(),
    "huber": SmoothL1Loss(),
    "mse": MSELoss(),
    # "ms_gmsdc": piq.muli
}


MODEL_FROM_NAME = {
    "unet": pt.segmentation_models.Unet,
    "linknet": pt.segmentation_models.Linknet,
    "deeplab": pt.segmentation_models.DeepLabV3,
    "unet_sm": sm.Unet,
    "linknet_sm": sm.Linknet,
    "fpn_sm": sm.FPN,
    "deeplab_sm": sm.DeepLabV3,
    "segm_fpn": pt.segmentation_models.SegmentationFPN,
    "segm_bifpn": pt.segmentation_models.SegmentationBiFPN,
}


class ToCudaLoader:
    "Simple wrapper to put dataset elements on GPU"
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return ([img.cuda(non_blocking=True), target.cuda(non_blocking=True)] for img, target in self.loader)

    def __len__(self):
        return len(self.loader)


class TensorBoard(pt.fit_wrapper.callbacks.TensorBoard):
    """Saves first batch and visualizes model predictions on it for every epoch"""

    def __init__(self, log_dir, log_every=40, num_images=4):
        """num_images (int): number of images to visualize"""
        super().__init__(log_dir, log_every=log_every)
        self.has_saved = False  # Flag to save first batch
        self.noisy_batch = None
        self.num_images = num_images

    def on_batch_end(self):
        super().on_batch_end()
        # save first val batch
        if not self.has_saved and not self.state.is_train:
            # self.state.logger.info(f"Input shape {self.state.input[0].shape}, {self.state.input[1].shape}")
            self.noisy_batch = self.state.input[0].detach()[:self.num_images]
            self.has_saved = True
            self.clean_batch = self.state.input[1].detach()[:self.num_images]

            self.state.logger.info(f"Clean images mean {self.clean_batch.mean()}, min {self.clean_batch.min()}, max {self.clean_batch.max()}")
            self.clean_grid = torchvision.utils.make_grid(self.clean_batch, nrow=2, normalize=True, scale_each=True)
            # self.state.logger.info(f"Clean batch {self.clean_batch.shape}, clean grid {self.clean_grid.shape}")
            logger.info(f"Clean images grid mean {self.clean_grid.mean()}")

    def on_epoch_end(self):
        super().on_epoch_end()
        self.state.model.eval()
        denoised_batch = self.state.model(self.noisy_batch)
        denoised_grid = torchvision.utils.make_grid(denoised_batch, nrow=2, normalize=True, scale_each=True)
        self.state.logger.info(f"Denoised images mean {denoised_batch.mean()}, min {denoised_batch.min()}, max {denoised_batch.max()}")

        error_map = torch.abs(denoised_batch - self.clean_batch)
        erro_grid = torchvision.utils.make_grid(error_map, nrow=2, normalize=True, scale_each=True)
        self.state.logger.info(f"Error mean {error_map.mean()}, min {error_map.min()}, max {error_map.max()}")    
        # self.state.logger.info(f"Clean grid shape {self.clean_grid.shape}, \
        #     denoised {denoised_grid.shape}, error {erro_grid.shape}")
        self.writer.add_image("val/clean", self.clean_grid , self.current_step)
        self.writer.add_image("val/denoised", denoised_grid, self.current_step)
        self.writer.add_image("val/error", erro_grid, self.current_step)