import torch
import numpy as np
import shutil
import torchvision.transforms.functional as TVF
import torch.utils.tensorboard as tb

from torchsummary import summary
from torchvision.ops import sigmoid_focal_loss
from functools import partial
from pathlib import Path
from .model_det2 import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms


DEFAULT_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class Trainer:

    _stdout_every = 10

    def __init__(self, model, epochs=10, lr=0.001, train_logger=None, val_logger=None, save_dir="./models", device=DEFAULT_DEVICE) -> None:
        self.device = device
        self.save_dir = save_dir
        self.n_epoch = epochs
        self.model = model

        # Learning functions
        self.loss_fn = partial(sigmoid_focal_loss, reduction="sum")

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=20)

        # Loggers
        self.train_logger = train_logger
        self.val_logger = val_logger

        # Reset loop parameters
        self._init_loop()

    def _init_loop(self):
        self.epoch = 0
        self.global_step = 0
        self.val_loss_min = np.Inf
        self._base_log = f"EPOCH {self.epoch:02d}: "

    def run(self, train_data, val_data):
        self.model.to(self.device)
        print(f"Using device {self.device} to train")

        for epoch in range(1, self.n_epoch+1):
            self.epoch = epoch
            self._base_log = f"EPOCH {epoch:02d}: "

            print(f"{self._base_log} Training...")
            self._train(train_data)

            print(f"{self._base_log} Evaluating...")
            self._eval(val_data)

    def _train(self, train_data):
        self.model.train()

        for i, (x, y) in enumerate(train_data):
            self.global_step += 1
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model.forward(x)
            loss = self.loss_fn(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging and Metrics
            if i % self._stdout_every == 0:
                print(f"{self._base_log} Train loss: {loss:.6f}")

            if self.train_logger:
                self.train_logger.add_scalar(
                    "loss", loss, self.global_step)

    def _eval(self, val_data):
        self.model.eval()

        with torch.no_grad():
            losses = []
            viz = {}

            for i, (x, y) in enumerate(val_data):
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model.forward(x)
                loss = self.loss_fn(y_hat, y)
                losses.append(loss)

                if not viz:
                    viz["x"] = x
                    viz["y"] = y
                    viz["logits"] = y_hat

            # Log data
            val_loss = torch.mean(torch.stack(losses))
            self.val_logger.add_scalar('loss', val_loss, self.global_step)

            log(self.val_logger, viz["x"], viz["y"],
                viz["logits"], self.global_step)

        # Step the learning rate schedule
        self.scheduler.step(val_loss)

        # Save model if it performs better on validation set
        if val_loss < self.val_loss_min:
            print(
                f"{self._base_log} Val loss decreased! {self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model.")
            save_model(self.model, outpath=self.save_dir,
                       outname=f"{self.epoch}")
            self.val_loss_min = val_loss
        else:
            print(
                f"{self._base_log} Val loss: {val_loss: .6f} (min = {self.val_loss_min:.6f})")


def train(args):
    """
    Your code here, modify your HW3 code
    """

    model = Detector(n_class=2)
    summary(model, input_size=(args.batch_size, 3, 400, 300))

    # Load model weights if continuing
    if args.continue_from:
        model.load_state_dict(torch.load(str(args.continue_from)))

    # Create loggers for train and validation
    train_logger, valid_logger = None, None
    model_dir = Path(f"./homework/{args.model_name}")

    if args.log_dir is not None:
        model_dir = args.log_dir / args.model_name

        if model_dir.exists():
            shutil.rmtree(model_dir)

        train_logger = tb.SummaryWriter(
            str(model_dir / 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(
            str(model_dir / 'valid'), flush_secs=1)

    # Load data for training
    train_transforms = dense_transforms.Compose([
        # dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(
            brightness=0.6, contrast=0.4, saturation=0.4, hue=0.2),
        dense_transforms.ToTensor(),
        dense_transforms.CenterToHeatmap(),
    ])
    val_transforms = dense_transforms.Compose([
        dense_transforms.ToTensor(),
        dense_transforms.CenterToHeatmap(),
    ])

    train_data = load_detection_data(
        str(args.train), batch_size=args.batch_size, transform=train_transforms)
    val_data = load_detection_data(
        str(args.val), batch_size=args.batch_size, transform=val_transforms)

    # Create and run trainer
    t = Trainer(
        model=model,
        epochs=args.epochs,
        lr=args.learn_rate,
        train_logger=train_logger,
        val_logger=valid_logger,
        save_dir=model_dir / "models",
    )
    t.run(train_data, val_data)


def overlay_heatmap(x, heatmap, alpha_threshold=0.1):
    background = TVF.to_pil_image(x.to('cpu'))  # assuming your image in x

    alpha = torch.sigmoid(torch.unsqueeze(heatmap.sum(dim=0), 0))
    alpha[alpha < alpha_threshold] = 0
    hm_alpha = torch.cat((heatmap, alpha), dim=0)
    foreground = TVF.to_pil_image(hm_alpha.to('cpu')).convert("RGBA")

    background.paste(foreground, (0, 0), foreground)
    overlay_tensor = TVF.to_tensor(background)

    return overlay_tensor


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """

    device = imgs.get_device()

    def pad_to_rgb(x):
        '''Pad so that tensor can be converted to RGB image'''
        blank = torch.zeros_like(x[:, 0, :, :]).unsqueeze(1)
        repeat_along = torch.ones(len(x.size()))
        repeat_along[1] = 3 - gt_det.size(1)
        repeat_along = [int(f) for f in repeat_along.tolist()]
        blank_channels = blank.repeat(*repeat_along)
        return torch.cat((x, blank_channels), dim=1)

    # Pad with blank channels if not enough to be an RGB image
    if gt_det.size(1) < 3:
        gt_det = pad_to_rgb(gt_det)

    # Ground truth
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)

    # Heatmaps
    normalized_detects = torch.sigmoid(det[:16])
    # print(normalized_detects.shape)
    if normalized_detects.size(1) < 3:
        normalized_detects = pad_to_rgb(normalized_detects)
    # print(normalized_detects.shape)

    overlays = torch.stack([overlay_heatmap(x, y)
                            for x, y in zip(imgs[:16], normalized_detects)]).to(device)
    # print(overlays.shape)
    # logger.add_images('pred', normalized_detects, global_step)
    logger.add_images('overlay', overlays, global_step)


if __name__ == '__main__':
    import argparse
    import time
    epoch_time = int(time.time())

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=Path,
                        default=Path(f"./logging"))
    # Put custom arguments here
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3)
    parser.add_argument('-t', '--train', type=Path)
    parser.add_argument('-v', '--val', type=Path)
    parser.add_argument('-c', '--continue_from', type=Path)
    parser.add_argument('--model_name', type=str, default=f"det_{epoch_time}")

    args = parser.parse_args()
    train(args)