import argparse
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import torch.utils.tensorboard as tb

from torchinfo import summary
from pathlib import Path

from . import dense_transforms
from .utils import load_detection_data
from .model_det import Detector

DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')

TRANSFORMS = dense_transforms.Compose([
    dense_transforms.RectifyData(),
    dense_transforms.ToTensor(),
    dense_transforms.CenterToHeatmap(),
])


def profile_model(model, data, device):
    model.to(device)
    model.eval()

    times = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (x, y) in enumerate(data):
        x, y = x.to(device), y.to(device)

        start.record()
        _ = model.forward(x)
        end.record()
        torch.cuda.synchronize()

        inf_time = start.elapsed_time(end)
        times.append(inf_time)

    times = np.array(times)
    return np.mean(times)


def main(data_path, model_path, n_class=3):

    model = Detector(n_class=n_class)
    model.load_state_dict(torch.load(str(model_path)))
    summary(model, input_size=(1, n_class, 400, 300), depth=1)

    dataloader = load_detection_data(
        data_path, num_workers=1, batch_size=1, transform=TRANSFORMS)

    print(f"Profiling '{model_path.name}' on {len(dataloader.dataset)} images")

    for d in DEVICES:
        device = torch.device(d)
        avg_runtime = profile_model(model, dataloader, device)
        print(f'Device {d}: {avg_runtime}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=Path, required=True)
    parser.add_argument('-w', '--weights', type=Path, required=True)
    parser.add_argument('-n', '--classes', type=int, required=True)

    args = parser.parse_args()

    main(
        model_path=args.weights,
        data_path=args.data,
        n_class=args.classes,
    )
