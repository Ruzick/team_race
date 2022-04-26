from PIL import Image
import numpy as np
from enum import IntEnum
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from custom import dense_transforms
TRACK_NAME = "icy_soccer_field_"
DATA_PATH = 'image_data'


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path = DATA_PATH , transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_segmentation.png')):
            self.files.append(im_f.replace('_segmentation.png', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        print(b)
        im = Image.open(b + '.png')
        lbl = Image.open(b + '_segmentation.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl

def load_dense_data(dataset_path=DATA_PATH, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


from custom import dense_transforms

class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATA_PATH, transform=dense_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = [ ]
        self.label = [ ]
        self.transform = transform

        for f in glob(path.join(dataset_path,'*.csv')):
          self.label.append(np.loadtxt(f, dtype=np.float32, delimiter=','))

        for seg_im_f in glob(path.join(dataset_path,'*_segmentation.png')):
            self.files.append(seg_im_f.replace('_segmentation.png', ''))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.label[idx]
        b = self.files[idx]
        im = Image.open(b +".png")
        lbl = Image.open(b + '_segmentation.png')
        lbl = self.transform(lbl) #everything has to be a tensor
        data = im, lbl, label
        if self.transform is not None:
            data = self.transform(*data)
        return data



def load_detection_data(dataset_path=DATA_PATH, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)



class Team(IntEnum):
    RED = 0
    BLUE = 1


def video_grid(team1_images, team2_images, team1_state='', team2_state=''):
    from PIL import Image, ImageDraw
    grid = np.hstack((np.vstack(team1_images), np.vstack(team2_images)))
    grid = Image.fromarray(grid)
    grid = grid.resize((grid.width // 2, grid.height // 2))

    draw = ImageDraw.Draw(grid)
    draw.text((20, 20), team1_state, fill=(255, 0, 0))
    draw.text((20, grid.height // 2 + 20), team2_state, fill=(0, 0, 255))
    return grid


def map_image(team1_state, team2_state, soccer_state, resolution=512, extent=65, anti_alias=1):
    BG_COLOR = (0xee, 0xee, 0xec)
    RED_COLOR = (0xa4, 0x00, 0x00)
    BLUE_COLOR = (0x20, 0x4a, 0x87)
    BALL_COLOR = (0x2e, 0x34, 0x36)
    from PIL import Image, ImageDraw
    r = Image.new('RGB', (resolution*anti_alias, resolution*anti_alias), BG_COLOR)

    def _to_coord(x):
        return resolution * anti_alias * (x + extent) / (2 * extent)

    draw = ImageDraw.Draw(r)
    # Let's draw the goal line
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][0]], width=5*anti_alias, fill=RED_COLOR)
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][1]], width=5*anti_alias, fill=BLUE_COLOR)

    # and the ball
    x, _, y = soccer_state['ball']['location']
    s = soccer_state['ball']['size']
    draw.ellipse((_to_coord(x-s), _to_coord(y-s), _to_coord(x+s), _to_coord(y+s)), width=2*anti_alias, fill=BALL_COLOR)

    # and karts
    for c, s in [(BLUE_COLOR, team1_state), (RED_COLOR, team2_state)]:
        for k in s:
            x, _, y = k['kart']['location']
            fx, _, fy = k['kart']['front']
            sx, _, sy = k['kart']['size']
            s = (sx+sy) / 2
            draw.ellipse((_to_coord(x - s), _to_coord(y - s), _to_coord(x + s), _to_coord(y + s)), width=5*anti_alias, fill=c)
            draw.line((_to_coord(x), _to_coord(y), _to_coord(x+(fx-x)*2), _to_coord(y+(fy-y)*2)), width=4*anti_alias, fill=0)

    if anti_alias == 1:
        return r
    return r.resize((resolution, resolution), resample=Image.ANTIALIAS)


# Recording functionality
class BaseRecorder:
    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        raise NotImplementedError

    def __and__(self, other):
        return MultiRecorder(self, other)

    def __rand__(self, other):
        return MultiRecorder(self, other)


class MultiRecorder(BaseRecorder):
    def __init__(self, *recorders):
        self._r = [r for r in recorders if r]

    def __call__(self, *args, **kwargs):
        for r in self._r:
            r(*args, **kwargs)


class VideoRecorder(BaseRecorder):
    """
        Produces pretty output videos
    """
    def __init__(self, video_file):
        import imageio
        self._writer = imageio.get_writer(video_file, fps=20)

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        if team1_images and team2_images:
            self._writer.append_data(np.array(video_grid(team1_images, team2_images,
                                                         'Blue: %d' % soccer_state['score'][1],
                                                         'Red: %d' % soccer_state['score'][0])))
        else:
            self._writer.append_data(np.array(map_image(team1_state, team2_state, soccer_state)))

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close()


class DataRecorder(BaseRecorder):
    def __init__(self, record_images=False):
        self._record_images = record_images
        self._data = []

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        data = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            data['team1_images'] = team1_images
            data['team2_images'] = team2_images
        self._data.append(data)

    def data(self):
        return self._data

    def reset(self):
        self._data = []


class StateRecorder(BaseRecorder):
    def __init__(self, state_action_file, record_images=False):
        self._record_images = record_images
        self._f = open(state_action_file, 'wb')

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        from pickle import dump
        data = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            data['team1_images'] = team1_images
            data['team2_images'] = team2_images
        dump(dict(data), self._f)
        self._f.flush()

    def __del__(self):
        if hasattr(self, '_f'):
            self._f.close()


def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


# if __name__ == '__main__':
#     dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose(
#         [
        
#         dense_transforms.RandomHorizontalFlip(), 

#         dense_transforms.ToTensor(),
#         dense_transforms.Normalize([0.32352477, 0.3310059 , 0.34449455], [0.25328732, 0.22241966, 0.24833776])
#         ]))
#     from pylab import show, imshow, subplot, axis

#     for i in range(15):
#         im, lbl = dataset[i]
#         subplot(5, 6, 2 * i + 1)
#         imshow(F.to_pil_image(im))
#         axis('off')
#         subplot(5, 6, 2 * i + 2)
#         imshow(dense_transforms.label_to_pil_image(lbl))
#         axis('off')
#     show()
#     import numpy as np

#     c = np.zeros(5)
#     for im, lbl in dataset:
#         c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
#     print(100 * c / np.sum(c))

#     #borrowed from hw2
# def accuracy(outputs, labels):
#     outputs_idx = outputs.max(1)[1].type_as(labels)
#     return outputs_idx.eq(labels).float().mean()