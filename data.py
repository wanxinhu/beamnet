"""Paper splits and green-channel preprocessing; no dataset copies are published."""
import re
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

EXTENSIONS = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.gif', '.bmp'}


def identity(path, dataset):
    stem = Path(path).stem
    pattern = r'^(\d{2})(?:_|$)' if dataset == 'DRIVE' else r'^Image_(\d{2}[LR])(?:_|$)'
    match = re.match(pattern, stem, re.I)
    if not match:
        raise ValueError('Unrecognized {} filename: {}'.format(dataset, Path(path).name))
    return match.group(1).upper()


def paper_ids(dataset, split):
    if dataset == 'DRIVE':
        ranges = {'train': range(21, 36), 'val': range(36, 41), 'test': range(1, 21)}
        return ['{:02d}'.format(n) for n in ranges[split]]
    if dataset != 'CHASE_DB1':
        raise ValueError('Expected DRIVE or CHASE_DB1')
    ids = ['{:02d}{}'.format(n, eye) for n in range(1, 29) for eye in 'LR']
    return {'train': ids[:32], 'val': ids[32:40], 'test': ids[40:]}[split]


def records(root, dataset, split):
    """Index existing images/1st_manual/masks directories by ID, never sorted zip."""
    root = Path(root)
    maps = {name: {} for name in ('images', '1st_manual', 'masks')}
    for path in sorted(root.rglob('*')):
        kind = path.parent.name
        if not path.is_file() or kind not in maps or path.suffix.lower() not in EXTENSIONS:
            continue
        key = identity(path, dataset)
        if key in maps[kind]:
            raise ValueError('Duplicate {} ID {} in dataset root'.format(kind, key))
        maps[kind][key] = path
    result = []
    for key in paper_ids(dataset, split):
        if key not in maps['images'] or key not in maps['1st_manual']:
            raise ValueError('Missing image/first annotation for {} {} ID {}'.format(dataset, split, key))
        if dataset == 'DRIVE' and key not in maps['masks']:
            raise ValueError('DRIVE official FOV mask required for ID {}'.format(key))
        result.append({'id': key, 'image': maps['images'][key],
                       'mask': maps['1st_manual'][key], 'fov': maps['masks'].get(key)})
    return result


def read_rgb(path):
    # PIL supports Unicode paths on Windows as well as TIFF/GIF.
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def preprocess(rgb):
    green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rgb[:, :, 1])
    return np.repeat(green[:, :, None], 3, axis=2)


def tensor_image(image):
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class VesselDataset(Dataset):
    def __init__(self, items, train=False, crop_size=512, augment=True):
        self.items, self.train, self.crop_size, self.augment = items, train, crop_size, augment
        if crop_size < 32 or crop_size % 32:
            raise ValueError('crop_size must be a positive multiple of 32')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record = self.items[index]
        image = preprocess(read_rgb(record['image']))
        with Image.open(record['mask']) as mask_file:
            mask = np.array(mask_file.convert('L'))
        if mask.shape != image.shape[:2]:
            raise ValueError('Image/mask dimensions disagree for '+record['id'])
        mask = (mask > (20 if mask.max() > 127 else 0)).astype(np.float32)
        height, width = mask.shape
        fov = np.ones_like(mask, dtype=bool)
        if record['fov']:
            with Image.open(record['fov']) as f:
                fov = np.array(f.convert('L'))
            if fov.shape != mask.shape:
                raise ValueError('FOV dimensions disagree for '+record['id'])
            fov = fov > (127 if fov.max() > 1 else 0)
        if self.train:
            if self.augment:
                if random.random() < .5:
                    matrix = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), random.uniform(-90, 90), 1)
                    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                if random.random() < .5:
                    image, mask = np.flip(image, 0), np.flip(mask, 0)
                if random.random() < .5:
                    image, mask = np.flip(image, 1), np.flip(mask, 1)
            dh, dw = max(0, self.crop_size-height), max(0, self.crop_size-width)
            image = np.pad(image, ((dh//2, dh-dh//2), (dw//2, dw-dw//2), (0, 0)))
            mask = np.pad(mask, ((dh//2, dh-dh//2), (dw//2, dw-dw//2)))
            top = random.randint(0, image.shape[0]-self.crop_size)
            left = random.randint(0, image.shape[1]-self.crop_size)
            image = image[top:top+self.crop_size, left:left+self.crop_size]
            mask = mask[top:top+self.crop_size, left:left+self.crop_size]
            return tensor_image(image), torch.from_numpy(np.ascontiguousarray(mask[None]))
        image = np.pad(image, ((0, (-height) % 32), (0, (-width) % 32), (0, 0)))
        return tensor_image(image), mask, fov, record['id']


def training_loader(items, batch_size=4, crop_size=512, workers=0, seed=42):
    if batch_size < 2:
        raise ValueError('SKU BatchNorm requires training batch size >= 2')
    if len(items) < batch_size:
        raise ValueError('Training set is smaller than batch size')
    return DataLoader(VesselDataset(items, train=True, crop_size=crop_size),
                      batch_size=batch_size, shuffle=True, num_workers=workers,
                      drop_last=True, generator=torch.Generator().manual_seed(seed))

