# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import numpy as np
import zipfile
import torch
import dnnlib
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None


# ----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            name,  # Name of the dataset.
            raw_shape,  # Shape of the raw image data (NCHW).
            max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
            use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
            xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
            random_seed=0,  # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # We don't Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._w[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    def __init__(
            self,
            path,  # Path to directory or zip.
            camera_path,  # Path to camera
            resolution=None,  # Ensure specific resolution, None = highest available.
            data_camera_mode='shapenet_car',
            add_camera_cond=False,
            split='all',
            **super_kwargs  # Additional arguments for the Dataset base class.
    ):
        self.data_camera_mode = data_camera_mode
        self._path = path
        self._zipfile = None
        self.root = path
        self.mask_list = None
        self.add_camera_cond = add_camera_cond
        root = self._path
        self.camera_root = camera_path
        if data_camera_mode == 'shapenet_car' or data_camera_mode == 'shapenet_chair' \
                or data_camera_mode == 'renderpeople' or data_camera_mode == 'shapenet_motorbike' \
                or data_camera_mode == 'ts_house' \
                or data_camera_mode == 'ts_animal':
            print('==> use shapenet dataset')
            if not os.path.exists(root):
                print('==> ERROR!!!! THIS SHOULD ONLY HAPPEN WHEN USING INFERENCE')
                n_img = 1234
                self._raw_shape = (n_img, 3, resolution, resolution)
                self.img_size = resolution
                self._type = 'dir'
                self._all_fnames = [None for i in range(n_img)]
                self._image_fnames = self._all_fnames
                name = os.path.splitext(os.path.basename(path))[0]
                print(
                    '==> use image path: %s, num images: %d' % (
                        self.root, len(self._all_fnames)))
                super().__init__(name=name, raw_shape=self._raw_shape, **super_kwargs)
                return
            folder_list = sorted(os.listdir(root))
            if data_camera_mode == 'shapenet_chair' or data_camera_mode == 'shapenet_car':
                if data_camera_mode == 'shapenet_car':
                    split_name = './3dgan_data_split/shapenet_car/%s.txt' % (split)
                    if split == 'all':
                        split_name = './3dgan_data_split/shapenet_car.txt'
                elif data_camera_mode == 'shapenet_chair':
                    split_name = './3dgan_data_split/shapenet_chair/%s.txt' % (split)
                    if split == 'all':
                        split_name = './3dgan_data_split/shapenet_chair.txt'
                valid_folder_list = []
                with open(split_name, 'r') as f:
                    all_line = f.readlines()
                    for l in all_line:
                        valid_folder_list.append(l.strip())
                valid_folder_list = set(valid_folder_list)
                useful_folder_list = set(folder_list).intersection(valid_folder_list)
                folder_list = sorted(list(useful_folder_list))
            if data_camera_mode == 'ts_animal':
                split_name = './3dgan_data_split/ts_animals/%s.txt' % (split)
                print('==> use ts animal split %s' % (split))
                if split != 'all':
                    valid_folder_list = []
                    with open(split_name, 'r') as f:
                        all_line = f.readlines()
                        for l in all_line:
                            valid_folder_list.append(l.strip())
                    valid_folder_list = set(valid_folder_list)
                    useful_folder_list = set(folder_list).intersection(valid_folder_list)
                    folder_list = sorted(list(useful_folder_list))
            elif data_camera_mode == 'shapenet_motorbike':
                split_name = './3dgan_data_split/shapenet_motorbike/%s.txt' % (split)
                print('==> use ts shapenet motorbike split %s' % (split))
                if split != 'all':
                    valid_folder_list = []
                    with open(split_name, 'r') as f:
                        all_line = f.readlines()
                        for l in all_line:
                            valid_folder_list.append(l.strip())
                    valid_folder_list = set(valid_folder_list)
                    useful_folder_list = set(folder_list).intersection(valid_folder_list)
                    folder_list = sorted(list(useful_folder_list))
            print('==> use shapenet folder number %s' % (len(folder_list)))
            folder_list = [os.path.join(root, f) for f in folder_list]
            all_img_list = []
            all_mask_list = []

            for folder in folder_list:
                rgb_list = sorted(os.listdir(folder))
                rgb_list = [n for n in rgb_list if n.endswith('.png') or n.endswith('.jpg')]
                rgb_file_name_list = [os.path.join(folder, n) for n in rgb_list]
                all_img_list.extend(rgb_file_name_list)
                all_mask_list.extend(rgb_list)

            self.img_list = all_img_list
            self.mask_list = all_mask_list

        else:
            raise NotImplementedError
        self.img_size = resolution
        self._type = 'dir'
        self._all_fnames = self.img_list
        self._image_fnames = self._all_fnames
        name = os.path.splitext(os.path.basename(self._path))[0]
        print(
            '==> use image path: %s, num images: %d' % (self.root, len(self._all_fnames)))
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        if self.data_camera_mode == 'shapenet_car' or self.data_camera_mode == 'shapenet_chair' \
                or self.data_camera_mode == 'renderpeople' \
                or self.data_camera_mode == 'shapenet_motorbike' or self.data_camera_mode == 'ts_house' or self.data_camera_mode == 'ts_animal' \
                :
            ori_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            img = ori_img[:, :, :3][..., ::-1]
            mask = ori_img[:, :, 3:4]
            condinfo = np.zeros(2)
            fname_list = fname.split('/')
            img_idx = int(fname_list[-1].split('.')[0])
            obj_idx = fname_list[-2]
            syn_idx = fname_list[-3]

            if self.data_camera_mode == 'shapenet_car' or self.data_camera_mode == 'shapenet_chair' \
                    or self.data_camera_mode == 'renderpeople' or self.data_camera_mode == 'shapenet_motorbike' \
                    or self.data_camera_mode == 'ts_house' or self.data_camera_mode == 'ts_animal':
                if not os.path.exists(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy')):
                    print('==> not found camera root')
                else:
                    rotation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy'))
                    elevation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'elevation.npy'))
                    condinfo[0] = rotation_camera[img_idx] / 180 * np.pi
                    condinfo[1] = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
        else:
            raise NotImplementedError

        resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if not mask is None:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)  ########
        else:
            mask = np.ones(1)
        img = resize_img.transpose(2, 0, 1)
        background = np.zeros_like(img)
        img = img * (mask > 0).astype(np.float) + background * (1 - (mask > 0).astype(np.float))
        return np.ascontiguousarray(img), condinfo, np.ascontiguousarray(mask)

    def _load_raw_image(self, raw_idx):
        if raw_idx >= len(self._image_fnames) or not os.path.exists(self._image_fnames[raw_idx]):
            resize_img = np.zeros((3, self.img_size, self.img_size))
            return resize_img

        img = cv2.imread(self._image_fnames[raw_idx])[..., ::-1]
        resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR) / 255.0
        resize_img = resize_img.transpose(2, 0, 1)
        return resize_img

    def _load_raw_labels(self):
        return None
