import os
import os.path
import random
from enum import Enum, auto

import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from torchvision.datasets.vision import VisionDataset

from typing import Any, Callable, Optional, Tuple


class TxtFileDataset(VisionDataset):
    def __init__(
            self,
            root,
            images: str,
            classes: str,
            separator : str,
            target_last: bool,
            loader: Callable[[str], Any] = default_loader,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            return_index: bool = False
    ) -> None:
        """Similar to torchvision.datasets.DatasetFolder class but initialized from txt file with path and targe mappings.
        
        Each line in Txt file must have an image path and an integer index representing the class of the image.
        Which one of the two is expected first can be changed with the target_last parameter. Separator can also be
        changed. Image paths should not repeat.
        Idexes MUST be integers ranging from 0 to NCLASSES-1.

        Another txt file is used to provide mappings between string labels and their corresponding class index
        (totaling one line for each class).
        This txt file should use the same separator as the image path mappings, and target index should also be
        first or second on each line according to target_last parameter.

        Object attributes:        
        classes (list): List of class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (image_path, class_index) tuples, where image_path is a string and class_index is int.

        Args:
            root (str): Path to folder containing txt files to be loaded.
            images (str): Name of txt file containing paths and labels.
            classes (str): Name of txt containing class names and their corresponding index. These index mappings should match
              with the ones used in 'images' txt file.
            separator (str): string to be used as separator between values on each line.
            target_last (bool): indicates whether the class index is the first or second value on each line. Used for both txt files.
            loader (Callable[[str], Any], optional): A function to load a sample given its path. By default, uses same loader as
              torchvision.datasets.DatasetFolder (pil loader or accimage loader, depending on image backend).
            transform (Optional[Callable], optional): A function/transform that takes in an sample loaded with 'loader' and returns a transformed version. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform that takes in the target and transforms it. Defaults to None.
            return_index (bool): if True, accesed items will include the unique sample index. This is useful when calculating retrieval metrics
              (like mAP) over samples batched with DataLoader objects, since these objects obscure the indexes retrieved, often required
              to distinguish each distinct query element appropriately. Defaults to False
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        images_path = os.path.join(root, images)
        assert os.path.exists(images_path)

        samples = []
        validation_set = set()
        with open(images_path) as file:
            for line in file:
                line = line.rstrip().split(separator)
                if target_last:
                    img_path, target = line[0], int(line[1])
                else:
                    img_path, target = line[1], int(line[0])
                samples.append((img_path, target))
                validation_set.add(target)
        assert min(validation_set) == 0
        class_qty = len(validation_set)
        assert max(validation_set) == class_qty - 1

        classes_path = os.path.join(root, classes)
        assert os.path.exists(classes_path)

        class_to_idx = dict()
        with open(classes_path) as file:
            lines = [tuple(line.rstrip().split(separator)) for line in file]
            if target_last:
                mappings = [(tup[0], int(tup[1])) for tup in lines]
            else:
                mappings = [(tup[1], int(tup[0])) for tup in lines]
            for label, idx in mappings:
                class_to_idx[label] = idx
        values = class_to_idx.values()
        assert len(values) == class_qty
        assert min(values) == 0
        assert max(values) == class_qty - 1
        
        self.return_index = return_index
        self.loader = loader
        self.classes = sorted(class_to_idx.keys())
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is the integer index representing the target class.
              and sample is the loaded image.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return sample, target, index
        else:
            return sample, target

    def __len__(self) -> int:
        return len(self.samples)

class SBIRDataset(Dataset):
    class MODES (Enum):
        PAIR = auto()
        TRIPLET = auto()

    def __init__(
            self,
            sketch_root: str,
            image_root: str,
            sketches_txt: str,
            images_txt: str,
            sketches_classes: str,
            images_classes: str,
            separator: str,
            target_last: str,
            sketch_transform: Optional[Callable] = None,
            image_transform: Optional[Callable] = None,
    ) -> None:
        """
        SBIR dataset object. Based on TxtFileDataset interface for txt file manipulation.

        Has two modes, PAIR and TRIPLET (must be set with setMode method), for constrastive and triplet training. Each sketch is used once
        and positive/negative pair/triplets are selected at random during runtime. Hence, dataset length
        is that of the sketch set used. For PAIR mode, even indexes make positive pairs, and odd ones make negative ones.

        Object attributes:        
        classes (list): List of class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        sketches (TxtFileDataset): sketch dataset object
        images (TxtFileDataset): photo dataset object

        Args:
            sketch_root (str): Path to folder containing txt files for sketch data.
            image_root (str): Path to folder containing txt files for photo data.
            sketches_txt (str): filename for sketch txt within sketch root. Example: 'train.txt'.
            images_txt (str): filename for photo txt within photo root. Example: 'train.txt'.
            sketches_classes (str): filename for sketch classes mappings file within sketch root. Example: 'mapping.txt',
            images_classes (str): filename for photo class mappings file within photo root. Example: 'mapping.txt',
            separator (str): string to be used as separator between values on each line.
            target_last (bool): indicates whether the class index is the first or second value on each line. Used for all files.
            sketch_transform (Optional[Callable], optional): Transform for sketch data. A function/transform that takes in an sample loaded with 'loader' and returns a transformed version. Defaults to None.
            image_transform (Optional[Callable], optional): same as sketch_transform, for photo data.
        """
        super().__init__()
        self.sketches = TxtFileDataset(sketch_root, sketches_txt, sketches_classes, separator=separator, target_last=target_last, transform=sketch_transform)
        self.images = TxtFileDataset(image_root, images_txt, images_classes, separator=separator, target_last=target_last, transform=image_transform)

        assert len(self.sketches.classes) == len(self.images.classes)
        for i in range(len(self.sketches.classes)):
            if self.sketches.classes[i] != self.images.classes[i]:
                raise ValueError('Image and sketch datasets class mismatch. Image: {}, sketch: {}. Make sure class names match when making dataset_paths file to ensure same class indexing.'.format(self.images.classes[i], self.sketches.classes[i]))
        self.classes = self.sketches.classes.copy()
        for category in self.classes:
            if self.sketches.class_to_idx[category] != self.images.class_to_idx[category]:
                raise ValueError('Index mismatch in image and sketch datasets. Class {} has index {} in image dataset and index {} in sketches. Make sure class names match when making dataset_paths file to ensure same class indexing.'.format(category, self.images.class_to_idx[category], self.sketches.class_to_idx[category]))
        self.class_to_idx = self.sketches.class_to_idx.copy()

        self.image_groups = self._make_grouped_samples(self.images)

    def setMode(self, mode: MODES):
        assert mode in SBIRDataset.MODES
        self.mode = mode

    def __getitem__(self, index: int) -> Tuple:
        """- Returns (sketch, image, sketch_target, image_target) tuple if self.mode == SBIRDataset.MODES.PAIR,
        - Returns (sketch, image_positive, image_negative, sketch_target, image_negative_target) tuple if self.mode == SBIRDataset.MODES.TRIPLET
        """
        image_positive, image_negative, image_negative_target = None, None, None
        pair_same_target = index % 2 == 0

        sketch, sketch_target = self.sketches[index]
        if self.mode == SBIRDataset.MODES.PAIR and pair_same_target: # same target for both samples
            image_positive = self.getRandomImagePath(sketch_target)
        elif self.mode == SBIRDataset.MODES.PAIR or self.mode == SBIRDataset.MODES.TRIPLET: # different target
            image_negative_target = random.choice(list(set(self.class_to_idx.values()) - {sketch_target}))
            image_negative = self.getRandomImagePath(image_negative_target)
            if self.mode == SBIRDataset.MODES.TRIPLET:
                image_positive = self.getRandomImagePath(sketch_target)
        else:
            raise RuntimeError("Invalid mode.")


        image_positive = self.images.loader(image_positive) if image_positive is not None else None
        image_negative = self.images.loader(image_negative) if image_negative is not None else None
        if self.images.transform is not None:
            image_positive = self.images.transform(image_positive) if image_positive is not None else None
            image_negative = self.images.transform(image_negative) if image_negative is not None else None

        if self.mode == SBIRDataset.MODES.PAIR and pair_same_target:
            return sketch, image_positive, sketch_target, sketch_target
        elif self.mode == SBIRDataset.MODES.PAIR: # not same target
            return sketch, image_negative, sketch_target, image_negative_target
        elif self.mode == SBIRDataset.MODES.TRIPLET:
            return sketch, image_positive, image_negative, sketch_target, image_negative_target
        else:
            raise RuntimeError("Invalid mode.")
        
    def __len__(self) -> int:
        return len(self.sketches)

    def getRandomImagePath(self, class_idx: int) -> str:
        sample_path = random.choice(self.image_groups[class_idx])
        return sample_path

    def _make_grouped_samples(self, txtDataset: TxtFileDataset):
        grouped_samples = [[] for i in range(len(txtDataset.classes))]
        for i, sample in enumerate(txtDataset.samples):
            grouped_samples[sample[1]].append(sample[0])
        return grouped_samples
