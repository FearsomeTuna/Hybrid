import os
import os.path

import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img

class PathsFileDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            filename: str = 'data_init.pt',
            loader: Callable[[str], Any] = pil_loader,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """Same as torchvision.datasets.DatasetFolder class but initialized from custom file (see torch.save)
        with dictionary object containing the following keys:
        
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (image_path, class_index) tuples, where image_path is a string and class_index is int.
        extensions (list): List of string with allowed image extensions.

        Args:
            root (str): Path to folder containing custom file to be loaded.
            filename (str, optional): Name of custom file. Defaults to 'data_init.pt'.
            loader (Callable[[str], Any], optional): A function to load a sample given its path. Defaults to pil_loader.
            transform (Optional[Callable], optional): A function/transform that takes in an sample loaded with 'loader' and returns a transformed version. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform that takes in the target and transforms it. Defaults to None.

        Raises:
            RuntimeError: In case zero samples are defined in custom file.
        """
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        path = os.path.join(root, filename)
        dataset_dict = torch.load(path)
        classes, class_to_idx = dataset_dict["classes"], dataset_dict["class_to_idx"]
        samples = dataset_dict["samples"]
        extensions = dataset_dict['extensions']
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

class InMemoryDataset(Dataset):
    def __init__(
            self,
            filepath: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """Initializes dataset from custom file (see torch.save) containing io.BytesIO representation of images.
    Whole dataset is loaded into memory.
    
    Initialized from custom object file (see with torch.save) containing a dictionary with the following keys:
    classes (list): List of the class names sorted alphabetically.
    class_to_idx (dict): Dict with items (class_name, class_index).
    samples (list): List of (byte_image, class_index) tuples, where byte_image is an io.BytesIO object and class_index is int.

        Args:
            filepath (str): path to file contaning initialization info.
            transform (Optional[Callable], optional): A function/transform that takes in an retrieved sample and returns a transformed version. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform that takes in the target and transforms it. Defaults to None.

        Raises:
            RuntimeError: In case zero samples are defined in custom file.
    """
        loaded_file = torch.load(filepath)
        classes, class_to_idx = loaded_file['classes'], loaded_file['class_to_idx']
        samples = loaded_file['samples']
        if len(samples) == 0:
            msg = "Found 0 files in loaded file"
            raise RuntimeError(msg)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        bytes_io, target = self.samples[index]
        sample = Image.open(bytes_io)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        return len(self.samples)
