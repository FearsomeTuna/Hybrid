import os
import os.path
import argparse
import torch

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset. Taken from torchvision.datasets.folder DatasetFolder implementation.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_dataset(
    checkpoint_directory: str,
    resume: bool,
    directory: str = None,    
    class_to_idx: Dict[str, int] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    save_freq: int = 10
) -> List[Tuple[str, int]]:
    """Searches a dataset directory and generates a list of (path_to_sample, class) tuples and saves checkpoint to file named 'data_init.pt'.

    Args:
        checkpoint_directory (str, optional): output directory folder. Partial results are saved here and are used as checkpoints.
        resume (bool): indicates whether to resume from previous (incomplete) run
        directory (str, optional): root dataset directory, unused when resuming from checkpoint.
        class_to_idx (Dict[str, int], optional): dictionary mapping class name to class index, unused when resuming from checkpoint
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.
        save_freq (int, optional): checkpoint will be saved every save_freq classes processed.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """

    checkpoint_file = os.path.join(checkpoint_directory, 'dataset_checkpoint.pt')
    is_file = os.path.isfile(checkpoint_file)
    if resume and not is_file:
        print("No checkpoint found. Starting from scratch.")

    if resume and is_file:
        results = torch.load(checkpoint_file)
        instances = results["instances"]
        directory = results["directory"]
        class_to_idx = results["class_to_idx"]
        extensions = results["extensions"]
        completed = results["completed"]
        print("Resuming after {} processed classes".format(len(completed)))
    else:
        completed = []
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    class_to_idx_len = len(class_to_idx)

    print("Start iterating...")
    for i, target_class in enumerate(sorted(class_to_idx.keys())):
        if target_class in completed:
            continue
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        print("Getting tree hierarchy for class {}/{} ({})".format(i+1,class_to_idx_len, target_class))
        os_walk = sorted(os.walk(target_dir, followlinks=True))
        os_walk_len = len(os_walk)
        print("Start processing for class {}/{}".format(i+1, class_to_idx_len))
        for root, _, fnames in os_walk:
            fnames_len = len(fnames)
            print("Processing {} files for class {}/{}".format(fnames_len, i+1, class_to_idx_len))
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
        completed.append(target_class)
        if i % save_freq == 0:
            print("Backing up last save...")
            backup = os.path.join(checkpoint_directory, 'dataset_checkpoint_backup.pt')
            if os.path.isfile(backup):
                os.remove(backup)
            if os.path.isfile(checkpoint_file):
                os.rename(checkpoint_file, backup)
            print("Saving iteration results to {}".format(checkpoint_file))
            torch.save({'class_to_idx': class_to_idx, 'instances': instances, 'extensions': extensions, "completed": completed, 'directory': directory}, checkpoint_file)
    return instances

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Searches a folder hierarchy and makes dataset path dictionary and saves it to custom
        file via torch.save. Custom file will contain dictionary with the following keys:
        
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (image_path, class_index) tuples, where image_path is a string and class_index is int.
        extensions (list): List of string with allowed image extensions.
        
        Process will save periodic checkpoints. If processed is completed, checkpoints will be deleted.
        If process is interrupted, next call with same arguments will resume from last checkpoint.
        In case checkpoint is corrupted (due to interruption mid-save), a checkpoint backup is also generated
        and can be renamed to dataset_checkpoint.pt''')
    parser.add_argument('path', type=str, help='Folder containing images in subfolders per each class (same as torchvision.datasets.ImageFolder requires).')
    parser.add_argument('output_path', type=str, help='Output folder location for final output and temporal checkpoints.')
    parser.add_argument('name', type=str, help='Output filename.')
    args = parser.parse_args()
    
    print("Finding classes...")
    classes, class_to_idx = find_classes(args.path)
    print("Making path dictionary:")
    samples = make_dataset(args.output_path, resume=True, directory=args.path, class_to_idx=class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=None, save_freq=10)
    print("Finished making path dictionary.")
    filename = os.path.join(args.output_path, args.name)
    print("Saving results to {}".format(filename))
    torch.save({'classes': classes, 'class_to_idx': class_to_idx, 'samples': samples, 'extensions': IMG_EXTENSIONS}, filename)
    print("Deleting checkpoints.")
    os.remove(os.path.join(args.output_path, 'dataset_checkpoint_backup.pt'))
    os.remove(os.path.join(args.output_path, 'dataset_checkpoint.pt'))
    print("Done")
    