from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import random
import numpy as np
import os
import os.path
import argparse
import pandas as pd
import torch
import pprint

def sub_sample(samples: List[Tuple[str, int]], n: Optional[int] = None, frac: Optional[float] = None, classes: Optional[List[int]] = None) -> List[List[Tuple[str,int]]]:
    """Returns list of tuples corresponding to a subsample of the samples provided.

    To replicate results, set manual seed with numpy.random.seed().

    Args:
        samples (List[Tuple[str, int]]): List of (img_path, class) tuples to be subsampled.
        n (int, optional): Number of random samples to take from each sampled class. Must be no larger than the smallest class. Cannot be used with frac parameter. Defaults to None.
        frac (float, optional): Fraction of items to be randomly sampled from each class. Cannot be used with n parameter. Defaults to None.
        classes (list[int], optional): list of class indices to sample from.
    Returns:
        List[Tuple[str,int]]: list of (img_path, class) elements corresponding to a subsample of the samples provided.
    """
    assert (n is None and frac is not None) or (n is not None and frac is None)
    if frac is not None and (frac < 0 or frac > 1):
        raise ValueError("frac argument must be between 0 and 1.")
    print("Transforming to dataframe...")
    df = pd.DataFrame(samples)     

    print("Grouping by class...")
    df_grouped = df.groupby(1)
    print("Counting classes...")
    df_classes = df_grouped.count()
    total_class_num = df_classes.size

    if n is not None:
        print("Verifying there's at least {} samples per class.".format(n))        
        if not total_class_num == df_classes[df_classes[0]>=n].size:
            raise  ValueError("Not all classes have at least {} examples. Try lowering number of examples per class in subsample or passing fractional argument.".format(n))
    
    print("Sampling...")
    sampled_df = df_grouped.sample(n=n, frac=frac)
    
    if classes:
        print("Removing undesired classes...")
        assert len(classes) <= total_class_num
        sampled_df = sampled_df[sampled_df[1].isin(classes)]

    print("Transforming to list of tuples...")
    result = list(sampled_df.itertuples(index=False, name=None))
    print("Returning...")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Sub sample dataset from custom file (see torch.save) with image paths and classes info.
        Custom file must have dictionary with at least the following keys:
        
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (image_path, class_index) tuples, where image_path is a string and class_index is int.
        extensions (list): List of string with allowed image extensions.
        
        Outputs a file 'sub_data_init.pt' with same dictionary keys as input file but values reflecting subsampled data.''')
    parser.add_argument('file_path', type=str, help='Path to file with dataset initialization info.')
    parser.add_argument('output_file', type=str, help='Output path for generated file.')
    parser.add_argument('-n', '--n', type=int, help='Number of random samples to take from each sampled class. Must be no larger than the smallest class. Cannot be used with frac argument.')
    parser.add_argument('-f', '--frac', type=float, help='Fraction of random samples to be taken from each sampled class. Must be number between zero and one. Cannot be used with n argument.')
    parser.add_argument('-c', '--subclassnum', type=int, help='Number of random classes to be included from original dataset. Cannot be used with excludefile argument.')
    parser.add_argument('-e', '--excludefile', type=str, help='Custom file with same format as file_path argument. Classes included in present subsample will be those in the set difference between file_path argument file classes and this argument file classes. Cannot be used with subclassnum argument.')
    parser.add_argument('-s', '--seed', type=int, help='Pseudo random number generator seed. Used for replicability.')
    args = parser.parse_args()

    assert (args.n is None and args.frac is not None) or (args.n is not None and args.frac is None)
    assert args.subclassnum is None or args.excludefile is None

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("Loading dataset initialization info...")
    load_file = torch.load(args.file_path)

    if args.subclassnum:
        print("Selecting {} random class indices from original dataset...".format(args.subclassnum))
        old_class_indices = list(range(len(load_file['classes'])))
        sampled_classes = sorted(random.sample(old_class_indices, args.subclassnum))
        old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_classes)}
        class_to_idx = {k: old_to_new_map[v] for k, v in load_file['class_to_idx'].items() if v in sampled_classes}
        classes = sorted(list(class_to_idx.keys()))
    elif args.excludefile:
        print("Excluding classes from exclude file...")
        exclude_file = torch.load(args.excludefile)
        exclude_classes = set(exclude_file['classes'])
        classes = sorted(list(set(load_file['classes']) - exclude_classes))
        sampled_classes = sorted([v for k, v in load_file['class_to_idx'].items() if k in classes])
        old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_classes)}
        class_to_idx = {k: old_to_new_map[v] for k, v in load_file['class_to_idx'].items() if v in sampled_classes}
    else:
        sampled_classes = None
        class_to_idx = load_file['class_to_idx']
        classes = load_file['classes']

    print("Initiating subsampling...")
    sub_samples = sub_sample(load_file['samples'], n=args.n, frac=args.frac, classes=sampled_classes)

    print("Sumsampled {} elements from original dataset.".format(len(sub_samples)))

    if args.subclassnum or args.excludefile:
        print("Reindexing classes...")
        final_samples = [(path, old_to_new_map[old_class_idx]) for (path, old_class_idx) in sub_samples]
    else:
        final_samples = sub_samples
    
    print("Displaying sampled classes")
    pprint.pprint(class_to_idx)

    filename = args.output_file
    print("Saving results to {}".format(filename))
    torch.save({
        'classes': classes,
        'class_to_idx': class_to_idx,
        'samples': final_samples,
        'extensions': load_file['extensions']
    }, filename)
