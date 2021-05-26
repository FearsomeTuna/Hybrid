from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import os
import os.path
import argparse
import pandas as pd
import torch

def split_dataset(ratios: Union[List[float], float], samples: List[Tuple[str, int]]) -> List[List[Tuple[str,int]]]:
    """Splits dataset in dictionary according to given ratios, returning the lists of tuples selected for each subset.

    Ratios provided must sum less than 1 in total, the last ratio is inferred.
    Thus, the number of subset lists returned is the number of ratios provided plus one.
    Split subsets are balanced (ratios are enforced within each class).

    To replicate results, set manual seed with numpy.random.seed().

    Args:
        ratios (List[float], float]): a list of positive ratios. Total sum must be less than 1. If only one ratio is provided, can be passed as a float. Last ratio is inferred from the total sum of provided ratios.
        samples (List[Tuple[str, int]]): List of (img_path, class) tuples to be split.

    Returns:
        List[List[Tuple[str,int]]]: List containing tuple lists of split samples for each subset.
    """
    if not isinstance(ratios, list):
        ratios = [ratios]
    assert len(ratios) > 0 and sum(ratios) <= 1

    print("Transforming to dataframe.")
    df = pd.DataFrame(samples)
    remaining_df = df
    split_samples = []
    
    for i, r in enumerate(ratios):
        print("Spliting subset {}/{}".format(i+1, len(ratios)+1))
        print("Grouping and sampling by class...")
        sampled_df = remaining_df.groupby(1).sample(frac=r)
        print("Transforming to tuple list...")
        split_samples.append(list(sampled_df.itertuples(index=False, name=None)))
        print("Calculating remaining data...")
        remaining_df = remaining_df.drop(sampled_df.index)
    print("Transforming last subset ({}/{}) to list".format(len(ratios)+1, len(ratios)+1))
    split_samples.append(list(remaining_df.itertuples(index=False, name=None)))
    print("Done")
    return split_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Splits dataset taken from custom file (see torch.save) with dataset images paths
        and outputs a new file with same format for each partition.
        Custom file must have dictionary with at least the following keys:
        
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (image_path, class_index) tuples, where image_path is a string and class_index is int.
        extensions (list): List of string with allowed image extensions.
        
        Outputs partition files are named 'split_<num>.pt', where <num> starts from zero, in the same
        order as provided ratios. Output format has same dictionary keys as input file but values reflecting split data.''')
    parser.add_argument('custom_path', type=str, help='Path to custom file with dataset initialization info.')
    parser.add_argument('output_path', type=str, help='Output folder location for generated files')
    parser.add_argument('-r', '--ratios', type=float, nargs='+', help='Ratios for data split. Should be positive and sum must be less than 1. Aditional ratio is inferred from difference to 1.')
    args = parser.parse_args()

    assert sum(args.ratios) < 1
    for r in args.ratios:
        assert r > 0

    print("Loading " + args.custom_path)
    data_init = torch.load(args.custom_path)
    print("Start splitting...")
    split = split_dataset(args.ratios, data_init['samples'])

    for i, sample in enumerate(split):
        data_init['samples'] = sample
        filename = os.path.join(args.output_path, 'split_{}.pt'.format(i))
        print("Saving split {}/{}, ({} samples) to {}".format(i+1, len(split), len(data_init['samples']), filename))
        torch.save(data_init, filename)