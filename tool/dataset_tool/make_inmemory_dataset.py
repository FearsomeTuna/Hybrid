import argparse
import torch
import torchvision.transforms as transforms
import os
import io
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Loads images objects , applies tranforms for --type val and saves them to file as io.BytesIO objects
        with torch.save().
        Generated file is meant to be loaded whole into memory during training.
        Transforms are not performed on --type train (because they're meant to be mostly random
        augmentations performed at runtime).
        For --type val, images are resized to 256x256 followed by center crop (224x224).
        No normalization or transformation to tensor is performed.
        This format uses less space than saving tensor objects directly.
        
        Input custom file is opened via torch.load and must be dictionary with at least the following keys:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (bytes, class_index) tuples, where bytes is a BytesIO object and class_index is int.

        Output file has at least the same keys.
        ''')
    parser.add_argument('file_path', type=str, help='Path to file with dataset initialization info, i.e. a list of (img_path, class_idx) tuples.')
    parser.add_argument('output_filepath', type=str, help='Output file path for generated file')
    parser.add_argument('-n', '--noValTransforms', nargs='?', const=True, default=False, help='Do not apply any transforms.')
    parser.add_argument('-t', '--type', type=str, required=True, help='Either train or val. Used to pick appropiate transforms.')
    args = parser.parse_args()
    
    if args.noValTransforms or args.type == 'train':
        transform = lambda x : x
    elif args.type == 'val':
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    else:
        raise ValueError("Wrong value for --type argument. Must be either train or val.")

    print("Loading dataset initialization info...")
    load_file = torch.load(args.file_path)
    samples = load_file['samples']
    total = len(samples)
    print("Samples to process: {}".format(total))

    i = 0
    def process(tup):
        global i
        with open(tup[0], 'rb') as f:
            img = Image.open(f)
            img_format = img.format
            img = transform(img)
            bytesIO = io.BytesIO()
            img.save(bytesIO, format=img_format)
        i+=1
        if(i % 50 == 0):
            print("\r{:.6f} %".format(100*i/total), end="")
        return (bytesIO, tup[1]) # bytesIO.getvalue() to get bytes content

    print("Processing images...")
    processed_samples = list(map(process, samples))
    print("\r{:.6f} %".format(100), end="")
    print("")

    load_file['samples'] = processed_samples

    filename = args.output_filepath
    print("Saving results to {}".format(filename))
    torch.save(load_file, filename)
