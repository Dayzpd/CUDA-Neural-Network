from PIL import Image
import numpy as np

import csv
from getopt import getopt, GetoptError
import glob
from math import ceil
from pathlib import Path
import os
import random
import sys

OUTPUT_DIR = 'formatted_dataset'
BATCH_DIR = 'batches'


def get_input_files(input_dir):
    '''
    Summary: Gets path and object classification information for all features.
    Args:
        - input_dir: Input directory for dataset.
    Returns:
        - objects: Maps integer class index to string object class name.
        - features: Contains information on object class index and the image's
                    path.
    Note: Input dataset directory should be structured as follows...
          - <dataset_folder>
              - <object name>_<object index>
                  - image.ext
                  - image.ext
                  - ...
              - <object name>_<object index>
                  - image.ext
                  - image.ext
                  - ...
              - <object name>_<object index>
                  - image.ext
                  - image.ext
                  - ...
          Indexing must start at 0 and increment by 1 for each following object
          class.
    '''
    objects = {}
    features = {}

    feature_num = 0
    sub_dirs = os.listdir(input_dir)

    for sub_dir in sub_dirs:
        obj_name, obj_index = sub_dir.split('_')
        objects[int(obj_index)] = obj_name

        sub_dir_path = os.path.join(input_dir, sub_dir)
        for feature in os.listdir(sub_dir_path):
            features[feature_num] = {
                'path': os.path.join(sub_dir_path, feature),
                'class': obj_index
            }

            feature_num += 1

    return objects, features


def object_labels(objects, output_path):
    '''
    Summary: Creates a CSV file mapping object class names to integer index
             values.
    Args:
        - objects: List mapping integer index value to string object class name.
        - output_path: Path to output object_labels.csv file.
    '''
    obj_labels_file = os.path.join(output_path, 'object_labels.csv')
    with open(obj_labels_file, mode='w+') as file:
        writer = csv.writer(file, delimiter=',')
        for x in range(0, len(objects)):
            writer.writerow([objects[x], x])


def read_image(path, dim_size):
    '''
    Summary: Reads image as grayscale, resizes it to standard size, and returns
             the image as a numpy array.
    Args:
        - path: Path to the image.
        - dim_size: Size of x and y dimensions used to standardize features
                    (e.g. dim_size: 28 -> image size: 28 x 28)
    Returns:
        - pixel_arr (ndarray): Pixel values for the image.
    Note: I do not catch errors for reading in images. If an images fails to be
          read by PIL, I would like to know that and I do not want the program
          to continue if that occurs.
    '''
    image = Image.open(path).convert('L').resize((dim_size, dim_size))
    pixel_arr = np.asarray(image)
    return pixel_arr


def format_features(features, batch_size, dim_size, num_classes, train_ratio,
                    output_path):
    '''
    Summary: Creates CSV files for batches and test features containing pixel
             data from each image and its corresponding classification.
    Args:
        - features: List containing path and object class for each image.
        - batch_size: Size of batch.
        - dim_size: Size of x and y dimensions used to standardize features
                    (e.g. dim_size: 28 -> image size: 28 x 28).
        - num_classes: Number of object classes.
        - train_ratio: Ratio used to split list of all features into a testing
                       and training set.
        - output_path: Path to output test features and batches CSV files.
    Note: Number of training features will always be slightly above
          (feature_size). This is done to make the number of training features
          divisible by the batch size which makes loading batches in C++ easier
          and safer (b/c all batches will have the same number of features.).
    '''
    features_size = len(features)

    train_size = ceil(features_size * train_ratio)
    train_mod_batch = train_size % batch_size
    if train_mod_batch != 0:
        train_size = int(train_size + batch_size - train_mod_batch)
    num_batches = int(train_size / batch_size)

    if train_size % batch_size != 0:
        raise ValueError('Number of train features could not be made ' +
                         'divisible by the batch_size.\n' +
                         'Number of features: ' + str(features_size) + '\n' +
                         'Number of train features: ' + str(train_size) + '\n' +
                         'Batch size: ' + str(batch_size) + '\n' +
                         'Train ratio: ' + str(train_ratio) + '\n')

    random.shuffle(features)

    batch_dir = os.path.join(output_path, BATCH_DIR)

    if not os.path.isdir(batch_dir):
        os.mkdir(batch_dir)

    # Create dataset init file
    dataset_init_file = os.path.join(output_path, 'dataset_init.csv')
    with open(dataset_init_file, mode='w+') as file:
        file.write('batch_size,{}\n'.format(batch_size))
        file.write('num_batches,{}\n'.format(num_batches))
        file.write('num_test_features,{}\n'.format(features_size - train_size))
        file.write('num_classes,{}\n'.format(num_classes))
        file.write('dim_size,{}\n'.format(dim_size))

    # Create batch files
    for batch_num in range(0, num_batches):
        batch_file = os.path.join(batch_dir, 'batch_{}.csv'.format(batch_num))
        with open(batch_file, mode='w+') as file:
            for batch_feature in range(0, batch_size):
                feature_num = batch_num * batch_feature + batch_feature
                pixels = read_image(features[feature_num]['path'], dim_size)
                for x in range(0, dim_size):
                    for y in range(0, dim_size):
                        if (x + 1) * (y + 1) < dim_size * dim_size:
                            file.write('{},'.format(pixels.item((x, y))))
                        else:
                            file.write('{}\n'.format(pixels.item((x, y))))
                file.write('{}\n'.format(features[feature_num]['class']))

    # Create test features file
    test_file = os.path.join(output_path, 'test_features.csv')
    with open(test_file, mode='w+') as file:
        for test_num in range(train_size, features_size):
            pixels = read_image(features[test_num]['path'], dim_size)
            for x in range(0, dim_size):
                for y in range(0, dim_size):
                    if (x + 1) * (y + 1) < dim_size * dim_size:
                        file.write('{},'.format(pixels.item((x, y))))
                    else:
                        file.write('{}\n'.format(pixels.item((x, y))))
            file.write('{}\n'.format(features[test_num]['class']))


def create_records(input_dir, batch_size, dim_size, train_ratio):
    '''
    Summary: Determines output path for formatted CSV files and calls other
             methods that handle the actual formatting of object labels,
             batches, and testing features.
    Args:
        - input_dir: Input directory for dataset.
        - batch_size: Size of batches.
        - batch_size: Size of batch.
        - dim_size: Size of x and y dimensions used to standardize features
                    (e.g. dim_size: 28 -> image size: 28 x 28)
        - train_ratio: Ratio used to split list of all features into a testing
                       and training set.
    '''
    print('Input Directory: ' + input_dir + '\n' +
          'Batch Size: ' + str(batch_size) + '\n' +
          'Dimension Size: ' + str(dim_size) + '\n' +
          'Train Ratio: ' + str(train_ratio) + '\n')

    input_abs_path = os.path.abspath(input_dir)
    input_parent_path = Path(input_abs_path).parents[0]
    output_path = os.path.join(input_parent_path, OUTPUT_DIR)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    objects, features = get_input_files(input_dir)

    object_labels(objects, output_path)

    format_features(features, batch_size, dim_size, len(objects), train_ratio,
                    output_path)


if __name__ == "__main__":
    try:
        opts, args = getopt(sys.argv[1:], 'i:b:d:r:')
    except GetoptError as e:
        print(str(e))
        print('Usage:\n' +
              '-i : Specify input directory that contains dataset images. ' +
              '(REQUIRED)\n' +
              '-b : Batch size. (REQUIRED)\n' +
              '-d : Image dimension (e.g. -d 28: All images sizes will be ' +
              'formatted to size 28 x 28). (REQUIRED)\n' +
              '-r : Train to test set ratio (e.g. -r 8: 80% of images in ' +
              'input directory will be designated to the training set). ' +
              'Valid input includes: [ 1, 2, 3, 4, 5, 6, 7, 8, 9]'
              '(DEFAULT: 7)\n\n' +
              'Example:\n' +
              'python create_records.py -i <input directory> -b 25 -r .8\n')
        sys.exit(2)

    input_dir = ''
    batch_size = ''
    dim_size = ''
    train_ratio = 7

    for opt, arg in opts:
        if opt == '-i':
            input_dir = str(arg)
        elif opt == '-b':
            batch_size = int(arg)
        elif opt == '-d':
            dim_size = int(arg)
        elif opt == '-r':
            train_ratio = int(arg)

    if (not os.path.isdir(input_dir) or input_dir == ''):
        print('Input directory provided does not exist.\n' +
              'Directory entered: ' + str(input_dir))
        sys.exit(2)

    if batch_size == '' or batch_size <= 0:
        print('You must specify a batch size greater than 0.\n' +
              'Batch size entered: ' + str(batch_size))
        sys.exit(2)

    if dim_size == '' or dim_size <= 0:
        print('You must specify a dimension size greater than 0.\n' +
              'Dimension size entered: ' + str(dim_size))
        sys.exit(2)

    if train_ratio < 1 or train_ratio > 9:
        print('Invalid train ratio value specified.\n' +
              'Train ratio entered: ' + str(train_ratio))
        sys.exit(2)

    create_records(input_dir, batch_size, dim_size, train_ratio * .1)
