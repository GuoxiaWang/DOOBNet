# --------------------------------------------------------
# Copyright (c) 2018 Guoxia Wang
# DOOBNet data augmentation and converting tool
# Include the PIOD and BSDS ownership dataset
# Written by Guoxia Wang
# --------------------------------------------------------

import sys
import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from PIL import ImageFilter
import scipy.ndimage
import cv2
import argparse


plt.rcParams['figure.figsize'] = 20, 5

val_cnt = 0
train_cnt = 0
total_pixel = 0
total_edge_pixel = 0


def PIOD_augmentation(mat_filename, h5_dir, img_src_dir, img_dst_dir, val_list):
    mat = scipy.io.loadmat(mat_filename)
    orient_map = mat['bndinfo_pascal']['OrientMap'][0][0].astype(np.float32)
    height, width = orient_map.shape
    imsize = mat['bndinfo_pascal']['imsize'][0][0][0]
    assert(height == imsize[0] and width == imsize[1])
    if (mat['bndinfo_pascal']['ne'] == 0):
        print('Gt has no edge fragment!')
        return

    global val_cnt
    global train_cnt

    mat_id = os.path.splitext(os.path.split(mat_filename)[1])[0]
    if (mat_id in val_list):
        val_cnt += 1
    else:
        train_cnt += 1

    indices = mat['bndinfo_pascal']['edges'][0][0][0]['indices'][0][0]
    edge_map = np.zeros(imsize, dtype=np.uint8)
    for idx in range(len(indices)):
        pixel_indices = indices[idx]
        # pixel_indices -1 beacause matlab index 1, but python index 0
        pixel_coords = np.unravel_index(pixel_indices-1, imsize, order='F')
        edge_map[pixel_coords] = 1 

    global total_pixel
    global total_edge_pixel
    total_edge_pixel += np.sum(edge_map)
    total_pixel += (height * width)
    # print total_edge_pixel, total_pixel

    label = np.zeros((1, 2,height,width), dtype=np.float32)
    label[0, 0, ...] = edge_map
    label[0, 1, ...] = orient_map

    img_src_filename = os.path.join(img_src_dir, '{}.jpg'.format(mat_id))
    img = Image.open(img_src_filename)

    if (height < 320 or width < 320) and (not (mat_id in val_list)):
        label = gt_rescale(label.copy(), 320)
        img = img_rescale(img, 320)

    img_dst_filename = os.path.join(img_dst_dir, '{}.jpg'.format(mat_id))
    img.save(img_dst_filename, quality=100)
    h5_filename = os.path.join(h5_dir, '{}.h5'.format(mat_id))
    with h5py.File(h5_filename, 'w') as f:
       f['label'] = label  

    if (mat_id in val_list):
        return

    # Data augmentation
    img_dst_filename_flip = os.path.join(img_dst_dir, '{}_flip.jpg'.format(mat_id))
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip.save(img_dst_filename_flip, quality=100)
    label_flip = gt_flip(label)
    h5_filename_flip = os.path.join(h5_dir, '{}_flip.h5'.format(mat_id))
    with h5py.File(h5_filename_flip, 'w') as f:
       f['label'] = label_flip


def gt_flip(edge_ori):
    edge = edge_ori[0, 0, ...]
    ori = edge_ori[0, 1, ...]

    edge = np.fliplr(edge)
    ori = np.fliplr(ori)

    mask = edge == 1
    ori[mask] = -ori[mask]

    edge_ori[0, 0, ...] = edge
    edge_ori[0, 1, ...] = ori

    return edge_ori

def gt_rotate(edge_ori, times):
    edge = edge_ori[0, 0, ...]
    ori = edge_ori[0, 1, ...]

    edge = np.rot90(edge, times)
    ori = np.rot90(ori, times)

    mask = edge == 1
    theta = ori[mask];
    radians = np.radians(90*times)
    theta = theta-radians;
    theta[theta > np.pi] = theta[theta > np.pi] - 2 * np.pi; 
    theta[theta < -np.pi] = theta[theta < -np.pi] + 2 * np.pi; 
    ori[mask] = theta; 

    height, width = edge.shape
    edge_ori = np.zeros((1, 2, height, width), dtype=np.float32)

    edge_ori[0, 0, ...] = edge
    edge_ori[0, 1, ...] = ori

    return edge_ori


def gt_scale(edge_ori, scale):
    edge = edge_ori[0, 0, ...]
    ori = edge_ori[0, 1, ...]

    edge = scipy.ndimage.zoom(edge, scale, order=0) # ordre 0 means NEAREST
    ori = scipy.ndimage.zoom(ori, scale, order=0)

    height, width = edge.shape
    edge_ori = np.zeros((1, 2, height, width), dtype=np.float32)

    edge_ori[0, 0, ...] = edge
    edge_ori[0, 1, ...] = ori

    return edge_ori

def cal_height_width(height, width, shortest_side):
    aspect = width / float(height)
    if (aspect > 1):
        new_width = int(aspect * shortest_side)
        new_height = shortest_side
    elif (aspect < 1):
        new_width = shortest_side
        new_height = int(shortest_side / aspect)
    else:
        new_width = shortest_side
        new_height = shortest_side
    return (new_height, new_width)

def img_rescale(img, shortest_side):
    height, width = img.size
    (new_height, new_width) = cal_height_width(height, width, shortest_side)
    img = img.resize((new_height, new_width), Image.BILINEAR)
    return img

def gt_rescale(edge_ori, shortest_side):
    edge = edge_ori[0, 0, ...]
    ori = edge_ori[0, 1, ...]

    height, width = edge.shape
    (new_height, new_width) = cal_height_width(height, width, shortest_side)

    edge = cv2.resize(edge, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    ori = cv2.resize(ori, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    height, width = edge.shape
    edge_ori = np.zeros((1, 2, height, width), dtype=np.float32)

    edge_ori[0, 0, ...] = edge
    edge_ori[0, 1, ...] = ori

    return edge_ori   

def BSDS__augmentation(mat_path, h5_dir, img_src_dir, img_dst_dir):
    mat_id = os.path.splitext(os.path.split(mat_path)[1])[0]
    mat = scipy.io.loadmat(mat_path)
    gt = mat['gtStruct']['gt_theta'][0][0][0]
    for idx in range(2):
        edge_map = gt[idx][:,:,0]
        ori_map = gt[idx][:,:,1]
        height, width = edge_map.shape
        label = np.zeros((1, 2, height, width), dtype=np.float32)
        label[0, 0, ...] = edge_map
        label[0, 1, ...] = ori_map
        img_src_filename = os.path.join(img_src_dir, '{}.jpg'.format(mat_id))
        img = Image.open(img_src_filename)

        for rot in range(4):
            img_rot = img
            label_rot = label
            if (rot != 0):
                img_rot = img.transpose(Image.ROTATE_90 + rot - 1) #  Image.ROTATE_90 = 2 
                label_rot = gt_rotate(label.copy(), rot)

            for flip in range(2):
                img_rot_flip = img_rot
                label_rot_flip = label_rot
                if (flip > 0):
                    img_rot_flip = img_rot.transpose(Image.FLIP_LEFT_RIGHT)
                    label_rot_flip = gt_flip(label_rot.copy())


                filename = '{}_idx{}_rot{}_flip{}'.format(mat_id, idx, rot*90, flip)
                # print filename 
                img_dst_filename = os.path.join(img_dst_dir, '{}.jpg'.format(filename))
                img_rot_flip.save(img_dst_filename, quality=100)
                h5_filename = os.path.join(h5_dir, '{}.h5'.format(filename))
                with h5py.File(h5_filename, 'w') as f:
                   f['label'] = label_rot_flip                 

def parse_args():
    parser = argparse.ArgumentParser(description='PIOD and BSDS ownership dataset converting and augmenting tool')
    parser.add_argument(
        '--dataset', help="dataset name, PIOD or BSDSownership",
        type=str, required=True)
    parser.add_argument(
        '--label-dir', help="the label directory that contains *.mat label files",
        type=str, required=True)
    parser.add_argument(
        '--img-dir', help="the source image directory",
        type=str, required=True)
    parser.add_argument(
        '--output-dir', help="the directory that save to augmenting images and .h5 label files",
        type=str, required=True)
    parser.add_argument(
        '--bsdsownership-testfg', help="testfg directory, it only require for BSDS ownership dataset",
        type=str, default=None)
    parser.add_argument(
        '--piod-val-list-file', help="the piod val_doc_2010.txt file, it only require for PIOD dataset",
        type=str, default=None)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(-1)
    return parser.parse_args()


def main():
    args = parse_args()
    assert(os.path.exists(args.label_dir))
    assert(os.path.exists(args.img_dir))
    assert(os.path.exists(args.output_dir))

    dst_label_dir = os.path.join(args.output_dir, 'Aug_HDF5EdgeOriLabel')
    dst_img_dir = os.path.join(args.output_dir, 'Aug_JPEGImages')
    if (not os.path.exists(dst_label_dir)):
        os.mkdir(dst_label_dir)
    if (not os.path.exists(dst_img_dir)):
        os.mkdir(dst_img_dir)

    if (args.dataset == 'BSDSownership'):
        print('Start converting and augmenting {} dataset ...'.format(args.dataset))
        if (not args.bsdsownership_testfg):
            print('Require testfg directory. For usage, see --help!')
            sys.exit(-1)
        assert(os.path.exists(args.bsdsownership_testfg))

        mat_list = glob.glob(os.path.join(args.label_dir, '*.mat'))
        for mat_path in mat_list:
            BSDS__augmentation(mat_path, dst_label_dir, args.img_dir, dst_img_dir)    

        # generate train and test file list for training and testing and test ids for matlab evaluation code
        train_val_pair_list = []
        test_list = []
        test_iids = []

        h5_list = glob.glob(os.path.join(dst_label_dir, '*.h5'))
        for h5_path in h5_list:
            h5_filename = os.path.split(h5_path)[1]
            h5_id = os.path.splitext(h5_filename)[0]

            # here use abspath such that you can move train and test list file to anywhere you like
            img_path = os.path.join(os.path.abspath(dst_img_dir), '{}.jpg'.format(h5_id))
            gt_path = os.path.join(os.path.abspath(dst_label_dir), h5_filename)
            train_val_pair_list.append((img_path, gt_path))
        
        mat_list = glob.glob(os.path.join(args.bsdsownership_testfg, '*.mat'))
        for mat_path in mat_list:
            mat_filename = os.path.split(mat_path)[1]
            iid = os.path.splitext(mat_filename)[0]
            img_path = os.path.join(os.path.abspath(args.img_dir), '{}.jpg'.format(iid))
            test_list.append(img_path)
            test_iids.append(iid)

        # save to file
        with open(os.path.join(args.output_dir, 'train_pair.lst'), 'w') as f:
            for img_path, gt_path in train_val_pair_list:
                f.write('{} {}\n'.format(img_path, gt_path))
        print('Write train list to {}.'.format(os.path.join(args.output_dir, 'train_pair.lst')))

        with open(os.path.join(args.output_dir, 'test.lst'), 'w') as f:
            for img_path in test_list:
                f.write('{}\n'.format(img_path))
        print('Write test list to {}.'.format(os.path.join(args.output_dir, 'test.lst')))

        with open(os.path.join(args.output_dir, 'test_ori_iids.lst'), 'w') as f:
            for iid in test_iids:
                f.write('{}\n'.format(iid))
        print('Write test ids to {}.'.format(os.path.join(args.output_dir, 'test_ori_iids.lst')))


    elif (args.dataset  == 'PIOD'):
        val_list = []
        if (not args.piod_val_list_file):
            print('Require val_doc_2010.txt. For usage, see --help!')
            sys.exit(-1)

        assert(os.path.exists(args.piod_val_list_file))
        with open(args.piod_val_list_file) as f:
            for line in f:
                val_list.append(line.strip())

        mat_list = glob.glob(os.path.join(args.label_dir, '*.mat'))
        print('Start converting and augmenting {} dataset ...'.format(args.dataset))
        for mat_path in mat_list:
            PIOD_augmentation(mat_path, dst_label_dir, args.img_dir, dst_img_dir, val_list)

        print('train: {} val: {}'.format(train_cnt, val_cnt))
        print('boundary pixel rate: {}'.format(float(total_edge_pixel)/total_pixel))

        train_val_pair_list = []
        test_list = []

        h5_list = glob.glob(os.path.join(dst_label_dir, '*.h5'))
        for h5_path in h5_list:
            h5_filename = os.path.split(h5_path)[1]
            h5_id = os.path.splitext(h5_filename)[0]
            img_path = os.path.join(os.path.abspath(dst_img_dir), '{}.jpg'.format(h5_id))
            if (h5_id in val_list):
                test_list.append(img_path)
            else:
                gt_path = os.path.join(os.path.abspath(dst_label_dir), h5_filename)
                train_val_pair_list.append((img_path, gt_path))
        with open(os.path.join(args.output_dir, 'train_pair_320x320.lst'), 'w') as f:
            for img_path, gt_path in train_val_pair_list:
                f.write('{} {}\n'.format(img_path,  gt_path))
        print('Write train list to {}.'.format(os.path.join(args.output_dir, 'train_pair_320x320.lst')))

        with open(os.path.join(args.output_dir, 'test.lst'), 'w') as f:
            for img_path in test_list:
                f.write('{}\n'.format(img_path))
        print('Write test list to {}.'.format(os.path.join(args.output_dir, 'test.lst')))

    print('Down!')


if __name__ == '__main__':
    main()
