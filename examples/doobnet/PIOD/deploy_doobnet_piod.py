# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io as sio
import os
import cv2
import time

# Make sure that caffe is on the python path:
caffe_root = '../../../'    
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

data_root = '../../../data/PIOD/Augmentation/'
save_root = '../Output/'
#model = '../Models/doobnet_piod.caffemodel'
model = '../snapshots/doobnet_piod_iter_30000.caffemodel'

deploy_prototxt = 'deploy_doobnet_piod.prototxt'

save_root = os.path.join(save_root, 'PIOD')
if not os.path.exists(save_root):
    os.mkdir(save_root)

with open(data_root+'test.lst') as f:
    test_lst = f.readlines()
    
test_lst = [x.strip() for x in test_lst]


im_lst = []
gt_lst = []

for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    im_lst.append(in_)

#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)
# load net
net = caffe.Net(deploy_prototxt, model , caffe.TEST)

start_time = time.time()
for idx in range(0, len(test_lst)):
    im_ = im_lst[idx]
    im_ = im_.transpose((2, 0, 1))
    
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im_.shape)
    net.blobs['data'].data[...] = im_
    # run net and take argmax for prediction
    net.forward()
    
    edgemap = net.blobs['sigmoid_edge'].data[0][0, :, :]
    orimap = net.blobs['unet1b_ori'].data[0][0, :, :]
    
    edge_ori = {}
    edge_ori['edge'] = edgemap
    edge_ori['ori'] = orimap
    # plt.imshow(edgemap)
    # plt.show()
    cv2.imwrite(save_root + '/' + os.path.split(test_lst[idx])[1].split('.')[0] + '.png', edgemap*255)
    sio.savemat(save_root + '/' + os.path.split(test_lst[idx])[1].split('.')[0] + '.mat', {'edge_ori':edge_ori})

diff_time = time.time() - start_time
print 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst))
