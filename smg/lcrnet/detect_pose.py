""" LCR-Net: Localization-Classification-Regression for Human Pose
Copyright (C) 2017 Gregory Rogez & Philippe Weinzaepfel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>"""

from __future__ import print_function
import os, sys
import numpy as np
import cv2


import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
sys.path.insert(0, os.path.join( os.path.dirname(__file__),"Detectron.pytorch/lib"))
from core.config import cfg
import utils.blob as blob_utils

from lcrnet_model import LCRNet


def _get_blobs(im, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    return blobs, im_scale


def detect_pose(img_output_list, anchor_poses, njts, model: LCRNet):
    """
    detect poses in a list of image
    img_output_list: list of couple (path_to_image, path_to_outputfile)
    ckpt_fname: path to the model weights
    cfg_dict: directory of configuration
    anchor_poses: file containing the anchor_poses or directly the anchor poses
    njts: number of joints in the model
    gpuid: -1 for using cpu mode, otherwise device_id
    """
    NT = 5  # 2D + 3D

    output = []
    # iterate over image
    for i, im in enumerate(img_output_list):
        print(f"processing image {i}")
        # prepare the blob
        inputs, im_scale = _get_blobs(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE) # prepare blobs

        # forward
        if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN: 
            _add_multilevel_rois_for_test(inputs, 'rois') # Add multi-level rois for FPN
        if cfg.PYTORCH_VERSION_LESS_THAN_040: # forward
            inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
            inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
            return_dict = model(**inputs)
        else:
            inputs['data'] = [torch.from_numpy(inputs['data'])]
            inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]
            with torch.no_grad():
              return_dict = model(**inputs)
        # get boxes
        rois = return_dict['rois'].data.cpu().numpy()
        boxes = rois[:, 1:5] / im_scale
        # get scores
        scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
        scores = scores.reshape([-1, scores.shape[-1]]) # In case there is 1 proposal
        # get pose_deltas
        pose_deltas = return_dict['pose_pred'].data.cpu().numpy().squeeze()
        # project poses on boxes
        boxes_size = boxes[:,2:4]-boxes[:,0:2]
        offset = np.concatenate( ( boxes[:,:2], np.zeros((boxes.shape[0],3),dtype=np.float32)), axis=1) # x,y top-left corner for each box
        scale = np.concatenate( ( boxes_size[:,:2], np.ones((boxes.shape[0],3),dtype=np.float32)), axis=1) # width, height for each box
        offset_poses = np.tile( np.concatenate( [np.tile( offset[:,k:k+1], (1,njts)) for k in range(NT)], axis=1), (1,anchor_poses.shape[0])) # x,y top-left corner for each pose 
        scale_poses = np.tile( np.concatenate( [np.tile( scale[:,k:k+1], (1,njts)) for k in range(NT)], axis=1), (1,anchor_poses.shape[0]))
            # x- y- scale for each pose 
        pred_poses = offset_poses + np.tile( anchor_poses.reshape(1,-1), (boxes.shape[0],1) ) * scale_poses # put anchor poses into the boxes
        pred_poses += scale_poses * pose_deltas[:,njts*NT:] # apply regression (do not consider the one for the background class)
        
        # we save only the poses with score over th with at minimum 500 ones 
        th = 0.1/(scores.shape[1]-1)
        Nmin = min(500, scores[:,1:].size-1)
        if np.sum( scores[:,1:]>th ) < Nmin: # set thresholds to keep at least Nmin boxes
            th = - np.sort( -scores[:,1:].ravel() )[Nmin-1]
        where = list(zip(*np.where(scores[:,1:]>=th ))) # which one to save 
        nPP = len(where) # number to save 
        regpose2d = np.empty((nPP,njts*2), dtype=np.float32) # regressed 2D pose 
        regpose3d = np.empty((nPP,njts*3), dtype=np.float32) # regressed 3D pose 
        regscore = np.empty((nPP,1), dtype=np.float32) # score of the regressed pose 
        regprop = np.empty((nPP,1), dtype=np.float32) # index of the proposal among the candidate boxes 
        regclass = np.empty((nPP,1), dtype=np.float32) # index of the anchor pose class 
        for ii, (i,j) in enumerate(where):
            regpose2d[ii,:] = pred_poses[i, j*njts*5:j*njts*5+njts*2]
            regpose3d[ii,:] = pred_poses[i, j*njts*5+njts*2:j*njts*5+njts*5]
            regscore[ii,0] = scores[i,1+j]
            regprop[ii,0] = i+1
            regclass[ii,0] = j+1
        tosave = {'regpose2d': regpose2d, 
                  'regpose3d': regpose3d, 
                  'regscore': regscore, 
                  'regprop': regprop, 
                  'regclass': regclass, 
                  'rois': boxes,
                 }
        output.append( tosave )

    return output
