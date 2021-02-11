import cv2
import io
import matplotlib.pyplot as plt
import nn as mynn
import numpy as np
import os
import pickle
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import smg.external.lcrnet.scene as scene

# FIXME: Make importing from Detectron.pytorch cleaner.
import utils.blob as blob_utils
import utils.net as net_utils

from collections import OrderedDict
from torch.autograd import Variable
from typing import Any, Dict, List, Tuple

from smg.external.lcrnet.lcrnet_model import LCRNet
from smg.external.lcrnet.lcr_net_ppi import LCRNet_PPI

# FIXME: Make importing from Detectron.pytorch cleaner.
from core.config import assert_and_infer_cfg, cfg, _merge_a_into_b
from utils.collections import AttrDict

from .skeleton import Skeleton


class SkeletonDetector:
    """A 3D skeleton detector based on LCR-Net."""

    # CONSTRUCTOR

    def __init__(self, *, model_name: str = "InTheWild-ResNet50"):
        self.__gpuid: int = 0
        self.__model_dir: str = os.path.join(os.path.dirname(__file__), "../external/lcrnet/models")
        self.__model_name: str = model_name

        # Specify the keypoint names.
        self.__keypoint_names: Dict[int, str] = {
            0: "RAnkle",
            1: "LAnkle",
            2: "RKnee",
            3: "LKnee",
            4: "RHip",
            5: "LHip",
            6: "RWrist",
            7: "LWrist",
            8: "RElbow",
            9: "LElbow",
            10: "RShoulder",
            11: "LShoulder",
            12: "Nose",

            # Virtual keypoints added by smg-lcrnet (to ensure that bones always join keypoints)
            13: "Neck",
            14: "MidHip"
        }

        # Specify which keypoints are joined to form bones.
        self.__keypoint_pairs: List[Tuple[str, str]] = [
            (self.__keypoint_names[i], self.__keypoint_names[j]) for i, j in [
                (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (6, 8), (7, 9), (8, 10), (9, 11), (10, 11),
                (4, 14), (5, 14), (10, 13), (11, 13), (12, 13), (13, 14)
            ]
        ]

        print(self.__keypoint_pairs)

        # Load the model and make the network.
        self.__anchor_poses: np.ndarray = self.__load_pickle("anchor_poses")
        self.__cfg: AttrDict = self.__load_pickle("cfg")
        self.__model: OrderedDict = torch.load(os.path.join(self.__model_dir, f"{model_name}_model.pth.tgz"))
        self.__ppi_params: Dict[str, Any] = self.__load_pickle("ppi_params")

        self.__K: int = self.__anchor_poses.shape[0]
        self.__njts: int = self.__anchor_poses.shape[1] // 5  # 5 = 2D + 3D

        self.__net: LCRNet = SkeletonDetector.__make_model(self.__model, self.__cfg, self.__njts, self.__gpuid)

        # TODO: Comment here.
        self.__projmat: np.ndarray = np.load(
            os.path.join(os.path.dirname(__file__), "../external/lcrnet/standard_projmat.npy")
        )

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_bone_key(keypoint1: Skeleton.Keypoint, keypoint2: Skeleton.Keypoint) -> Tuple[str, str]:
        """
        Make a key that can be used to look up a bone in a dictionary.

        :param keypoint1:   The keypoint at one end of the bone.
        :param keypoint2:   The keypoint at the other end of the bone.
        :return:            The key for the bone.
        """
        # noinspection PyTypeChecker
        return tuple(sorted([keypoint1.name, keypoint2.name]))

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray, *, visualise: bool = False) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 3D skeletons in an RGB image using LCR-Net.

        :param image:       The RGB image.
        :param visualise:   Whether to make the output visualisation.
        :return:            A tuple consisting of the detected 3D skeletons and the output visualisation (if requested).
        """
        res = SkeletonDetector.__detect_pose([image], self.__anchor_poses, self.__njts, self.__net)

        projMat_block_diag, M = scene.get_matrices(self.__projmat, self.__njts)

        i = 0

        resolution = image.shape[:2]

        # perform postprocessing
        print('postprocessing (PPI) on image ', i)
        detections = LCRNet_PPI(res[i], self.__K, resolution, J=self.__njts, **self.__ppi_params)

        # move 3d pose into scene coordinates
        print('3D scene coordinates regression on image ', i)
        for detection in detections:
            delta3d = scene.compute_reproj_delta_3d(detection, projMat_block_diag, M, self.__njts)
            detection['pose3d'][:  self.__njts] += delta3d[0]
            detection['pose3d'][self.__njts:2 * self.__njts] += delta3d[1]
            detection['pose3d'][2 * self.__njts:3 * self.__njts] -= delta3d[2]

        # Make the skeletons.
        skeletons: List[Skeleton] = []
        for detection in detections:
            detected_keypoints: np.ndarray = detection["pose3d"].reshape(3, self.__njts).transpose()
            skeleton_keypoints: Dict[str, Skeleton.Keypoint] = {}

            for i in range(detected_keypoints.shape[0]):
                name: str = self.__keypoint_names[i]
                position: np.ndarray = detected_keypoints[i, :]
                position[0] *= -1
                position[1] *= -1
                skeleton_keypoints[name] = Skeleton.Keypoint(name, position)

            skeleton_keypoints["MidHip"] = Skeleton.Keypoint(
                "MidHip", (skeleton_keypoints["LHip"].position + skeleton_keypoints["RHip"].position) / 2
            )

            skeleton_keypoints["Neck"] = Skeleton.Keypoint(
                "Neck", (skeleton_keypoints["LShoulder"].position + skeleton_keypoints["RShoulder"].position) / 2
            )

            skeletons.append(Skeleton(skeleton_keypoints, self.__keypoint_pairs))

        # If requested, make the output visualisation. Otherwise, just use the input image.
        output_image: np.ndarray = SkeletonDetector.__display_poses(image[:, :, [2, 1, 0]], detections, self.__njts) \
            if visualise else image

        return skeletons, output_image

    # PRIVATE METHODS

    def __load_pickle(self, specifier: str) -> Any:  # TODO: Check the right return type.
        filename: str = os.path.join(self.__model_dir, f"{self.__model_name}_{specifier}.pkl")
        with open(filename, "rb") as f:
            return pickle.load(f)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __detect_pose(img_output_list, anchor_poses, njts, model: LCRNet):
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
            inputs, im_scale = SkeletonDetector.__get_blobs(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)  # prepare blobs

            # forward
            if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
                _add_multilevel_rois_for_test(inputs, 'rois')  # Add multi-level rois for FPN
            if cfg.PYTORCH_VERSION_LESS_THAN_040:  # forward
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
            scores = scores.reshape([-1, scores.shape[-1]])  # In case there is 1 proposal
            # get pose_deltas
            pose_deltas = return_dict['pose_pred'].data.cpu().numpy().squeeze()
            # project poses on boxes
            boxes_size = boxes[:, 2:4] - boxes[:, 0:2]
            offset = np.concatenate((boxes[:, :2], np.zeros((boxes.shape[0], 3), dtype=np.float32)),
                                    axis=1)  # x,y top-left corner for each box
            scale = np.concatenate((boxes_size[:, :2], np.ones((boxes.shape[0], 3), dtype=np.float32)),
                                   axis=1)  # width, height for each box
            offset_poses = np.tile(np.concatenate([np.tile(offset[:, k:k + 1], (1, njts)) for k in range(NT)], axis=1),
                                   (1, anchor_poses.shape[0]))  # x,y top-left corner for each pose
            scale_poses = np.tile(np.concatenate([np.tile(scale[:, k:k + 1], (1, njts)) for k in range(NT)], axis=1),
                                  (1, anchor_poses.shape[0]))
            # x- y- scale for each pose
            pred_poses = offset_poses + np.tile(anchor_poses.reshape(1, -1),
                                                (boxes.shape[0], 1)) * scale_poses  # put anchor poses into the boxes
            pred_poses += scale_poses * pose_deltas[:,
                                        njts * NT:]  # apply regression (do not consider the one for the background class)

            # we save only the poses with score over th with at minimum 500 ones
            th = 0.1 / (scores.shape[1] - 1)
            Nmin = min(500, scores[:, 1:].size - 1)
            if np.sum(scores[:, 1:] > th) < Nmin:  # set thresholds to keep at least Nmin boxes
                th = - np.sort(-scores[:, 1:].ravel())[Nmin - 1]
            where = list(zip(*np.where(scores[:, 1:] >= th)))  # which one to save
            nPP = len(where)  # number to save
            regpose2d = np.empty((nPP, njts * 2), dtype=np.float32)  # regressed 2D pose
            regpose3d = np.empty((nPP, njts * 3), dtype=np.float32)  # regressed 3D pose
            regscore = np.empty((nPP, 1), dtype=np.float32)  # score of the regressed pose
            regprop = np.empty((nPP, 1), dtype=np.float32)  # index of the proposal among the candidate boxes
            regclass = np.empty((nPP, 1), dtype=np.float32)  # index of the anchor pose class
            for ii, (i, j) in enumerate(where):
                regpose2d[ii, :] = pred_poses[i, j * njts * 5:j * njts * 5 + njts * 2]
                regpose3d[ii, :] = pred_poses[i, j * njts * 5 + njts * 2:j * njts * 5 + njts * 5]
                regscore[ii, 0] = scores[i, 1 + j]
                regprop[ii, 0] = i + 1
                regclass[ii, 0] = j + 1
            tosave = {'regpose2d': regpose2d,
                      'regpose3d': regpose3d,
                      'regscore': regscore,
                      'regprop': regprop,
                      'regclass': regclass,
                      'rois': boxes,
                      }
            output.append(tosave)

        return output

    @staticmethod
    def __display_poses(image, detections, njts) -> np.ndarray:
        if njts == 13:
            left = [(9, 11), (7, 9), (1, 3), (3, 5)]  # bones on the left
            right = [(0, 2), (2, 4), (8, 10), (6, 8)]  # bones on the right
            right += [(4, 5), (10, 11)]  # bones on the torso
            # (manually add bone between middle of 4,5 to middle of 10,11, and middle of 10,11 and 12)
            head = 12
        elif njts == 17:
            left = [(9, 11), (7, 9), (1, 3), (3, 5)]  # bones on the left
            right = [(0, 2), (2, 4), (8, 10), (6, 8)]  # bones on the right and the center
            right += [(4, 13), (5, 13), (13, 14), (14, 15), (15, 16), (12, 16), (10, 15),
                      (11, 15)]  # bones on the torso
            head = 16

        fig = plt.figure()

        # 2D
        ax = fig.add_subplot(211)
        ax.imshow(image)
        for det in detections:
            pose2d = det['pose2d']
            score = det['cumscore']
            lw = 2
            # draw green lines on the left side
            for i, j in left:
                ax.plot([pose2d[i], pose2d[j]], [pose2d[i + njts], pose2d[j + njts]], 'g', scalex=None, scaley=None,
                        lw=lw)
            # draw blue linse on the right side and center
            for i, j in right:
                ax.plot([pose2d[i], pose2d[j]], [pose2d[i + njts], pose2d[j + njts]], 'b', scalex=None, scaley=None,
                        lw=lw)
            if njts == 13:  # other bones on torso for 13 jts
                def avgpose2d(a, b, offset=0):  # return the coordinate of the middle of joint of index a and b
                    return (pose2d[a + offset] + pose2d[b + offset]) / 2.0

                ax.plot([avgpose2d(4, 5), avgpose2d(10, 11)],
                        [avgpose2d(4, 5, offset=njts), avgpose2d(10, 11, offset=njts)], 'b', scalex=None, scaley=None,
                        lw=lw)
                ax.plot([avgpose2d(12, 12), avgpose2d(10, 11)],
                        [avgpose2d(12, 12, offset=njts), avgpose2d(10, 11, offset=njts)], 'b', scalex=None, scaley=None,
                        lw=lw)
                # put red markers for all joints
            ax.plot(pose2d[0:njts], pose2d[njts:2 * njts], color='r', marker='.', linestyle='None', scalex=None,
                    scaley=None)
            # legend and ticks
            ax.text(pose2d[head] - 20, pose2d[head + njts] - 20, '%.1f' % (score), color='blue')
        ax.set_xticks([])
        ax.set_yticks([])

        # 3D
        ax = fig.add_subplot(212, projection='3d')
        for i, det in enumerate(detections):
            pose3d = det['pose3d']
            score = det['cumscore']
            lw = 2

            def get_pair(i, j, offset):
                return [pose3d[i + offset], pose3d[j + offset]]

            def get_xyz_coord(i, j):
                return get_pair(i, j, 0), get_pair(i, j, njts), get_pair(i, j, njts * 2)

            # draw green lines on the left side
            for i, j in left:
                x, y, z = get_xyz_coord(i, j)
                ax.plot(x, y, z, 'g', scalex=None, scaley=None, lw=lw)
            # draw blue linse on the right side and center
            for i, j in right:
                x, y, z = get_xyz_coord(i, j)
                ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)
            if njts == 13:  # other bones on torso for 13 jts
                def avgpose3d(a, b, offset=0):
                    return (pose3d[a + offset] + pose3d[b + offset]) / 2.0

                def get_avgpair(i1, i2, j1, j2, offset):
                    return [avgpose3d(i1, i2, offset), avgpose3d(j1, j2, offset)]

                def get_xyz_avgcoord(i1, i2, j1, j2):
                    return get_avgpair(i1, i2, j1, j2, 0), get_avgpair(i1, i2, j1, j2, njts), get_avgpair(i1, i2, j1,
                                                                                                          j2, njts * 2)

                x, y, z = get_xyz_avgcoord(4, 5, 10, 11)
                ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)
                x, y, z = get_xyz_avgcoord(12, 12, 10, 11)
                ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)
            # put red markers for all joints
            ax.plot(pose3d[0:njts], pose3d[njts:2 * njts], pose3d[2 * njts:3 * njts], color='r', marker='.',
                    linestyle='None', scalex=None, scaley=None)
            # score
            ax.text(pose3d[head] + 0.1, pose3d[head + njts] + 0.1, pose3d[head + 2 * njts], '%.1f' % (score),
                    color='blue')
        # legend and ticks
        ax.set_aspect('auto')
        ax.elev = -90
        ax.azim = 90
        ax.dist = 8
        ax.set_xlabel('X axis', labelpad=-5)
        ax.set_ylabel('Y axis', labelpad=-5)
        ax.set_zlabel('Z axis', labelpad=-5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Convert the pyplot figure to a BGR image.
        # See also: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format='rgba')
            buffer.seek(0)
            output_image: np.ndarray = np.reshape(
                np.frombuffer(buffer.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
            )[:, :, [2, 1, 0]]

        plt.close(fig)

        return output_image

    @staticmethod
    def __get_blobs(im, target_scale, target_max_size):
        """Convert an image and RoIs within that image into network inputs."""
        blobs = {}
        blobs['data'], im_scale, blobs['im_info'] = blob_utils.get_image_blob(im, target_scale, target_max_size)
        return blobs, im_scale

    @staticmethod
    def __make_model(ckpt, cfg_dict, njts: int, gpuid: int) -> LCRNet:
        # load the anchor poses and the network
        if gpuid >= 0:
            assert torch.cuda.is_available(), "You should launch the script on cpu if cuda is not available"
            torch.device('cuda:0')
        else:
            torch.device('cpu')

        # load config and network
        print('loading the model')
        _merge_a_into_b(cfg_dict, cfg)
        cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
        cfg.CUDA = gpuid >= 0
        assert_and_infer_cfg()
        model = LCRNet(njts)
        if cfg.CUDA: model.cuda()
        net_utils.load_ckpt(model, ckpt)
        model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])
        model.eval()
        return model
