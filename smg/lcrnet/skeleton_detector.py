import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

import smg.external.lcrnet.scene as scene

from collections import OrderedDict
from PIL import Image
from typing import Any, Dict, List, Tuple

from smg.external.lcrnet.lcr_net_ppi import LCRNet_PPI

# FIXME: Make importing from Detectron.pytorch cleaner.
from utils.collections import AttrDict

# FIXME: This is bad, and should be refactored out gradually.
from .detect_pose import *

from .skeleton import Skeleton


class SkeletonDetector:
    """A 3D skeleton detector based on LCR-Net."""

    # CONSTRUCTOR

    def __init__(self, model_name: str = "InTheWild-ResNet50"):
        self.__gpuid: int = 0
        self.__model_dir: str = os.path.join(os.path.dirname(__file__), "../external/lcrnet/models")
        self.__model_name: str = model_name

        self.__anchor_poses: np.ndarray = self.__load_pickle("anchor_poses")
        self.__cfg: AttrDict = self.__load_pickle("cfg")
        self.__model: OrderedDict = torch.load(os.path.join(self.__model_dir, f"{model_name}_model.pth.tgz"))
        self.__ppi_params: Dict[str, Any] = self.__load_pickle("ppi_params")

        self.__K: int = self.__anchor_poses.shape[0]
        self.__njts: int = self.__anchor_poses.shape[1] // 5  # 5 = 2D + 3D

        self.__projmat: np.ndarray = np.load(
            os.path.join(os.path.dirname(__file__), "../external/lcrnet/standard_projmat.npy")
        )

        self.__net: LCRNet = make_model(self.__model, self.__cfg, self.__njts, self.__gpuid)

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 3D skeletons in an RGB image using LCR-Net.

        :param image:   The RGB image.
        :return:        A tuple consisting of the detected 3D skeletons and the LCR-Net visualisation of what
                        it detected.
        """
        res = detect_pose([image], self.__anchor_poses, self.__njts, self.__net)

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

        # show results
        print('displaying results of image ', i)
        SkeletonDetector.__display_poses(image[:, :, [2, 1, 0]], detections, self.__njts)

    # PRIVATE METHODS

    def __load_pickle(self, specifier: str) -> Any:  # TODO: Check the right return type.
        filename: str = os.path.join(self.__model_dir, f"{self.__model_name}_{specifier}.pkl")
        with open(filename, "rb") as f:
            return pickle.load(f)

    # def __make_model(ckpt, cfg_dict, njts: int, gpuid: int) -> LCRNet:
    #     # load the anchor poses and the network
    #     if gpuid >= 0:
    #         assert torch.cuda.is_available(), "You should launch the script on cpu if cuda is not available"
    #         torch.device('cuda:0')
    #     else:
    #         torch.device('cpu')
    #
    #     # load config and network
    #     print('loading the model')
    #     _merge_a_into_b(cfg_dict, cfg)
    #     cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
    #     cfg.CUDA = gpuid >= 0
    #     assert_and_infer_cfg()
    #     model = LCRNet(njts)
    #     if cfg.CUDA: model.cuda()
    #     net_utils.load_ckpt(model, ckpt)
    #     model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])
    #     model.eval()
    #     return model

    # PRIVATE STATIC METHODS

    @staticmethod
    def __display_poses(image, detections, njts):
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

        # plt.show()
        plt.savefig("foo.png")
        points = pose3d.reshape(3, njts).transpose()
        print(points)
