import numpy as np
import os
import pickle
import torch

import smg.external.lcrnet.scene as scene

from collections import OrderedDict
from PIL import Image
from typing import Any, Dict, List, Tuple

from smg.external.lcrnet.demo import display_poses
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

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 3D skeletons in an RGB image using OpenPose.

        :param image:   The RGB image.
        :return:        A tuple consisting of the detected 3D skeletons and the LCR-Net visualisation of what
                        it detected.
        """
        pass

    def detect(self, imagename: str) -> None:
        img_output_list = [(imagename, None)]

        # run lcrnet on a list of images
        model: LCRNet = make_model(self.__model, self.__cfg, self.__njts, self.__gpuid)
        res = detect_pose(img_output_list, self.__anchor_poses, self.__njts, model)

        # projmat = np.load(os.path.join(os.path.dirname(__file__), 'standard_projmat.npy'))
        projMat_block_diag, M = scene.get_matrices(self.__projmat, self.__njts)

        for i, (imname, _) in enumerate(img_output_list):  # for each image
            image = np.asarray(Image.open(imname))
            resolution = image.shape[:2]

            # perform postprocessing
            print('postprocessing (PPI) on image ', imname)
            detections = LCRNet_PPI(res[i], self.__K, resolution, J=self.__njts, **self.__ppi_params)

            # move 3d pose into scene coordinates
            print('3D scene coordinates regression on image ', imname)
            for detection in detections:
                delta3d = scene.compute_reproj_delta_3d(detection, projMat_block_diag, M, self.__njts)
                detection['pose3d'][:  self.__njts] += delta3d[0]
                detection['pose3d'][self.__njts:2 * self.__njts] += delta3d[1]
                detection['pose3d'][2 * self.__njts:3 * self.__njts] -= delta3d[2]

            # show results
            print('displaying results of image ', imname)
            display_poses(image, detections, self.__njts)

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
