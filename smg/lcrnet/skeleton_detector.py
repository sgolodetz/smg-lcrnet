import numpy as np
import os
import pickle
import torch

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from smg.external.lcrnet.detect_pose import detect_pose

# FIXME: Make importing from Detectron.pytorch cleaner.
from utils.collections import AttrDict

from .skeleton import Skeleton


class SkeletonDetector:
    """A 3D skeleton detector based on LCR-Net."""

    # CONSTRUCTOR

    def __init__(self, model_name: str = "InTheWild-ResNet50"):
        self.__model_dir: str = os.path.join(os.path.dirname(__file__), "../external/lcrnet/models")
        self.__model_name: str = model_name

        self.__anchor_poses: np.ndarray = self.__load_pickle("anchor_poses")
        self.__cfg: AttrDict = self.__load_pickle("cfg")
        self.__model: OrderedDict = torch.load(os.path.join(self.__model_dir, f"{model_name}_model.pth.tgz"))
        self.__ppi_params: Dict[str, Any] = self.__load_pickle("ppi_params")

    # PUBLIC METHODS

    def detect_skeletons(self, image: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 3D skeletons in an RGB image using OpenPose.

        :param image:   The RGB image.
        :return:        A tuple consisting of the detected 3D skeletons and the LCR-Net visualisation of what
                        it detected.
        """
        pass

    # PRIVATE METHODS

    def __load_pickle(self, specifier: str) -> Any:  # TODO: Check the right return type.
        filename: str = os.path.join(self.__model_dir, f"{self.__model_name}_{specifier}.pkl")
        with open(filename, "rb") as f:
            return pickle.load(f)
