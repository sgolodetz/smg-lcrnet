import numpy as np

from typing import Dict, List, Tuple


class Skeleton:
    """A skeleton detected by LCR-Net."""

    # NESTED TYPES

    class Keypoint:
        """A 3D keypoint detected by LCR-Net."""

        # CONSTRUCTOR

        def __init__(self, name: str, position: np.ndarray):
            """
            Construct a keypoint.

            :param name:        The name of the keypoint.
            :param position:    The position of the keypoint.
            """
            self.__name: str = name
            self.__position: np.ndarray = position

        # SPECIAL METHODS

        def __repr__(self) -> str:
            return f"Keypoint({self.__name}, {self.__position})"

        # PROPERTIES

        @property
        def name(self) -> str:
            """
            Get the name of the keypoint.

            :return:    The name of the keypoint.
            """
            return self.__name

        @property
        def position(self) -> np.ndarray:
            """
            Get the position of the keypoint.

            :return:    The position of the keypoint.
            """
            return self.__position

    # CONSTRUCTOR

    def __init__(self, keypoints: Dict[str, Keypoint], keypoint_pairs: List[Tuple[str, str]]):
        """
        Construct a skeleton.

        :param keypoints:       The keypoints that have been detected for the skeleton.
        :param keypoint_pairs:  Pairs of names denoting keypoints that should be joined by bones.
        """
        self.__keypoints: Dict[str, Skeleton.Keypoint] = keypoints
        self.__keypoint_pairs: List[Tuple[str, str]] = keypoint_pairs

    # SPECIAL METHODS

    def __repr__(self) -> str:
        return f"Skeleton({self.__keypoints})"
