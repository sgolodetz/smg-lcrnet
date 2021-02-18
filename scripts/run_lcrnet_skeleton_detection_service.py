# noinspection PyPackageRequirements
import numpy as np

from timeit import default_timer as timer
from typing import Callable, List

from smg.comms.skeletons import SkeletonDetectionService
from smg.lcrnet import SkeletonDetector
from smg.skeletons import Skeleton


def make_frame_processor(skeleton_detector: SkeletonDetector, *, debug: bool = False) -> \
        Callable[[np.ndarray, np.ndarray, np.ndarray], List[Skeleton]]:
    """
    Make a frame processor for a skeleton detection service that forwards to an LCR-Net skeleton detector.

    :param skeleton_detector:   The LCR-Net skeleton detector.
    :param debug:               Whether to print debug messages.
    :return:                    The frame processor.
    """
    # noinspection PyUnusedLocal
    def detect_skeletons(colour_image: np.ndarray, depth_image: np.ndarray,
                         world_from_camera: np.ndarray) -> List[Skeleton]:
        """
        Detect 3D skeletons in an RGB image using LCR-Net.

        :param colour_image:        The RGB image.
        :param depth_image:         Passed in by the skeleton detection service, but ignored.
        :param world_from_camera:   The camera pose.
        :return:                    The detected 3D skeletons.
        """
        if debug:
            start = timer()

        skeletons, _ = skeleton_detector.detect_skeletons(colour_image, world_from_camera)

        if debug:
            end = timer()

            # noinspection PyUnboundLocalVariable
            print(f"Detection Time: {end - start}s")

        return skeletons

    return detect_skeletons


def main() -> None:
    skeleton_detector: SkeletonDetector = SkeletonDetector()
    service: SkeletonDetectionService = SkeletonDetectionService(make_frame_processor(skeleton_detector))
    service.run()


if __name__ == "__main__":
    main()
