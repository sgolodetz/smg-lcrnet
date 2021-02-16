import cv2
import numpy as np

from typing import List

from smg.comms.skeletons import SkeletonDetectionService
from smg.lcrnet import SkeletonDetector
from smg.skeletons import Skeleton


def frame_processor(skeleton_detector: SkeletonDetector):
    def inner(colour_image: np.ndarray, _, world_from_camera: np.ndarray) -> List[Skeleton]:
        # cv2.imshow("Colour Image", colour_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        skeletons, _ = skeleton_detector.detect_skeletons(colour_image, world_from_camera)
        return skeletons
    return inner


def main() -> None:
    skeleton_detector: SkeletonDetector = SkeletonDetector()

    with SkeletonDetectionService(frame_processor=frame_processor(skeleton_detector)) as service:
        service.start()
        while not service.should_terminate():
            pass


if __name__ == "__main__":
    main()
