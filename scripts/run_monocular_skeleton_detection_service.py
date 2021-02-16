import numpy as np
import time

from timeit import default_timer as timer
from typing import List

from smg.comms.skeletons import SkeletonDetectionService
from smg.lcrnet import SkeletonDetector
from smg.skeletons import Skeleton


def frame_processor(skeleton_detector: SkeletonDetector):
    def inner(colour_image: np.ndarray, _, world_from_camera: np.ndarray) -> List[Skeleton]:
        start = timer()
        skeletons, _ = skeleton_detector.detect_skeletons(colour_image, world_from_camera)
        end = timer()
        print(f"Detection Time: {end - start}s")
        return skeletons
    return inner


def main() -> None:
    skeleton_detector: SkeletonDetector = SkeletonDetector(debug=True)

    # import cv2
    # image: np.ndarray = cv2.imread("D:/LCRNet_v2.0/skeleton.png")
    # world_from_camera: np.ndarray = np.eye(4)
    # while True:
    #     skeletons = frame_processor(skeleton_detector)(image, image, world_from_camera)

    with SkeletonDetectionService(frame_processor=frame_processor(skeleton_detector)) as service:
        service.start()
        while not service.should_terminate():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
