import numpy as np

from timeit import default_timer as timer
from typing import List

from smg.comms.skeletons import SkeletonDetectionService
from smg.lcrnet import SkeletonDetector
from smg.skeletons import Skeleton


def make_frame_processor(skeleton_detector: SkeletonDetector):
    def detect_skeletons(colour_image: np.ndarray, _, world_from_camera: np.ndarray) -> List[Skeleton]:
        start = timer()
        skeletons, _ = skeleton_detector.detect_skeletons(colour_image, world_from_camera)
        end = timer()
        print(f"Detection Time: {end - start}s")
        return skeletons
    return detect_skeletons


def main() -> None:
    skeleton_detector: SkeletonDetector = SkeletonDetector()
    service: SkeletonDetectionService = SkeletonDetectionService(make_frame_processor(skeleton_detector))
    service.run()


if __name__ == "__main__":
    main()
