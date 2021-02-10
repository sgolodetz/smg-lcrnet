import cv2
import os

from smg.lcrnet import Skeleton, SkeletonDetector


def main() -> None:
    skeleton_detector: SkeletonDetector = SkeletonDetector()
    filename: str = os.path.join(os.path.dirname(__file__), "../smg/external/lcrnet/058017637.jpg")
    skeleton_detector.detect_skeletons(cv2.imread(filename))


if __name__ == "__main__":
    main()
