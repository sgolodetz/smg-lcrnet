import cv2
import numpy as np

from pprint import pprint
from timeit import default_timer as timer
from typing import Optional

from smg.lcrnet import Skeleton, SkeletonDetector


def main() -> None:
    np.set_printoptions(suppress=True)
    skeleton_detector: SkeletonDetector = SkeletonDetector(model_name="DEMO_ECCV18")
    frame_idx: int = 0
    while True:
        filename: str = f"C:/smglib/smg-mapping/output-skeleton/frame-{frame_idx:06d}.color.png"
        image: Optional[np.ndarray] = cv2.imread(filename)
        if image is None:
            break

        start = timer()
        skeletons, output_image = skeleton_detector.detect_skeletons(image, visualise=False)
        end = timer()
        print(f"Time: {end - start}s")

        pprint(skeletons)

        cv2.imshow("Output Image", output_image)
        cv2.waitKey(1)
        frame_idx += 1


if __name__ == "__main__":
    main()
