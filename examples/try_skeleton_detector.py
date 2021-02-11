import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from pprint import pprint
from timeit import default_timer as timer
from typing import List, Tuple

from smg.lcrnet import Skeleton, SkeletonDetector, SkeletonRenderer
from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.openni import OpenNICamera
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter


def main() -> None:
    np.set_printoptions(suppress=True)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("3D Skeleton Detector")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    skeleton_detector: SkeletonDetector = SkeletonDetector(model_name="DEMO_ECCV18")
    skeletons: List[Skeleton] = []

    # Construct the camera.
    with OpenNICamera(mirror_images=True) as camera:
        # Repeatedly:
        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                # If the user wants us to quit:
                if event.type == pygame.QUIT:
                    # Shut down pygame, and destroy any OpenCV windows.
                    pygame.quit()
                    cv2.destroyAllWindows()

                    # Forcibly terminate the whole process. This isn't graceful, but ORB-SLAM can sometimes
                    # take a long time to shut down, and it's dull to wait for it.
                    # TODO: Update this comment.
                    # noinspection PyProtectedMember
                    os._exit(0)

            colour_image, depth_image = camera.get_images()

            start = timer()
            skeletons, output_image = skeleton_detector.detect_skeletons(colour_image, visualise=False)
            end = timer()
            print(f"Time: {end - start}s")

            pprint(skeletons)

            cv2.imshow("Output Image", output_image)
            cv2.waitKey(1)

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                camera.get_colour_intrinsics(), *window_size
            )):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
                )):
                    # Render a voxel grid.
                    glColor3f(0.0, 0.0, 0.0)
                    OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                    # Render the 3D skeletons.
                    for skeleton in skeletons:
                        SkeletonRenderer.render_skeleton(skeleton)

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
