# noinspection PyPackageRequirements
import cv2
# noinspection PyPackageRequirements
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

# noinspection PyPackageRequirements
from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.comms.mapping import MappingServer, RGBDFrameMessageUtil, RGBDFrameReceiver
from smg.lcrnet import SkeletonDetector
from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton, SkeletonRenderer
from smg.utility import GeometryUtil, PooledQueue


def main() -> None:
    np.set_printoptions(suppress=True)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("LCR-Net 3D Skeleton Detector")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    # Construct the mapping server.
    with MappingServer(
        frame_decompressor=RGBDFrameMessageUtil.decompress_frame_message,
        pool_empty_strategy=PooledQueue.PES_REPLACE_RANDOM
    ) as server:
        client_id: int = 0
        image_size: Optional[Tuple[int, int]] = None
        intrinsics: Optional[Tuple[float, float, float, float]] = None
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        skeletons_3d: List[Skeleton] = []

        # Construct the skeleton detector.
        skeleton_detector: SkeletonDetector = SkeletonDetector()

        # Start the server.
        server.start()

        while True:
            # Process any PyGame events.
            for event in pygame.event.get():
                # If the user wants us to quit:
                if event.type == pygame.QUIT:
                    # Shut down pygame, and destroy any OpenCV windows.
                    pygame.quit()
                    cv2.destroyAllWindows()

                    # Forcibly terminate the whole process.
                    # noinspection PyProtectedMember
                    os._exit(0)

            # If the server has a frame from the client that has not yet been processed:
            if server.has_frames_now(client_id):
                # Get the camera parameters from the server.
                height, width, _ = server.get_image_shapes(client_id)[0]
                image_size = (width, height)
                intrinsics = server.get_intrinsics(client_id)[0]

                # Get the newest frame from the server.
                server.peek_newest_frame(client_id, receiver)
                colour_image: np.ndarray = receiver.get_rgb_image()
                tracker_w_t_c: np.ndarray = receiver.get_pose()

                # Use LCR-Net to detect 3D skeletons in the colour image.
                start = timer()
                skeletons_3d, visualisation = skeleton_detector.detect_skeletons(
                    colour_image, tracker_w_t_c, visualise=False
                )
                end = timer()
                print(f"Skeleton Detection Time: {end - start}s")

                # Show any visualisation produced during the detection process.
                cv2.imshow("Output Visualisation", visualisation)
                cv2.waitKey(1)

            # Allow the user to control the camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Clear the colour and depth buffers.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Once at least one frame has been received:
            if image_size is not None:
                # Set the projection matrix.
                with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                    GeometryUtil.rescale_intrinsics(intrinsics, image_size, window_size), *window_size
                )):
                    # Set the model-view matrix.
                    with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                        CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
                    )):
                        # Render a voxel grid.
                        glColor3f(0.0, 0.0, 0.0)
                        OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                        # Render the 3D skeletons.
                        for skeleton_3d in skeletons_3d:
                            SkeletonRenderer.render_skeleton(skeleton_3d)

            # Swap the front and back buffers.
            pygame.display.flip()


if __name__ == "__main__":
    main()
