"""Skywalk-framing smoke test: one RGB frame from a fixed elevated camera.

Validates the server/client are healthy and gives a visual baseline to compare
against real skywalk footage (4-speeding/video.mp4, 1920x1080 @ 30 fps).
"""

import os
import queue
import sys

import carla

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
OUT_PATH = os.path.join(OUT_DIR, "carla_first_frame.png")

WIDTH, HEIGHT, FPS, FOV = 1920, 1080, 30, 80
CAM_HEIGHT_M = 8.0
CAM_PITCH_DEG = -30.0
MAP_NAME = "Town10HD"


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    print("server:", client.get_server_version(), "client:", client.get_client_version())

    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("no spawn points", file=sys.stderr)
        return 1
    road = spawn_points[0]

    cam_transform = carla.Transform(
        carla.Location(x=road.location.x, y=road.location.y, z=CAM_HEIGHT_M),
        carla.Rotation(pitch=CAM_PITCH_DEG, yaw=road.rotation.yaw, roll=0.0),
    )

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(WIDTH))
    bp.set_attribute("image_size_y", str(HEIGHT))
    bp.set_attribute("fov", str(FOV))

    camera = world.spawn_actor(bp, cam_transform)
    q: "queue.Queue[carla.Image]" = queue.Queue()
    camera.listen(q.put)

    try:
        for _ in range(10):
            world.tick()
        image = q.get(timeout=10.0)
        image.save_to_disk(OUT_PATH)
        print("saved:", OUT_PATH)
    finally:
        camera.stop()
        camera.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
