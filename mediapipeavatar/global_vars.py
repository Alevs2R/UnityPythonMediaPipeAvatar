# Internally used, don't mind this.
KILL_THREADS = False

# Toggle this in order to view how your WebCam is being interpreted (reduces performance).
DEBUG = True 

# Change UDP connection settings (must match Unity side)
USE_LEGACY_PIPES = False # Only supported on Windows (if True, use NamedPipes rather than UDP sockets)
HOST = '127.0.0.1'
PORT = 52733

# Settings do not universally apply, not all WebCams support all frame rates and resolutions
CAM_INDEX = "human_moving.mp4" # OpenCV2 webcam index, try changing for using another (ex: external) webcam.
PLAY_FROM_FILE = True
USE_CUSTOM_CAM_SETTINGS = True
FPS = 30
WIDTH = 848
HEIGHT = 480

# [0, 2] Higher numbers are more precise, but also cost more performance. The demo video used 2 (good environment is more important).
MODEL_COMPLEXITY = 1