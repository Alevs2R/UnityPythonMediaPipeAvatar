# MediaPipe Body
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2    
from clientUDP import ClientUDP

import cv2
import threading
import time
import global_vars 
import struct
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


# the body thread actually does the 
# processing of the captured images, and communication with unity
class BodyThread(threading.Thread):
    data = ""
    dirty = True
    pipe = None
    timeSinceCheckedConnection = 0
    timeSincePostStatistics = 0
    frame_timestamp_ms_old = -1
    frame_timestamp_ms = 0

    def run(self):
        self.setup_comms()
        
        self.cap = cv2.VideoCapture(global_vars.CAM_INDEX) # sometimes it can take a while for certain video captures
        
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,global_vars.HEIGHT)

        time.sleep(1)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
        running_mode=VisionRunningMode.VIDEO)   

        with PoseLandmarker.create_from_options(options) as landmarker:
                
            while not global_vars.KILL_THREADS and self.cap.isOpened():
                ti = time.time()

                ret, image = self.cap.read()
                self.frame_timestamp_ms += 1000.0 / global_vars.FPS

                if image is None:
                    continue


                # Image transformations and stuff
                image = cv2.flip(image, 1)
                image.flags.writeable = global_vars.DEBUG

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                
                # Detections
                if self.frame_timestamp_ms_old == self.frame_timestamp_ms:
                    continue
                self.frame_timestamp_ms_old = self.frame_timestamp_ms
                        
                results = landmarker.detect_for_video(mp_image, int(self.frame_timestamp_ms))

                tf = time.time()
                
                # Rendering results
                if global_vars.DEBUG:
                    # if time.time()-self.timeSincePostStatistics>=1:
                    print("Detection fps: %f"%(1/(tf-ti)))
                    self.timeSincePostStatistics = time.time()
                        
                    annotated_image = draw_landmarks_on_image(image, results)

                    cv2.imshow('Body Tracking', cv2.flip(annotated_image,1))
                    cv2.waitKey(1)

                # Set up data for relay
                landmark_pack = lambda i,landmark: "{}|{}|{}|{}".format(i,landmark.x,landmark.y,landmark.z)
                self.data = ''
                if results.pose_world_landmarks:
                    # print("zero landmark",results.pose_world_landmarks[0])
                    packed_landmarks = [landmark_pack(i,l) for i,l in enumerate(results.pose_world_landmarks[0])]
                    self.data = '\n'.join(packed_landmarks)

                self.send_data(self.data)
                    
        self.pipe.close()
        self.cap.release()
        cv2.destroyAllWindows()
        pass

    def setup_comms(self):
        if not global_vars.USE_LEGACY_PIPES:
            self.client = ClientUDP(global_vars.HOST,global_vars.PORT)
            self.client.start()
        else:
            print("Using Pipes for interprocess communication (not supported on OSX or Linux).")
        pass      

    def send_data(self,message):
        if not global_vars.USE_LEGACY_PIPES:
            self.client.sendMessage(message)
            pass
        else:
            # Maintain pipe connection.
            if self.pipe==None and time.time()-self.timeSinceCheckedConnection>=1:
                try:
                    self.pipe = open(r'\\.\pipe\UnityMediaPipeBody1', 'r+b', 0)
                except FileNotFoundError:
                    print("Waiting for Unity project to run...")
                    self.pipe = None
                self.timeSinceCheckedConnection = time.time()

            if self.pipe != None:
                try:     
                    s = self.data.encode('utf-8') 
                    self.pipe.write(struct.pack('I', len(s)) + s)   
                    self.pipe.seek(0)    
                except Exception as ex:  
                    print("Failed to write to pipe. Is the unity project open?")
                    self.pipe= None
        pass
                        

thread = BodyThread()
thread.start()