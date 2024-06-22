#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import time

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaFont
from timeit import default_timer as timer


def check_body_part_visible(pose, joint1_idx, joint2_idx, joint3_idx)->bool:
    if joint1_idx < 0 or joint2_idx < 0 or joint3_idx < 0: 
        return False
    return True

def check_body_part_at_exercise(pose, joint1_idx, joint2_idx, joint3_idx, location)->bool:

    if joint1_idx < 0 or joint2_idx < 0 or joint3_idx < 0:
        return False
    

    if location == "upper":
        wrist = pose.Keypoints[joint3_idx]
        shoulder = pose.Keypoints[joint1_idx]
        return True if (shoulder.y - wrist.y) > 0.0 else False
    elif location == "lower":
        hip = pose.Keypoints[joint1_idx]
        ankle = pose.Keypoints[joint2_idx]
        knee = pose.Keypoints[joint3_idx]

        return True if (abs(hip.y-knee.y) < abs(0.8*(knee.y-ankle.y))) else False
    elif location == "torso":
        knee1 = pose.Keypoints[joint1_idx]
        knee2 = pose.Keypoints[joint2_idx]
        shoulder = pose.Keypoints[joint3_idx]

        return True if (knee1.x > shoulder.x and shoulder.x > knee2.x) or (knee1.x < shoulder.x and shoulder.x < knee2.x) else False


# load the pose estimation model
net = poseNet("resnet18-body", 0, 0.15)

# create video sources & outputs
input = videoSource("/dev/video0")
output = videoOutput()
output.SetStatus("Exercise Tracker")

font = cudaFont()
part_at_exercise_previously:bool = False
count_of_body_part_movement = 0

limbs = [{"name": "Left Arm", 
          "joint1": "left_shoulder", 
          "joint2": "left_elbow", 
          "joint3": "left_wrist",
          "location": "upper"
          }, 
        {"name": "Right Arm", 
          "joint1": "right_shoulder", 
          "joint2": "right_elbow", 
          "joint3": "right_wrist",
          "location": "upper"
          },
        {"name": "Left Leg", 
          "joint1": "left_hip", 
          "joint2": "left_ankle", 
          "joint3": "left_knee",
          "location": "lower"
          },
        {"name": "Right Leg", 
          "joint1": "right_hip", 
          "joint2": "right_ankle", 
          "joint3": "right_knee",
          "location": "lower"},
        {"name": "Right Torso", 
          "joint1": "right_knee", 
          "joint2": "left_knee", 
          "joint3": "right_shoulder",
          "location": "torso"},
        {"name": "Left Torso", 
          "joint1": "right_knee", 
          "joint2": "left_knee", 
          "joint3": "left_shoulder",
          "location": "torso"}]
          

exercises = [{"name": "Lift Left Arm", 
          "body_parts": ["Left Arm"], 
          "duration": 5,
          "repeat":2,
          "caption": "Lower Left Arm",
          "describe": "Left Arm is" }, 
          {"name": "Lift Right Arm", 
          "body_parts": ["Right Arm"], 
          "duration": 5,
          "repeat":2,
          "caption": "Lower Right Arm",
          "describe": "Right Arm is"  },
          {"name": "Lift Both Arms", 
          "body_parts": ["Left Arm", "Right Arm"],
          "duration": 3,
          "repeat": 3 ,
          "caption": "Lower Both Arm",
          "describe": "Both Arms are"},
          {"name": "Lift Left Leg", 
          "body_parts": ["Left Leg"],
          "duration": 3,
          "repeat": 2,
          "caption": "Lower Left Leg",
          "describe": "Left Leg is"},
          {"name": "Lift Right Leg", 
          "body_parts": ["Right Leg"],
          "duration": 3,
          "repeat": 2,
          "caption": "Lower Right Leg",
          "describe": "Right Leg is"},
          {"name": "Rotate Right Torso", 
          "body_parts": ["Right Torso"],
          "duration": 3,
          "repeat": 2,
          "caption": "Rotate Left Torso",
          "describe": "Left Torso is"},
          {"name": "Rotate Right Torso", 
          "body_parts": ["Left Torso"],
          "duration": 3,
          "repeat": 2,
          "caption": "Rotate Left Torso",
          "describe": "Left Torso is"}]


current_exercise_index = 0
exercise_completed = False

# loop to process each frame 
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # procss the new frame
    poses = net.Process(img)
    
    # get the current exercise
    exercise:dict = exercises[current_exercise_index]

    # get the body parts involved
    body_parts = exercise["body_parts"]

    body_part_visible = False
    body_part_at_exercise = False
        

    if len(poses) == 0:
        font.OverlayText(img, text=f"Body is not detected.",
                    x=0, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
        
    font.OverlayText(img, text=f"Current exercise: {exercise['name']} ",
                    x=0, y=0 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
    if len(poses) > 1:
        # when there are more than one body detected
        font.OverlayText(img, text=f"Too many people",
                    x=0, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
    elif len(poses) == 1:  
        pose = poses[0] # get the only body

        # find all the detected key points
        left_wrist_idx = pose.FindKeypoint('left_wrist')
        left_shoulder_idx = pose.FindKeypoint('left_shoulder')
        left_eye_idx = pose.FindKeypoint('left_eye')
        right_ankle_idx = pose.FindKeypoint('right_ankle')
        left_ankle_idx = pose.FindKeypoint('left_ankle')
        left_ear_idx = pose.FindKeypoint('left_ear')
        left_hip_idx = pose.FindKeypoint('left_hip')
        left_knee_idx = pose.FindKeypoint('left_knee')
        left_elbow_idx = pose.FindKeypoint('left_elbow')
        right_ear_idx = pose.FindKeypoint('right_ear')
        right_shoulder_idx = pose.FindKeypoint('right_shoulder')
        right_elbow_idx = pose.FindKeypoint('right_elbow')
        right_hip_idx = pose.FindKeypoint('right_hip')
        right_knee_idx = pose.FindKeypoint('right_knee')
        right_wrist_idx = pose.FindKeypoint('right_wrist')
        right_eye_idx = pose.FindKeypoint('right_eye')
        nose_idx = pose.FindKeypoint('nose')

        # check all the body parts each consist of some keypoints
        for body_part in body_parts:  #body_part is a str
            matches = [x for x in limbs if x["name"] == body_part]
            part = matches[0]
            body_part_visible = check_body_part_visible(pose, joint1_idx=pose.FindKeypoint(part["joint1"]),
                                        joint2_idx=pose.FindKeypoint(part["joint2"]),  
                                        joint3_idx=pose.FindKeypoint(part["joint3"]))
        
            body_part_at_exercise = check_body_part_at_exercise(pose, joint1_idx=pose.FindKeypoint(part["joint1"]),
                                        joint2_idx=pose.FindKeypoint(part["joint2"]),  
                                        joint3_idx=pose.FindKeypoint(part["joint3"]), location=part["location"])
            if not body_part_visible or not body_part_at_exercise:
                break

        # Flag the issue when the whole body part is not visible.
        if not body_part_visible:
            font.OverlayText(img, text=f"{part['name']} is not visible.",
                                x=5, y=50 + (font.GetSize()),
                                color=font.White, background=font.Gray40)
            
        else:          

            if body_part_at_exercise:
                if not part_at_exercise_previously: #if this is the firat frame when the body part is at the exercising position
                    time_start = timer() 
                    part_at_exercise_previously = True
                    exercise_completed = False                
            else:
                # when the body is not at the exercising position
                part_at_exercise_previously = False
                font.OverlayText(img, text=f"{exercise['describe']} not raised ",
                    x=5, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40)    
                if exercise_completed:
                    current_exercise_index = (current_exercise_index+ 1) % len(exercises) 
                    exercise_completed = False

            if part_at_exercise_previously:
                elasped_time = timer() - time_start
                if elasped_time > exercise['duration']:
                    font.OverlayText(img, text=f"{exercise['caption']}",
                                    x=0, y=50 + (font.GetSize()),
                                    color=font.White, background=font.Gray40) 
                    if not exercise_completed:
                        count_of_body_part_movement += 1       
        
                    exercise_completed = True

                else:
                    font.OverlayText(img, text=f"{exercise['describe']} lifted for{time_start + exercise['duration']- timer(): .0f} seconds.",
                        x=0, y=50 + (font.GetSize()),
                        color=font.White, background=font.Gray40)
                        
    font.OverlayText(img, text=f"Total exercise: {count_of_body_part_movement}",
            x=0, y=100 + (font.GetSize()),
            color=font.White, background=font.Gray40)

    # draw the visual
    output.Render(img)


    if not input.IsStreaming() or not output.IsStreaming():
        break
