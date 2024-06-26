#!/usr/bin/env python3

import sys
import argparse
import time

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaFont
from timeit import default_timer as timer
from adafruit_servokit import ServoKit


kit = ServoKit(channels=16)

def check_body_part_visible(pose, joint1_idx, joint2_idx, joint3_idx, track_idx)->bool:
    joint_tracking(pose, track_idx)
    if joint1_idx < 0 or joint2_idx < 0 or joint3_idx < 0: 
        return False
    return True

def check_body_part_at_exercise(pose, joint1_idx, joint2_idx, joint3_idx, location)->bool:
    if joint1_idx < 0 or joint2_idx < 0 or joint3_idx < 0:
        return False
    
    if location == "upper":
        shoulder = pose.Keypoints[joint1_idx]
        elbow = pose.Keypoints[joint2_idx]
        wrist = pose.Keypoints[joint3_idx]
        return True if shoulder.y > elbow.y else False
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



def joint_tracking(pose, joint_idx):
    if joint_idx < 0:
        return False
    joint = pose.Keypoints[joint_idx]
    print(kit.servo[0].angle, joint.x, joint.y)
    if joint.x > 680:
        kit.servo[0].angle = min(max(0.0, kit.servo[0].angle - 3.0), 180.0)
    elif joint.x < 600 :
        kit.servo[0].angle = min(max(0.0, kit.servo[0].angle + 3.0), 180.0)
    
    if joint.y > 390:
        kit.servo[1].angle = min(max(0.0, kit.servo[1].angle + 3.0), 180.0)
    elif joint.y < 330:
        kit.servo[1].angle = min(max(0.0, kit.servo[1].angle - 3.0), 180.0)


# load the pose estimation model
model = poseNet("resnet18-body", 0, 0.15)

# create video sources & outputs
input = videoSource()
output = videoOutput()
output.SetStatus("Exercise Tracker")

font = cudaFont()
part_at_exercise_previously:bool = False
count_of_body_part_movement = 0

limbs = [{"name": "Left Arm", 
          "joint1": "left_shoulder", 
          "joint2": "left_elbow", 
          "joint3": "left_wrist",
          "location": "upper",
          "track": "left_shoulder"
          }, 
        {"name": "Right Arm", 
          "joint1": "right_shoulder", 
          "joint2": "right_elbow", 
          "joint3": "right_wrist",
          "location": "upper",
          "track": "right_shoulder"
          },
        {"name": "Left Leg", 
          "joint1": "left_hip", 
          "joint2": "left_ankle", 
          "joint3": "left_knee",
          "location": "lower",
          "track": "left_hip"
          },
        {"name": "Right Leg", 
          "joint1": "right_hip", 
          "joint2": "right_ankle", 
          "joint3": "right_knee",
          "location": "lower",
          "track": "right_hip"
          },
        {"name": "Right Torso", 
          "joint1": "right_knee", 
          "joint2": "left_knee", 
          "joint3": "right_shoulder",
          "location": "torso",
          "track": "left_hip"
          },
        {"name": "Left Torso", 
          "joint1": "right_knee", 
          "joint2": "left_knee", 
          "joint3": "left_shoulder",
          "location": "torso",
          "track": "right_hip"
          }]
          

exercises = [{"name": "Lift Left Arm", 
          "body_parts": ["Left Arm"], 
          "duration": 5,
          "repeat":2,
          "return_caption": "Lower Left Arm",
          "description": "Raise Left Arm" }, 
          {"name": "Lift Right Arm", 
          "body_parts": ["Right Arm"], 
          "duration": 5,
          "repeat":2,
          "return_caption": "Lower Right Arm",
          "description": "Raise Right Arm"  },
          {"name": "Lift Both Arms", 
          "body_parts": ["Left Arm", "Right Arm"],
          "duration": 3,
          "repeat": 3 ,
          "return_caption": "Lower Both Arm",
          "description": "Raise Both Arms"},
          {"name": "Lift Left Leg", 
          "body_parts": ["Left Leg"],
          "duration": 3,
          "repeat": 2,
          "return_caption": "Lower Left Leg",
          "description": "Lift Left Leg"},
          {"name": "Lift Right Leg", 
          "body_parts": ["Right Leg"],
          "duration": 3,
          "repeat": 2,
          "return_caption": "Lower Right Leg",
          "description": "Lift Right Leg"},
          {"name": "Rotate Right Torso", 
          "body_parts": ["Right Torso"],
          "duration": 3,
          "repeat": 2,
          "return_caption": "Return Torso to the front",
          "description": "Rotate Right Torso"},
          {"name": "Rotate Left Torso", 
          "body_parts": ["Left Torso"],
          "duration": 3,
          "repeat": 2,
          "return_caption": "Rotate Torso to the front",
          "description": "Rotate Left Torso"}]


current_exercise_index = 0
exercise_completed = False
repeat = -1
kit.servo[0].angle = 90
kit.servo[1].angle = 90

# loop to process each frame 
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # procss the new frame
    poses = model.Process(img)
    
    # get the current exercise
    exercise:dict = exercises[current_exercise_index]

    if repeat < 0:
        repeat = exercise["repeat"]
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
                                        joint3_idx=pose.FindKeypoint(part["joint3"]), 
                                        track_idx=pose.FindKeypoint(part["track"]))
        
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
                if not part_at_exercise_previously: #if this is the first frame when the body part is at the exercising position
                    time_start = timer() 
                    part_at_exercise_previously = True
                    exercise_completed = False                
            else:
                # when the body is not at the exercising position
                part_at_exercise_previously = False
                font.OverlayText(img, text=f"{exercise['description']}",
                    x=5, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40)    
                if exercise_completed:
                    repeat = repeat - 1 
                    if repeat == 0:
                        current_exercise_index = (current_exercise_index + 1) % len(exercises)
                        kit.servo[1].angle = 90
                    exercise_completed = False

            if part_at_exercise_previously:
                elasped_time = timer() - time_start
                if elasped_time > exercise['duration']:
                    font.OverlayText(img, text=f"{exercise['return_caption']}",
                                    x=0, y=50 + (font.GetSize()),
                                    color=font.White, background=font.Gray40) 
                    if not exercise_completed:
                        count_of_body_part_movement += 1       
        
                    exercise_completed = True

                else:
                    font.OverlayText(img, text=f"Holds this position for{time_start + exercise['duration'] - timer(): .0f} seconds.",
                        x=0, y=50 + (font.GetSize()),
                        color=font.White, background=font.Gray40)
                        
    font.OverlayText(img, text=f"Total # exercises: {count_of_body_part_movement}",
            x=0, y=100 + (font.GetSize()),
            color=font.White, background=font.Gray40)
    

    # draw the visual
    output.Render(img)


    if not input.IsStreaming() or not output.IsStreaming():
        break
