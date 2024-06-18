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


def check_body_part_visible(pose, shoulder_idx, elbow_idx, wrist_idx)->bool:
    if shoulder_idx < 0 or elbow_idx < 0 or wrist_idx < 0: 
        return False
    return True

def check_body_part_raised(pose, shoulder_idx, elbow_idx, wrist_idx)->bool:

    if shoulder_idx < 0 or elbow_idx < 0 or wrist_idx < 0:
        return False
    
    wrist = pose.Keypoints[wrist_idx]
    shoulder = pose.Keypoints[shoulder_idx]

    return True if (shoulder.y - wrist.y) > 0.0 else False

def check_lower_body_part_visible(pose, hip_idx, ankle_idx, knee_idx)->bool:
    if hip_idx < 0 or ankle_idx < 0 or knee_idx < 0: 
        return False
    return True

def check_lower_body_part_raised(pose, hip_idx, ankle_idx, knee_idx)->bool:

    if hip_idx < 0 or ankle_idx < 0 or knee_idx < 0:
        return False
    
    hip = pose.Keypoints[hip_idx]
    ankle = pose.Keypoints[ankle_idx]
    knee = pose.Keypoints[knee_idx]

    legs_y = (hip.y + ankle.y)/2

    return True if legs_y > 0.0 else False

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

font = cudaFont()
left_body_part_raised:bool = False
right_body_part_raised:bool = False
part_raised_previously:bool = False
part_raised_previously:bool = False
part_raised_previously:bool = False
count_of_body_part_movement = 0

limbs = [{"name": "Left Arm", 
        "joint1": "left_shoulder", 
        "joint2": "left_elbow", 
        "joint3": "left_wrist"}, 
        {"name": "Right Arm", 
        "joint1": "right_shoulder", 
        "joint2": "right_elbow", 
        "joint3": "right_wrist"},
        {"name": "Left Leg", 
        "joint1": "left_hip", 
        "joint2": "left_ankle", 
        "joint3": "left_knee"},
        {"name": "Right Leg", 
        "joint1": "right_hip", 
        "joint2": "right_ankle", 
        "joint3": "right_knee"}]

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
          "describe": "Right Leg is"}]

current_exercise_index = 0
exercise_started = False
exercise_completed = True


# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)
    

    exercise:dict = exercises[current_exercise_index]

    body_parts = exercise["body_parts"]

    body_part_visible = False
    body_part_raised = False
        

    if len(poses) == 0:
        font.OverlayText(img, text=f"Body is not detected.",
                    x=0, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
        
    font.OverlayText(img, text=f"Current exercise: {exercise['name']} ",
                    x=0, y=0 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
    if len(poses) > 1:
        font.OverlayText(img, text=f"Too many people",
                    x=0, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
    elif len(poses) == 1:
        pose = poses[0]
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


        for body_part in body_parts:  #body_part is a str
            matches = [x for x in limbs if x["name"] == body_part]
            part = matches[0]
            body_part_visible = check_body_part_visible(pose, shoulder_idx=pose.FindKeypoint(part["joint1"]),
                                        elbow_idx=pose.FindKeypoint(part["joint2"]),  
                                        wrist_idx=pose.FindKeypoint(part["joint3"]))
        
            body_part_raised = check_body_part_raised(pose, shoulder_idx=pose.FindKeypoint(part["joint1"]),
                                        elbow_idx=pose.FindKeypoint(part["joint2"]),  
                                        wrist_idx=pose.FindKeypoint(part["joint3"]))
            lower_body_part_visible  = check_lower_body_part_visible(pose, hip_idx=pose.FindKeypoint(part["joint1"]), 
                                                            ankle_idx=pose.FindKeypoint(part["joint2"]), 
                                                            knee_idx=pose.FindKeypoint(part["joint3"]))
            lower_body_part_raised = check_lower_body_part_raised(pose, hip_idx=pose.FindKeypoint(part["joint1"]), 
                                                            ankle_idx=pose.FindKeypoint(part["joint2"]), 
                                                            knee_idx=pose.FindKeypoint(part["joint3"]))
            if not body_part_visible or not body_part_raised:
                break

        if not body_part_visible:
            font.OverlayText(img, text=f"{part['name']} is not visible.",
                                x=5, y=50 + (font.GetSize()),
                                color=font.White, background=font.Gray40)
            
        else:            
            if body_part_raised:
                if not part_raised_previously:
                    part_start = timer()
                    part_raised_previously = True
                    exercise_completed = False                
            else:
                part_raised_previously = False
                font.OverlayText(img, text=f"{exercise['describe']} not raised ",
                    x=5, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40)    
                if not exercise_started:
                    if exercise_completed:
                        current_exercise_index = (current_exercise_index+ 1) % len(exercises) 
                        exercise_completed = False
                    exercise_started = True

            if part_raised_previously:
                elasped_time = timer() - part_start
                if elasped_time > exercise['duration']:
                    font.OverlayText(img, text=f"{exercise['caption']}",
                                    x=0, y=50 + (font.GetSize()),
                                    color=font.White, background=font.Gray40) 
                    if not exercise_completed:
                        count_of_body_part_movement += 1       
        
                    exercise_completed = True
                    exercise_started = False

                else:
                    font.OverlayText(img, text=f"{exercise['describe']} lifted for{part_start + exercise['duration']- timer(): .0f} seconds.",
                        x=0, y=50 + (font.GetSize()),
                        color=font.White, background=font.Gray40)
                        
    font.OverlayText(img, text=f"Total exercise: {count_of_body_part_movement}",
            x=0, y=100 + (font.GetSize()),
            color=font.White, background=font.Gray40)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
