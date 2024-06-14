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


def check_arm_visible(pose, shoulder_idx, elbow_idx, wrist_idx)->bool:
    if shoulder_idx < 0 or elbow_idx < 0 or wrist_idx < 0: 
        return False
    return True

def check_arm_raised(pose, shoulder_idx, elbow_idx, wrist_idx)->bool:

    if shoulder_idx < 0 or elbow_idx < 0 or wrist_idx < 0:
        return False
    
    wrist = pose.Keypoints[wrist_idx]
    shoulder = pose.Keypoints[shoulder_idx]

    return True if (shoulder.y - wrist.y) > 0.0 else False



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
left_arm_raised:bool = False
right_arm_raised:bool = False
left_arm_raised_previously:bool = False
right_arm_raised_previously:bool = False
arm_raised_previously:bool = False
count_of_arm_movement = 0

arms = [{"name": "Left Arm", 
        "shoulder": "left_shoulder", 
        "elbow": "left_elbow", 
        "wrist": "left_wrist"}, 
        {"name": "Right Arm", 
        "shoulder": "right_shoulder", 
        "elbow": "right_elbow", 
        "wrist": "right_wrist"}]

current_exercise = 0
task_started = True

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    if len(poses) == 0:
        font.OverlayText(img, text=f"Body is not detected.",
                    x=5, y=5 + (font.GetSize()),
                    color=font.White, background=font.Gray40)

    for pose in poses:
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

        arm = arms[current_exercise]     
        arm_visible = check_arm_visible(pose, shoulder_idx=pose.FindKeypoint(arm["shoulder"]),
                                        elbow_idx=pose.FindKeypoint(arm["elbow"]),  
                                        wrist_idx=pose.FindKeypoint(arm["wrist"]))
        
        arm_raised =  check_arm_raised(pose, shoulder_idx=pose.FindKeypoint(arm["shoulder"]),
                                        elbow_idx=pose.FindKeypoint(arm["elbow"]),  
                                        wrist_idx=pose.FindKeypoint(arm["wrist"]))

        if not arm_visible:
            font.OverlayText(img, text=f"I can't see your entire {arm['name']}.",
                                x=5, y=5 + (font.GetSize()),
                                color=font.White, background=font.Gray40)
        else:            
            if arm_raised:
                if not arm_raised_previously:
                    arm_start = timer()
                    arm_raised_previously = True
                    task_completed = False                
            else:
                arm_raised_previously = False
                font.OverlayText(img, text=f"Raise your {arm['name']} please",
                    x=500, y=5 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
                if not task_started:
                    current_exercise = (current_exercise + 1) % len(arms)
                    task_started = True


            if arm_raised_previously:
                elasped_time = timer() - arm_start
                if elasped_time > 5:
                    font.OverlayText(img, text=f"Lower your arm {arm['name']}",
                                    x=500, y=50 + (font.GetSize()),
                                    color=font.White, background=font.Gray40) 
                    if not task_completed:
                        count_of_arm_movement += 1        
                    task_completed = True
                    task_started = False
 
                else:
                    font.OverlayText(img, text=f"{arm['name']} lifted for {arm_start + 5 - timer(): .0f} seconds.",
                        x=500, y=50 + (font.GetSize()),
                        color=font.White, background=font.Gray40)
                    
            font.OverlayText(img, text=f"{arm['name']} raised: {count_of_arm_movement}",
                    x=500, y=100 + (font.GetSize()),
                    color=font.White, background=font.Gray40)

        
    '''    
        if not left_arm_raised:
            if not right_arm_visible:
                font.OverlayText(img, text=f"I can't see your whole right arm.",
                                 x=5, y=5 + (font.GetSize()),
                                 color=font.White, background=font.Gray40)
                continue
            else:
                if right_arm_raised:
                    if not right_arm_raised_previously :
                        right_arm_start = timer()
                        right_arm_raised_previously = True
                        task_completed = False                
                else:
                    right_arm_raised_previously = False
                    font.OverlayText(img, text=f"Raise your right arm please",
                    x=5, y=5 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
                if right_arm_raised_previously:
                    elasped_time = timer() - right_arm_start
                    if elasped_time > 5:
                        font.OverlayText(img, text=f"Lower your right arm.",
                        x=5, y=50 + (font.GetSize()),
                        color=font.White, background=font.Gray40) 
                        if not task_completed:
                            count_of_right_arm_movement += 1
                        
                        task_completed = True
                    else:
                        font.OverlayText(img, text=f"Right arm lifted for {right_arm_start +5 -timer(): .0f} seconds.",
                            x=5, y=50 + (font.GetSize()),
                            color=font.White, background=font.Gray40)
                font.OverlayText(img, text=f"Right arm raised: {count_of_right_arm_movement}",
                        x=500, y=5 + (font.GetSize()),
                        color=font.White, background=font.Gray40)
    '''
    
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
