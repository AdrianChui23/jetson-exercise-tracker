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
arm_raised = False
count_of_left_arm_movement = 0
left_arm_pos = 0
left_prev_arm_pos = 0

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    if len(poses) == 0:
        font.OverlayText(img, text=f"Body is not detected.",
                    x=5, y=5 + (font.GetSize()),
                    color=font.White, background=font.Gray40)


    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)


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


        if left_shoulder_idx < 0 or left_elbow_idx < 0 or left_wrist_idx < 0:
            font.OverlayText(img, text=f"I can't see your whole left arm.",
                    x=5, y=5 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
            continue

        left_prev_arm_pos = left_arm_pos
        print(left_arm_pos, count_of_left_arm_movement)


        left_wrist = pose.Keypoints[left_wrist_idx]
        left_shoulder = pose.Keypoints[left_shoulder_idx]

        left1_x = left_shoulder.x - left_wrist.x
        left1_y = left_shoulder.y - left_wrist.y

        if left_shoulder_idx > 0 and left_wrist_idx > 0:
            if left1_y > 0:
                if not arm_raised:
                    start = timer()
                    arm_raised = True
                    task_completed = False
                
            else:
                arm_raised = False
                font.OverlayText(img, text=f"Raise your arm please",
                    x=5, y=5 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
            if arm_raised:
                elasped_time = timer() - start
                if elasped_time > 5:
                    font.OverlayText(img, text=f"Lower your left arm.",
                    x=5, y=50 + (font.GetSize()),
                    color=font.White, background=font.Gray40) 
                    if not task_completed:
                        count_of_left_arm_movement += 1
                        
                    task_completed = True
                else:
                    font.OverlayText(img, text=f"Lift the arm for { start +5 -timer()} seconds.",
                        x=5, y=50 + (font.GetSize()),
                        color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"arm is raised: {count_of_left_arm_movement}",
                    x=500, y=5 + (font.GetSize()),
                    color=font.White, background=font.Gray40)
        else:
            continue



        right_wrist_idx = pose.FindKeypoint('right_wrist')
        right_shoulder_idx = pose.FindKeypoint('right_shoulder')


        if right_wrist_idx < 0 or right_shoulder_idx < 0:
            continue


        right_wrist = pose.Keypoints[right_wrist_idx]
        right_shoulder = pose.Keypoints[right_shoulder_idx]

        right1_x = right_shoulder.x - right_wrist.x
        right1_y = right_shoulder.y - right_wrist.y

        right_eye_idx = pose.FindKeypoint('right_eye')
        right_ear_idx = pose.FindKeypoint('right_ear')

        if right_eye_idx < 0 or right_ear_idx < 0:
            continue


        right_eye = pose.Keypoints[right_eye_idx]
        right_ear = pose.Keypoints[right_ear_idx]

        right2_x = right_eye.x - right_ear.x
        right2_y = right_eye.y - right_ear.y

        right02_x = right_ear.x
        right002_x = right_eye.x
        right02_y = right_ear.y
        right002_y = right_eye.y






    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
