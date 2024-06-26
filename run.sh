#!/usr/bin/env bash
sudo docker run --runtime nvidia -it --rm \
    --network host \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v /etc/nv_tegra_release:/etc/nv_tegra_release \
    -v /tmp/nv_jetson_model:/tmp/nv_jetson_model \
    -v /var/run/dbus:/var/run/dbus \
    -v /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    -v $(pwd):/jetson-exercise-tracker \
    --privileged \
    -v /proc/device-tree/compatible:/proc/device-tree/compatible \
    -v /proc/device-tree/chosen:/proc/device-tree/chosen \
    -v $(pwd)/data:/jetson-inference/data \
    -e DISPLAY=:0 -v /tmp/.X11-unix/:/tmp/.X11-unix \
    --device /dev/video0  --device /dev/video1 --device /dev/gpiochip0 \
    adrian-chui/jetson-exercise-tracker:1.0.0 