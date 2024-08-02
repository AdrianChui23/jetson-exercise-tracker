FROM dustynv/jetson-inference:r32.7.1 
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
RUN python3 get-pip.py
RUN pip3 install adafruit-circuitpython-busdevice==4.3.2   adafruit-circuitpython-servokit==1.3.0 adafruit-circuitpython-register==1.9.12  adafruit-circuitpython-motor==3.3.5 Adafruit-Blinka==6.10.0  Jetson.GPIO face_recognition dlib==19.9.0
RUN apt-get update
RUN apt install qt5-default
RUN apt-get install -y python3-pyqt5.qtmultimedia
#RUN pip3 install PyQt5 
RUN apt-get install -y qttools5-dev-tools libqt5multimedia5-plugins
RUN apt-get install python3-pyqt5
RUN apt-get install pyqt5-dev-tools

WORKDIR /jetson-exercise-tracker 








