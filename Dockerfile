FROM dustynv/jetson-inference:r32.7.1 
RUN pip3 install adafruit-circuitpython-busdevice==4.3.2   adafruit-circuitpython-servokit==1.3.0 adafruit-circuitpython-register==1.9.12  adafruit-circuitpython-motor==3.3.5 Adafruit-Blinka==6.10.0  Jetson.GPIO
