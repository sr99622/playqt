# playqt
Windows GUI Version of ffplay integrated with darknet neural network.
Compiled with Qt Creator using Qt version 6.0.2 which in turn
depends upon MS Visual Studio 2019.

**WORK IN PROGRESS**

To compile, you will need the contrib libraries found at 

https://sourceforge.net/projects/playqt/files/

The build is set up to find the contrib libraries using the
environment variable CONTRIB_PATH pointing to the location
where the libraries were unzipped.

You will need to have NVIDIA cuda gpu development toolkit set up as well.
The build will look for the evironment variable CUDA_PATH which should
be set up by the toolkit install.

https://developer.nvidia.com/cuda-toolkit

The program will respond to the same commands as ffplay from the prompt.
Click the play button to start playback.  The Engage Filter check box will 
not work for you unless you configure Tools->Model Options with a valid 
model weights file etc.

Darknet model code was compiled from a fork of AlexeyAB/darknet

https://github.com/sr99622/darknet

Several libraries are derived from the excellent Shift Media Project

https://github.com/ShiftMediaProject