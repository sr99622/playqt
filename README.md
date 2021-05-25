# playqt
Windows GUI Version of ffplay integrated with darknet neural network.

To compile, you will need the contrib libraries found at 

https://sourceforge.net/projects/playqt/files/

The program will respond to the same commands as ffplay from the prompt.
Click the play button to start playback.  The Engage Filter check box will 
not work for you unless you change the hardcoded file names in model.cpp
and have a valid model weights file etc.

Model code was compiled from a fork of AlexeyAB/darknet

https://github.com/sr99622/darknet