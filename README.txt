Due to hardware requirements, I created a demo video so you can see how this system works with all of the hardware.
This video can be found on youtube, 
via the link: https://youtu.be/pVVvlTuyg8Y?si=OluCe_zkKUe0nytu


In the python script file titled 'PlantBot', you will find the main code for this software. This code is responsible for the hardware integration to the Kivy user interface, the CNN classifications and all other system computation. This file is the main python script.


In 'EnsembleCNN', you will find the CNN used by the system, as well as a version of my script for training and creating the EnsembleCNN.

'ExampleTrainData', is a subset of my traindata that I have included for you to better understand the dataset I created entirely myself, as well as being able to see some of the data augmentation methods I deployed to increase my dataset.

'plantstatus' and 'waterstatus' are folders that contain he assets used by my Kivy application. 