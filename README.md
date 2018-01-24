# Groggy-driver

### Inspiration
We were inspired by the great service that Lyft or Uber provides in order to have an easy an affordable commute. We wanted to add a feature in their services or even personal driving to increase the safety by avoiding road accidents due to fatigued driving i.e. if a driver is tired or sleepy would tend to shut his or her eyes and/or yawn often while driving.

### What it does
Groggy driver detection can form the basis of the system to possibly reduce the accidents related to driver’s drowsiness. The purpose of such a system is to perform detection of driver fatigue. By placing the camera inside the car, we can monitor the face of the driver and look for the eye and yawn movements which indicate that the driver is no longer in condition to drive. In such a case, a warning signal should be issued.

### How we built it
The general flow is as follows:

Setup a camera that monitors a stream of faces.
Apply facial landmark detection and extract eye and mouth regions. 2.a. Now that we have the eye regions, we can compute the eye aspect ratio(EAR) to determine if the eyes are closed.EAR algorithm was referred from the following research paper. http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf 2.b. Yawn detection was implemented using a face detection perpetual interface in opencv CV_HAAR_SCALE_IMAGE. It uses a statistical model (often called classifier), which is trained to find the object we are looking for. The training consists in a set of images, divided into “positive” samples and “negative” samples. The positive samples are instances of the object class of interest and the “negative”, images that don’t contain the object of interest. This interface has been referred from http://www.xavigimenez.net/blog/2010/02/face-detection-how-to-find-faces-with-opencv/
After the detection of eyes and mouth we use thresholding and pattern iteration technique to fin if the eye are closed or yawning happens consistently over a considerable number of frames. If the action is repeated consistently above the threshold then we signal an alarm to the driver to give a warning.

Technolgies used: Python, Opencv, numpy, flask, HTML5

### Challenges we ran into
The application deals with the facial features like eyes, ears and mouth. The Eye Aspect Ratio which decides if a blink is detected in consecutive frames in the video captured is a critical part of it. Also we detect the opening of the mouth and then detect yawning which is strictly increasing. But we had to set a minimum number of such yawns to determine if a driver is ‘groggy’ and raise the alarm. These two features run parallel and detect yawning and shutting of eyes which is one frame and divides the eye and mouth regions and to do their jobs to determine a sleepy/tired driver by raising an alarm.

### Accomplishments that we're proud of
To think, implement and test the idea which could be used by a driving service company/ car manufacturing company which could save life of millions with the concept of algorithms, computer vision, machine learning and human computer interaction.

### What we learned
Time management, team work and above all the passion for coding which can propel towards the solution in 24 hours.

### What's next for Groggy driver
We are brimming with ideas to extend this project and implement features like: Detecting drivers wearing spectacles. Also the application can be implemented using night vision cameras for dim or no light. Groggy driver is using OpenCV which can be used as driver authorization and prevent thievery by any other unauthorized person in that case a direct report can be sent if the car has been taken away. The face detection would give driving access only to the driver. Else it would be notified to the admin.
