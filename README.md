This project is related to Intel Edge AI Fundamentals Udacity Course in Intel Edge AI Scholarship Challenge.

The purpose of this project is to try to make an app that could save pedestrian life.

Please find the project visualisation images:

 - [image1](https://github.com/ireneuszcierpisz/intel-edge-ai-challenge/blob/master/v1-img.png)    
 
 - [image2](https://github.com/ireneuszcierpisz/intel-edge-ai-challenge/blob/master/v2-img.png)

Goals:

I consider the possibility of utilization OpenVINO IR of pedestrian and vehicle detection network based on MobileNet v1.0 + SSD for detecting objects and:

	1)localize theirs position,
	2)tracking motion,
	3)predict future position using extrapolation,
	4) (phase 4) compute potential point of vehicle-person collision as an intersection point,
	5)compute a rate of changing objects position,
	6)emitting the warning

Idea:

	App should detect people and vehicle approaching to the same point.
	App should predict direction of movement of objects and potential collision point. 
	If objects are too close app should produce warning.
	In the future app like this could warn off people by their smartphone or rather special very small 
	devices build for this purpose by beeping, shaking or comunicating e.g. "STOP! DON'T CROSS THE ROAD!"
	Potentialy app like this could use video stream from the nearest cctv camera.


To build this app I have used Intel OpenVINO TOOLKIT and object detection model pedestrian-and-vehicle-detector-adas-0001 from Intel OpenVino Open ModelZoo. 

I've worked in Udacity classroom workspace with Intel OpenVINO environment.

The steps:

	a) downloading the model Intermediate Representation files 
		via intel/openvino/deployment_tools/tools/model_downloader
	b) building up command line app to use Intel OpenVINO Inference Engine


To run the project: 	
	
	python app.py -m "The_location/of_the_model_xml_file"

	-h usage: Run inference on an input live stream or video
       [-h] [-m M] [-i I] [-d D] [-c C]
       [-ct CT]

Required arguments:

  	-m M    The location of the model XML file

Optional arguments:

  	-i I    Live stream 'CAM' or the location of the video file or image
  
  	-d D    The device name ('GPU', 'MYRIAD') if not 'CPU'
  
  	-c C    The color of the bounding boxes to draw; RED, GREEN or BLUE
  
  	-ct CT  The confidence threshold to use with the bounding boxes

This is my first attempt and this working app shows rather the idea of something much bigger to build.

In that stage of app development process (phase 4) I found problems to solve:

	1)stabilize bounding boxes position for static objects,
	2)improve "tracker" for frames without previously localized bounding boxes,
	3)approximation extrapolated lines direction because of 2D presentation of 3D movement.

Files: 

	check_layers.py   -> Checks if there are unsupported layers in the model
	get_args.py       -> Gets the arguments from the command line
	capture_stream.py -> Checks if the input is image, webcam or video 
				and if allowed open video capture and create the video writer
	inference.py      -> Creates class Network with added def of function that returns not supported layers 
	app.py            -> with main function
    
I think that even in the future world of autonomus cars pedestrians will be still in danger on the road.
