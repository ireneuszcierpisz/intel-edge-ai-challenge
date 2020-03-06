# Saving pedestrian life App. 
# Goals: I consider the possibility of utilize pedestrian and vehicle detector model for detecting objects and: 1)localize theirs position, 2)tracking motion, 3)predict future position using extrapolation, 4)compute potential point of vehicle-person collision as a intersection point, 5)compute a rate of changing position, 6)emitting warning
# Problems: 1)stabilize b.box position for static object, 2) improve "tracker" for avoid frames without some b.boxes, 3)approximation extrapolated lines direction because of 2D presentation of 3D movement.
# App can detect people going toward the edge of a road and cars approaching to the same point. App can predict direction of movement of objects and potential collision point. If object are too close produce warning. In the future app like this could be warning people by their smartphone or special small(maybe size of a button) things build for this purpose by beeping, shaking or comunicating: "STOP! DON'T CROSS THE ROAD!" Potentialy (in my imagination) app like this could use video stream from nearest to pedestrian cctv camera.
# 
# This is my first attempt and my project shows rather the idea of something much bigger to build. This app doesn't solve a real life problem like accidents with pedestrians. I think that even in the future world of autonomus cars pedestrians will be still in danger on the road.
# To build my app I have used Intel OpenVINO tools and object detection model from Intel OpenVino ModelZoo. Files: check_layers->Checks if there are unsupported layers in the model,  get_args->Gets the arguments from the command line, capture_stream->Checks if the input is image, webcam or video and if allowed open video capture and create the video writer, inference->Creates class Network with def of function returns not supported layers added, myapp->with main function.

from inference import Network
import cv2
import numpy as np
import sys
from get_args import get_args
from capture_stream import capture_stream
from check_layers import check_layers
import logging

logging.getLogger().setLevel(logging.INFO)

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}

def infer_on_video(args):
    net = Network()
    logging.info('IE initialized.')    
    model = args.m
    device = args.d
    color = colors[args.c]
    
    # Initialize plugin, add CPU extension and read network from IR
    net.load_model(model, device, CPU_EXTENSION)
    logging.info('Network model loaded into the IE.')
    
    # checks supported/not supported layers in the model and logging.info or exit
    check_layers(net, device)
    
    # Get and open video capture
    cap, video_writer, image_flag, height, width = capture_stream(args)
    logging.info('Video captured.')
    
    # create variable to count frames
    count = 0
    # creates tracker dict to collect object location points
    tracker = {} 
    
    # Process frames until the video ends, or process is exited
    logging.info('Processing frames...')
    
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        # Waits for a key event for 60 milliseconds
        key_pressed = cv2.waitKey(60)  
        
        # Pre-process the input frame to required shape 
        model_shape = net.get_input_shape()
        model_w = model_shape[3]
        model_h = model_shape[2]
        # copy the frame as numpy.ndarray which is an object stands for a N-dimensional array 
        # and then the returning copy assignes to the frame4infer variable.
        frame4infer = np.copy(frame)
        frame4infer = cv2.resize(frame4infer, (model_w, model_h))
        frame4infer = frame4infer.transpose((2,0,1))
        frame4infer = frame4infer.reshape(1, 3, model_h, model_w)
        
        # Perform inference on the frame4infer
        net.async_inference(frame4infer)
        
        # Get the output of inference
        frame_copy = frame.copy()
        if net.wait() == 0:
            output = net.extract_output()
            
            count += 1
            f = "frame"+str(count) 
            if count == 1:
                last_f = f
            else:
                last_f = "frame"+str(count-1)
            tracker[f] = {}
            tracker[f]["vehicle"] = {}
            tracker[f]["person"] = {}
            i = 0
            vehicles, persons = [], [] #

            # Update the frame to include detected bounding boxes
            for box in output[0][0]: # Output shape is 1x1x100x7                
                confidence = box[2]
                if confidence >= args.ct:
                    i += 1
                    xmin = int(box[3] * width)
                    ymin = int(box[4] * height)
                    xmax = int(box[5] * width)
                    ymax = int(box[6] * height)
                    #gets the center point of bounding box
                    xc, yc = (xmax-xmin)//2 + xmin, (ymax-ymin)//2 + ymin
                    
                    # sets object's id and bounding_box color
                    if box[1] == 1: obj_id, color = "vehicle", colors["RED"]
                    else: obj_id, color = "person", colors["BLUE"]   
                    
                    # updates the tracker                    
                    b = "box"+str(i)
                    if b in tracker[last_f][obj_id]:
                        last_points = tracker[last_f][obj_id][b]
                        all_points = last_points + [(xc, yc)]
                        tracker[f][obj_id][b] = all_points                  
                    else:                        
                        tracker[f][obj_id].update({b: [(xc, yc)]})

                    # drawing bounding box and object location center point
                    cv2.rectangle(frame_copy, (xmin,ymin),(xmax,ymax), color, 2 )
                    for point in tracker[f][obj_id][b]:
                        cv2.circle(frame_copy, point, 5, color, -1)
                    cv2.putText(frame_copy, obj_id, (xc - 20, yc + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1)
                    # creates the list of bounding boxes center points
                    points = tracker[f][obj_id][b]
                    
                    
                    ''' COMPUTE EXTRAPOLATION LINE'''   
                    if len(points) > 1:
                        (x1, y1), (x2, y2) = points[0], points[-1]
                        x_list = []
                        y_list = []
                    # There is possible division by zero error 
                    # if object doesn't move in direction or if len(points)==1 for first frame
                        if x1 == x2:  
                            x2 = x2 + 1
                    # since linear equation y=a*x+b gets a,b
                        a = (y2 - y1) / (x2 - x1)
                        b = y1 - (a * x1)    
                        x_extr, y_extr = x1, y1   # create extrapolation variables
                    # sets limits to computing because of the frame  size
                        while (y_extr>=1 and y_extr<=height and x_extr>=1 and x_extr<=width):
                            x_list.append(x_extr)
                            y_list.append(y_extr)
                            if x2 > x1:
                                x_extr += 1
                                y_extr = int(x_extr * a + b) 
                            elif x2 < x1:
                                x_extr -= 1
                                y_extr = int(x_extr * a + b)
                        cv2.line(frame_copy, (x_list[0], y_list[0]), (x_list[-1], y_list[-1]), color, 2)
                    
                        ''' COMPUTE INTERSECTION POINTS'''

                        # appends constants a and b needed to compute (x,y) intersection points
                        if obj_id == "vehicle":
                            vehicles.append((a, b))
                        else: 
                            persons.append((a, b))
            # for given frame computes intersection point 
            # for each two extrapolation lines person-vehicle
            for p in persons:
                for v in vehicles:
                    p0, v0 = p[0], v[0]
                    if v0 == 0:  
                        v0 = 0.000001
                    if p0 == v0:
                        p0 -= 0.000001
                    y = int((p[1] - (p0 * v[1]) / v0) / (1 - p0/v0))
                    x = int((y - v[1]) / v0)
                    if (y>=1 and y<=height and x>=1 and x<=width):
                        cv2.circle(frame_copy, (x,y), 30, colors["GREEN"], -1)
                        
                    
            # Write out the frame_copy depending of image or video
            if image_flag:
                cv2.imwrite("output_image.jpg", frame_copy)
                logging.info("Got output image!")
            else:
                video_writer.write(frame_copy)
                
        # Break if escape key pressed
        if key_pressed == 27:
            break
    
    # Release the video_writer, capture, and destroy any OpenCV windows
    # Close the video writer if the input is not image
    if not image_flag:
        video_writer.release()
        logging.info("Got output_video!")
    # Close video file and allow OpenCV to release the captured file
    cap.release()
    # Destroy all of the opened HighGUI windows
    cv2.destroyAllWindows()
    

def main():
    args = get_args()

    infer_on_video(args)


if __name__ == "__main__":
    main()
