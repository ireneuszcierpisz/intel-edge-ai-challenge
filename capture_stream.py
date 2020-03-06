import cv2

""" Checks if the input is image, webcam or video and if allowed     open video capture and create the video writer"""

def capture_stream(args):

    # Create a flag for single images
    image_flag = False
    
    # Check if the input is a webcam or picture
    if args.i == 'CAM':
        args.i = 0
        
    # Check if the input is a picture   
    elif args.i.split('.')[-1] in ['jpg', 'gif', 'png', 'tiff', 'bmp']: 
        image_flag = True

    # Gets and open video capture       
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    
    # Grab the shape of the video frame 
    width = int(cap.get(3))
    height = int(cap.get(4))
    print('The original video frame shape: frame width={}, frame height={}'.format(width, height))

    # Create a video writer for the output video
    if not image_flag:
        video_writer = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (width,height))
    else:
        video_writer = None
        
    return cap, video_writer, image_flag, height, width
