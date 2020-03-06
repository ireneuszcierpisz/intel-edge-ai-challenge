import argparse

MODEL = "models/FP16/pedestrian-and-vehicle-detector-adas-0001.xml"
VIDEO = "images/Person_Vehicle.mp4"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input live stream or video")
    
    # Descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "Live stream 'CAM' or the location of the video file or image"
    d_desc = "The device name ('GPU', 'MYRIAD') if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    
    # Required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # Arguments
    required.add_argument("-m", help=m_desc, default= MODEL)
    optional.add_argument("-i", help=i_desc, default= VIDEO)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='RED')
    optional.add_argument("-ct", help=ct_desc, default=0.3)
    args = parser.parse_args()

    return args

