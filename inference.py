import os
import sys
import logging
from openvino.inference_engine import IENetwork, IECore
logging.getLogger().setLevel(logging.INFO)

""" Creates class Network with def of function returns not supported layers"""


class Network:
    '''
    Load and store information for working with the Inference Engine and any loaded models.
    '''
    
    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()
        logging.info('Plugin initialized.')
        
        
        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
            logging.info('CPU extension added')
            
            
        # Read the IR as a IENetwork
        logging.info('Reading network from IR as IENetwork.')        
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)
        logging.info('IENetwork loaded into the plugin as exec_network.')
        
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        logging.info('Got input_blob from network.imputs generator.')        
        self.output_blob = next(iter(self.network.outputs))
        
        logging.info('Model files:\n{}\n{}\nDevice:{}'.format(model_xml, model_bin, device))
        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        #logging.info('Starting asynchronous inference')        
        self.exec_network.start_async(request_id=0, 
            inputs={self.input_blob: image})
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        #logging.info('Waiting for async request to be complete')        
        status = self.exec_network.requests[0].wait(-1)
        #logging.info('Status of the inference request: {}'.format(status))        
        return status


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs[self.output_blob]

    
    #Gets not supported layers
    def not_supported_layers(self, device):
            #gets supported layers dictionary
        supported_layers = self.plugin.query_network(self.network, device)
        print('Supported layers:')
        count = 0
        for k, v in supported_layers.items():
            count += 1
            print(' #{:3}   {:40}    {}'.format(count, k, v))
        
        #gets list of not supported layers 
        not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        print('Found: ', len(not_supported_layers),' Unsupported layers:\n', not_supported_layers)

        return not_supported_layers
