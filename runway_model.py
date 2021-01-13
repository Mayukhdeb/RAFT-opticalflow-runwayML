import cv2
import torch
import logging    
import numpy as np
from PIL import Image

import runway
from runway.data_types import image, boolean, number

from core.raft import RAFT
from core.frame_buffer import frame_buffer
from core.constants import args as model_args
from core.utils.utils import InputPadder, preprocess_image, generate_flow_output

if model_args.log_filename is not None:
    logging.basicConfig(filename= model_args.log_filename, level = logging.INFO)


buffer = frame_buffer()

input_dict = { 
    "image": image(), 
    "side_by_side": boolean(default = False, description = "Place the original video to the side of the output for reference."),
    "iterations": number(default = 20, max = 50, min = 1, step = 1, description = " Number of iterations to be made within the model's update block. [Higher -> better quality].")
}

@runway.setup
def setup():
    model = torch.nn.DataParallel(RAFT(model_args))
    model.load_state_dict(torch.load(model_args.model_path))
    model = model.module
    model.to(model_args.device)
    model.eval()
    return model

@runway.command(
    name = "generate", 
    inputs= input_dict, 
    outputs={ "image": image() }
)

def generate(model, args):

    with torch.no_grad():
        
        input_image = preprocess_image(args["image"], device = model_args.device)
        buffer.set_current(input_image)

        image1, image2 = buffer.frame1, buffer.frame2
        if image1 is not None and image2 is not None:

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_up = model(image1, image2, iters=args["iterations"])
            out = generate_flow_output(flow_up).astype(np.uint8) ## range [0., 255.] numpy array

            if args["side_by_side"]== True:
                original_resized = cv2.resize(np.array(args["image"]), (out.shape[1], out.shape[0]))
                out = cv2.hconcat([original_resized, out])

            out = Image.fromarray(out)
        else:
            out = None
        return {"image": out}

if __name__ == '__main__':
    runway.run()