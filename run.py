"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argh

from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet


class Runner:
    def __init__(self, model_path, device_name='cuda'):
        if device_name == 'cuda' and not torch.cuda.is_available():
            print("WARN: cuda was selected as device but was not found")
            device_name = 'cpu'
        self.device = torch.device(device_name)

        print(f"device: {device_name}")
        self.model = MidasNet(model_path, non_negative=True)

        self.preprocessor = Compose(
            [
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        self.model.to(self.device)
        self.model.eval()

    def predict_depth(self, img_rgb):
        img_input = self.preprocessor({"image": img_rgb/255.0})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return prediction

    def weighted_filtering(self, rgb_image, depth_image):
        return cv2.ximgproc.weightedMedianFilter(rgb_image, depth_image.astype('float32'), 5, 15)


def run(input_path, output_path, model_path, median_filter=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """

    runner = Runner(model_path)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = utils.read_image(img_name)

        prediction = runner.predict_depth(img)

        if median_filter:
            prediction = runner.weighted_filtering(img, prediction)

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, prediction, bits=2)

    print("finished")


def main(input_path = "input",output_path = "output",model_path = "model.pt",
         median_filter=False):
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(input_path, output_path, model_path, median_filter)
