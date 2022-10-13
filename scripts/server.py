import torch
from flask import request
from flask import Flask
import json
from omegaconf import OmegaConf

from custom_runner import upscale_image, text2img2, load_model_from_config
import os

from ldm.models.diffusion.plms import PLMSSampler


class Server:

    def __init__(self):
        self.sampler = None
        self.model = self.load_model()
        self.app = Flask(__name__)

        @self.app.route('/api', methods=['POST'])
        def api():
            return self.run_api()

    def load_model(self):
        os.chdir('..')
        config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
        model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sampler = PLMSSampler(model)
        return model.to(device)

    def run_api(self):
        request_data = request.get_json()
        _id = request_data['id']
        prompt = request_data['prompt']
        path = text2img2(self.model, self.sampler, prompt, _id)
        full_path = os.getcwd() + "/" + path
        print(full_path)
        #upscale_image(_id)
        val = {'path': full_path}
        response = self.app.response_class(
            response=json.dumps(val),
            status=200,
            mimetype='application/json'
        )
        return response


if __name__ == '__main__':
    server = Server()
    server.app.run()
