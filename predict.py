import torch
from model_utils import build_model, predict_image

def run_prediction(model_path, bw_image_path, output="result.png"):
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    predict_image(model, bw_image_path, output)
