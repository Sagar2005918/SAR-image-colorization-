from train import train_model
from predict import run_prediction

# TRAIN:
#train_model("dataset/", save_path="colorize_model.pth", epochs=5)

# PREDICT:
run_prediction("colorize_model.pth", "gray.jpeg", "colored_output.png")
 