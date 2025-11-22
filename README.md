# PyTorch SAR Image Colorization (Functional)

## 📁 Folder Structure
```
colorization/
│
├── data_utils.py
├── model_utils.py
├── train.py
├── predict.py
├── main.py
├── dataset/
│   └── class1/
│       └── sample1.jpg
└── test_bw.jpg
```

## Running the Project

### 1. Install dependencies
```
pip install torch torchvision pillow numpy scikit-image
```

### 2. Add your training images
Place your *colored* images here:
```
dataset/class1/
```

### 3. Train the model
In `main.py`, uncomment:
```
train_model("dataset/", save_path="colorize_model.pth", epochs=5)
```

Run:
```
python main.py
```

### 4. Predict from a new black & white image
In `main.py`, uncomment:
```
run_prediction("colorize_model.pth", "test_bw.jpg", "colored_output.png")
```

Run:
```
python main.py
```

