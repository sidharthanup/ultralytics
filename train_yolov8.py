from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


model = YOLO("yolov8m.pt")
add_wandb_callback(model, enable_model_checkpointing=False)

results = model.train(project = "INat-yolov8", data = "../custom_dataset2.yaml",epochs = 100, batch = 20, save_period = 10, imgsz=600, device="0")

# Finish the W&B run
wandb.finish()