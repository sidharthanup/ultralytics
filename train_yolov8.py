from ultralytics import YOLO
#from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks


model = YOLO("yolov8m.yaml")
#add_wandb_callbacks(model, project = "INat-yolov8")


results = model.train(data = "../custom_dataset.yaml",epochs = 2, save_period = 10)