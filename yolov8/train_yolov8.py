from ultralytics import YOLO

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="yolov8x.pt")
parser.add_argument("--config", type=str, default="configs/exp_conf.yaml")
parser.add_argument("--project", type=str, default="my_project", help="Path to save experiments")
parser.add_argument("--name", type=str, default="train")
parser.add_argument("--epoch", type=int, default=60)
parser.add_argument("--img_size", type=int, default=1472)
parser.add_argument("--single_cls", action="store_true", help="Enable training with single class")
parser.add_argument("--batch", type=int, default=-1, help="Default = -1 to auto batch finding")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--device", nargs="+", default=[0], help="GPU IDs. E.g., 0 1 2")

"""Example training command
python train_yolov8.py --config configs/exp_conf.yaml \
    --batch 12 --device 0,1,2,3 --name cancer_gastric_detection --epoch 100
"""
if __name__ == "__main__":
    # for more details: https://docs.ultralytics.com/modes/train
    args = parser.parse_args()
    print(args)

    model = YOLO(args.model)  # build from model weights

    # Train the model
    results = model.train(
        data=args.config, 
        project=args.project, 
        name=args.name, 
        epochs=args.epoch, 
        imgsz=args.img_size, 
        batch=args.batch, 
        workers=args.workers, 
        device=args.device,
    )