from ultralytics import YOLO

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--ckpt", type=str, help="Path to checkpoint YOLOv8 model")
parser.add_argument("--source", type=str, help="Path to test image")
parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence threshold for detection")
parser.add_argument("--iou", type=float, default=0.7, help="Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS)")
parser.add_argument("--imgsz", type=int, default=1472, help="Test image size")

if __name__=="__main__":
    # for more details: https://docs.ultralytics.com/modes/predict
    args = parser.parse_args()
    print(args)
    
    # load YOLOv8 model
    model = YOLO(args.ckpt)

    # run inference on the image
    results = model(
        source=args.source, 
        conf=args.conf, 
        iou=args.iou, 
        imgsz=args.imgsz
    )

    # get the predicted boxes in YOLO format: [x_c, y_c, w, h]
    pred_boxes = results[0].boxes.xywhn
    print(pred_boxes)