# -*- coding: utf-8 -*-
# https://medium.com/@alimustoofaa/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6
# https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?
from ultralytics import YOLO
import argparse


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        help='train dataset or test data'
    )
    parser.add_argument(
        '--mod',
        type=str,
        default='pred', 
        help='train dataset or test data'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8s.pt',
        help='base AI model to use'
    )
    parser.add_argument(
        '--save',
        type=bool,
        default=True,
        help='save model (True | False)'
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.mod == "train":    
        model.train(data=args.input, epochs=3, batch=12, save=args.save)  # train the model
        model.val()  # evaluate model performance on the validation s

    elif args.mod == "pred":
        results = model(data=args.input)

        for result in results: 
            boxes = result.boxes
            print(boxes)