import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision import ColorPalette, Detections, BoxAnnotator
 
class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model=self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(color=ColorPalette(10).by_idx(0), thickness=3, text_thickness=3, text_scale=1.5)

    def load_model(self):
            model = YOLO("yolov8m.pt")
            model.fuse()
            return model
        
    def predict(self, frame):
            results = self.model(frame)
            return results
        
    def plot_bboxes(self, results, frame):

            xyxys=[]
            confidences=[]
            class_ids=[]

            for result in results[0]:
                class_id = results [0].boxes.cls.cpu().numpy().astype(int)

                if class_id==0:
                    xyxys.append(result.boxes.xyxy.cpu().numpy())
                    confidences.append(result.boxes.conf.cpu().numpy())
                    class_ids.append(result.boxes.cls.cpu().numpy().astype(int))  

            detections = Detections(
                xyxys=results[0].boxes.xyxy.cpu().numpy(),
                confidences=results[0].boxes.conf.cpu().numpy(),
                class_ids=results[0].boxes.cls.cpu().numpy().astype(int)
            )

            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, confidence, class_id in detections]   

            frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
            return frame
        
    def __call__(self):
            cap = cv2.VideoCapture(self.capture_index)
            assert cap.isOpened()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while True:
                
                start_time = time()
                ret, frame = cap.read()
                assert ret

                results = self.predict(frame)
                frame = self.plot_bboxes(results, frame)

                end_time = time()

                fps = 1 / (end_time - start_time)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("YOLOv8 Object Detection", frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
         
