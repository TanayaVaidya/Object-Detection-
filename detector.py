import torch
from ultralytics import YOLO
import supervision as sv

class ObjectDetection:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO("yolov8n.pt")

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator()

    def predict(self, frame):

        results = self.model(frame, verbose=False)

        return results

    def plot_bboxes(self, results, frame):

        boxes = results[0].boxes

        detections = sv.Detections(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int)
        )

        labels = []

        for class_id, confidence in zip(detections.class_id, detections.confidence):
            label = f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
            labels.append(label)

        frame = self.box_annotator.annotate(scene=frame, detections=detections)

        frame = self.label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        return frame