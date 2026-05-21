import streamlit as st
import torch
import cv2
from ultralytics import YOLO
import supervision as sv
from time import time

st.title("Object Detection Model")

class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write("Using device:", self.device)

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


# ---------------- STREAMLIT UI ---------------- #

detector = ObjectDetection(0)

run_camera = st.button("Start Live Camera")

frame_placeholder = st.empty()

if run_camera:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        start = time()

        ret, frame = cap.read()
        if not ret:
            st.write("Camera not working")
            break

        results = detector.predict(frame)

        frame = detector.plot_bboxes(results, frame)

        fps = 1 / (time() - start)

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB")

    cap.release()