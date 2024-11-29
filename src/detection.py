import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

RESOLUTION = 128
OVERLAP_RATIO = 0.2

# input: image format must be ***unnormalized RGB*** (contiguous array of ints)
# output: image format is ***unnormalized RGB***
def detect(image):
    model = YOLO("weights/xview128px100epochs.pt")

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback = callback, slice_wh=(RESOLUTION, RESOLUTION), overlap_ratio_wh=None, overlap_wh=(OVERLAP_RATIO, OVERLAP_RATIO))
    detections = slicer(image)

    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(
      scene=image.copy(), detections=detections
    )
    ndetections = len(detections)
    labels = None
    if ndetections > 0: # if ndetections is 0 then detections.data["class_name"] is not set. Trying to access it would throw exception
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections.data["class_name"], detections.confidence)
        ]
    label_annotator = sv.LabelAnnotator()
    annotated_image = label_annotator.annotate(
      scene=annotated_image, detections=detections, labels=labels
    )

    return {"annotated_image": annotated_image, "ndetections": ndetections}
