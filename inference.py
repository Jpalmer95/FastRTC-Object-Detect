import cv2
import numpy as np
import onnxruntime
import time

# COCO class names
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Generate colors for bounding boxes
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color and thickness. """
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image, text, box, color, font_size=0.5, text_thickness=1):
    """ Draws text on an image with a given color and font size. """
    x1, y1, x2, y2 = box.astype(int)
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, text_thickness, cv2.LINE_AA)


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.4):
    """ Draws detections on an image. """
    det_img = image.copy()
    img_height, img_width = det_img.shape[:2]
    font_size = min(img_width, img_height) * 0.0006
    text_thickness = int(min(img_width, img_height) * 0.001)

    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]
        draw_box(det_img, box, color)
        label = f"{class_names[class_id]}: {score:.2f}"
        draw_text(det_img, label, box, color, font_size, text_thickness)
    return det_img


class YOLOv10:
    def __init__(self, path):
        self.session, self.input_name, self.output_name, self.input_width, self.input_height, self.input_shape = self.initialize_model(path)
        self.draw_detections_helper = draw_detections # Assign helper function

    def initialize_model(self, path):
        session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())
        input_details = self.get_input_details(session)
        output_details = self.get_output_details(session)

        input_name = input_details[0]['name']
        input_shape = input_details[0]['shape']
        input_width = input_shape[2]
        input_height = input_shape[3] # Corrected typo from input_h eight

        output_name = output_details[0]['name']
        return session, input_name, output_name, input_width, input_height, input_shape

    def detect_objects(self, image, conf_threshold=0.3):
        input_tensor = self.prepare_input(image)
        # The inference method now returns the image with detections drawn on it.
        output_image_with_detections = self.inference(image, input_tensor, conf_threshold)
        return output_image_with_detections

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, image, input_tensor, conf_threshold=0.3):
        start = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})

        boxes, scores, class_ids = self.process_output(outputs[0], conf_threshold)

        # Call the internal drawing method that uses the helper
        output_image = self.draw_detections_internal(image.copy(), boxes, scores, class_ids) # Pass a copy of the image

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return output_image


    def process_output(self, output, conf_threshold=0.3):
        predictions = np.squeeze(output).T

        # Filter out detections with low confidence
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        if scores.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        # x_center, y_center, width, height -> x1, y1, x2, y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes = np.column_stack((x1, y1, x2, y2))
        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        # input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        # boxes = np.divide(boxes, input_shape, dtype=np.float32) # This was for yolov8, yolov10 output is already 0-1

        # For YOLOv10, the output format is [x_center, y_center, width, height] (normalized to 0-1)
        # So, we just need to multiply by original image dimensions
        boxes[:, 0] *= self.img_width
        boxes[:, 1] *= self.img_height
        boxes[:, 2] *= self.img_width
        boxes[:, 3] *= self.img_height
        return boxes

    def get_input_details(self, session):
        model_inputs = session.get_inputs()
        input_details = [{'name': input.name, 'shape': input.shape, 'type': input.type} for input in model_inputs]
        return input_details

    def get_output_details(self, session):
        model_outputs = session.get_outputs()
        output_details = [{'name': output.name, 'shape': output.shape, 'type': output.type} for output in model_outputs]
        return output_details

    def draw_detections_internal(self, image, boxes, scores, class_ids, mask_alpha=0.4):
        """ Internal method to call the global draw_detections helper. """
        return self.draw_detections_helper(image, boxes, scores, class_ids, mask_alpha)
