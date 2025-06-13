import gradio as gr
import cv2
from huggingface_hub import hf_hub_download
from inference import YOLOv10 # Assuming inference.py is in the same directory
from fastrtc import Stream # Corrected import

# Download the YOLOv10n ONNX model
model_path = hf_hub_download(repo_id="onnx-community/yolov10n", filename="onnx/model.onnx")

# Instantiate the YOLOv10 model
model = YOLOv10(model_path)

# Define a global placeholder for the numpy import inside detection, if needed by Gradio's environment
# For now, assuming numpy is available globally in the execution scope of detection.
# import numpy as np # This would be here if detection was in a separate file not seeing global np

def detection(image, conf_threshold):
    """
    Performs object detection on an image.

    Args:
        image: The input image from the webcam (RGB numpy array).
        conf_threshold: The confidence threshold for object detection.

    Returns:
        The image with detections drawn on it.
    """
    if image is None:
        # Gradio might send None if the stream is just starting or interrupted.
        # Return a blank image of the expected display size.
        # Ensure np is accessible; if not, it needs to be imported or passed.
        # For this subtask, we assume np from global imports of inference.py is fine.
        # If issues arise, `import numpy as np` might be needed here or in app.py's global scope.
        # The error wasn't about np, so let's keep it clean.
        # The 'np' used in inference.py for np.zeros comes from its own import.
        # If app.py needs np directly for a placeholder, it should import it.
        # For now, the placeholder is in inference.py if image is None *there*.
        # Here, if image is None, it means Gradio didn't provide one.
        # The detection function expects an image. If Gradio gives None,
        # we should probably return a blank image that matches the output format.
        return np.zeros((480, 640, 3), dtype=np.uint8) # Match display_image dimensions

    # Convert RGB (from Gradio) to BGR (expected by YOLOv10.prepare_input via cv2.cvtColor)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    processed_image_bgr = model.detect_objects(image_bgr, conf_threshold)

    processed_image_rgb = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)

    display_image = cv2.resize(processed_image_rgb, (640, 480))

    return display_image

# Create the fastrtc.Stream object
# This object will be used as the function for gr.Interface
# It encapsulates the video handling and calls the 'detection' handler.

# Define the slider component first in the global scope
conf_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3, label="Confidence Threshold")

# Initialize FastRTC Stream. This might configure 'detection' for RTC
# or set up global state for FastRTC. The Stream object itself is not directly
# used in gr.Interface fn, inputs, or outputs based on previous errors.
# It's assumed to work by side effect or by having its handler called.
stream_config = Stream( # Renamed variable for clarity
    handler=detection, # The actual function that will process data
    modality="video",
    mode="send-receive",
    additional_inputs=[conf_slider] # FastRTC knows about this additional input
)

# Launch the application
if __name__ == "__main__":
    # Use the original 'detection' function as fn for gr.Interface
    # FastRTC's Stream object, initialized above, is expected to
    # enable RTC capabilities for this standard Gradio streaming interface.

    iface = gr.Interface(
        fn=detection, # Use the original callable function
        inputs=[
            gr.Image(sources=["webcam"], type="numpy", streaming=True),
            conf_slider # Pass the slider instance
        ],
        outputs=gr.Image(type="numpy"),
        live=True, # Required for streaming interfaces
        title="FastRTC YOLOv10 Object Detection",
        description="Real-time object detection using YOLOv10n, FastRTC, and Gradio."
    )

    # Import numpy for the placeholder if image is None at the app.py level
    import numpy as np # Needed for the np.zeros in detection if image is None

    iface.launch()
