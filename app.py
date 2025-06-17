import gradio as gr
import cv2
from huggingface_hub import hf_hub_download
# Updated import: also get class_names and draw_detections from inference
from inference import YOLOv10, class_names as coco_class_names_from_inference, draw_detections
from fastrtc import Stream # Corrected import
import time # For timestamp in email

# Download the YOLOv10n ONNX model
model_path = hf_hub_download(repo_id="onnx-community/yolov10n", filename="onnx/model.onnx")

# Instantiate the YOLOv10 model
model = YOLOv10(model_path)

# Global dictionary to store object counts per frame
object_counts = {}

# Define the slider component first in the global scope
conf_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3, label="Confidence Threshold")

# --- Firebase Settings Integration ---
import json # json was already imported below, but placing it here is fine too
import requests

GET_PREFS_URL = "YOUR_GET_USER_PREFERENCES_FUNCTION_URL" # TODO: Update with deployed Firebase Function URL
SET_PREFS_URL = "YOUR_SET_USER_PREFERENCES_FUNCTION_URL" # TODO: Update with deployed Firebase Function URL
SEND_EMAIL_URL = "YOUR_SEND_EMAIL_NOTIFICATION_FUNCTION_URL" # TODO: Update with deployed Firebase Function URL
SIMULATED_USER_ID_TOKEN = "dummy-firebase-id-token"

COCO_CLASSES_FOR_UI = ["person", "car", "dog", "cat", "bottle"]

current_user_settings = {
    "watchedObjects": {cls: False for cls in COCO_CLASSES_FOR_UI},
    "objectActions": {
        cls: {"count": False, "notifyOnDetect": False, "recordOnDetect": False}
        for cls in COCO_CLASSES_FOR_UI
    },
    "notificationEmail": "user@example.com"
}

def detection(image, conf_threshold):
    """
    Performs object detection on an image, filters based on settings, and draws results.
    """
    if image is None: # Gradio might pass None if webcam isn't ready
        # Ensure np is available if not already. It's imported in __main__ for now.
        # For robustness, one might ensure np is imported globally in app.py
        return np.zeros((480, 640, 3), dtype=np.uint8)

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raw_boxes, raw_scores, raw_class_ids = model.detect_objects(bgr_image, conf_threshold)

    global object_counts
    object_counts.clear()

    watched_boxes = []
    watched_scores = []
    watched_class_ids = []

    if raw_boxes is not None and len(raw_boxes) > 0:
        for box, score, class_id_val in zip(raw_boxes, raw_scores, raw_class_ids): # Renamed class_id to avoid conflict
            class_name = coco_class_names_from_inference[int(class_id_val)]

            if class_name in COCO_CLASSES_FOR_UI and current_user_settings["watchedObjects"].get(class_name, False):
                watched_boxes.append(box)
                watched_scores.append(score)
                watched_class_ids.append(class_id_val)

                actions = current_user_settings["objectActions"].get(class_name, {})
                if actions.get("count", False):
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1

                if actions.get("notifyOnDetect", False):
                    trigger_email_notification(class_name) # Call new function

                if actions.get("recordOnDetect", False):
                    print(f"RECORDING TRIGGER: Detected {class_name}")

    output_image_bgr = bgr_image.copy()
    if watched_boxes:
        output_image_bgr = draw_detections(output_image_bgr, watched_boxes, watched_scores, watched_class_ids)

    y_offset = 30
    for class_name_count, count_val in object_counts.items():
        text = f"{class_name_count}: {count_val}"
        cv2.putText(output_image_bgr, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    output_image_rgb = cv2.cvtColor(output_image_bgr, cv2.COLOR_BGR2RGB)
    display_image = cv2.resize(output_image_rgb, (640, 480))
    return display_image

# Initialize FastRTC Stream
stream_config = Stream(
    handler=detection,
    modality="video",
    mode="send-receive",
    additional_inputs=[conf_slider]
)

def load_user_settings_from_firebase():
    global current_user_settings
    headers = {"Authorization": f"Bearer {SIMULATED_USER_ID_TOKEN}", "Content-Type": "application/json"}
    try:
        print(f"Attempting to load settings from: {GET_PREFS_URL}")
        response = requests.post(GET_PREFS_URL, headers=headers, json={"data": {}})
        response.raise_for_status()
        data = response.json()
        if "result" in data:
            loaded_settings = data["result"]
        elif "error" in data:
            print(f"Error from Firebase function: {data['error']}")
            return "Error loading settings from Firebase."
        else:
            loaded_settings = data

        current_user_settings["notificationEmail"] = loaded_settings.get("notificationEmail", current_user_settings["notificationEmail"])
        loaded_watched = loaded_settings.get("watchedObjects", {})
        for cls in COCO_CLASSES_FOR_UI:
            current_user_settings["watchedObjects"][cls] = loaded_watched.get(cls, False)
        loaded_actions = loaded_settings.get("objectActions", {})
        for cls in COCO_CLASSES_FOR_UI:
            cls_actions = loaded_actions.get(cls, {})
            current_user_settings["objectActions"][cls]["count"] = cls_actions.get("count", False)
            current_user_settings["objectActions"][cls]["notifyOnDetect"] = cls_actions.get("notifyOnDetect", False)
            current_user_settings["objectActions"][cls]["recordOnDetect"] = cls_actions.get("recordOnDetect", False)
        print("Successfully loaded settings:", json.dumps(current_user_settings, indent=2))
        return json.dumps(current_user_settings, indent=2)
    except requests.exceptions.RequestException as e:
        print(f"Error loading settings: {e}")
        if "YOUR_GET_USER_PREFERENCES_FUNCTION_URL" in GET_PREFS_URL:
            print("Note: Firebase URL is a placeholder. Actual call would fail.")
            return "Error: Firebase URL not configured. Using default/current settings."
        return f"Error loading settings: {str(e)}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Unexpected error: {str(e)}"

def save_user_settings_to_firebase(*args):
    global current_user_settings
    new_settings = {"notificationEmail": args[0], "watchedObjects": {}, "objectActions": {}}
    arg_idx = 1
    for cls in COCO_CLASSES_FOR_UI:
        new_settings["watchedObjects"][cls] = args[arg_idx]
        new_settings["objectActions"][cls] = {
            "count": args[arg_idx+1],
            "notifyOnDetect": args[arg_idx+2],
            "recordOnDetect": args[arg_idx+3]
        }
        arg_idx += 4
    headers = {"Authorization": f"Bearer {SIMULATED_USER_ID_TOKEN}", "Content-Type": "application/json"}
    try:
        print(f"Attempting to save settings to: {SET_PREFS_URL}")
        print("Settings to save:", json.dumps(new_settings, indent=2))
        response = requests.post(SET_PREFS_URL, headers=headers, json={"data": new_settings})
        response.raise_for_status()
        current_user_settings = new_settings
        return f"Settings saved successfully: {response.json().get('message', 'OK')}"
    except requests.exceptions.RequestException as e:
        print(f"Error saving settings: {e}")
        if "YOUR_SET_USER_PREFERENCES_FUNCTION_URL" in SET_PREFS_URL:
            print("Note: Firebase URL is a placeholder. Actual call would fail.")
            return "Error: Firebase URL not configured. Settings not saved to cloud."
        return f"Error saving settings: {str(e)}"
    except Exception as e:
        print(f"An unexpected error occurred during save: {e}")
        return f"Unexpected error during save: {str(e)}"

def trigger_email_notification(class_name_detected):
    global current_user_settings, SIMULATED_USER_ID_TOKEN, SEND_EMAIL_URL
    target_email = current_user_settings.get("notificationEmail")
    if not target_email:
        print(f"No notification email configured globally. Cannot send notification for {class_name_detected}.")
        return
    subject = f"Object Detection Alert: {class_name_detected}"
    body = f"A '{class_name_detected}' has been detected by your application at {time.strftime('%Y-%m-%d %H:%M:%S')}."
    payload = {"recipient_email": target_email, "subject": subject, "body": body}
    headers = {"Authorization": f"Bearer {SIMULATED_USER_ID_TOKEN}", "Content-Type": "application/json"}
    try:
        print(f"Attempting to call email notification function at {SEND_EMAIL_URL} for {class_name_detected} to {target_email}")
        if "YOUR_SEND_EMAIL_NOTIFICATION_FUNCTION_URL" in SEND_EMAIL_URL:
            print("Note: Email Firebase URL is a placeholder. Actual call would fail. Simulating success.")
            print(f"Payload that would be sent: {json.dumps({'data': payload})}")
            mock_response_data = {"result": {"message": f"Email simulation successful for: {target_email}"}}
            print(f"Mocked successful call for {class_name_detected}. Response: {mock_response_data}")
            return
        response = requests.post(SEND_EMAIL_URL, json={"data": payload}, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"Email notification function called successfully for {class_name_detected}. Response: {response.json()}")
    except requests.exceptions.Timeout:
        print(f"Timeout when calling email notification function for {class_name_detected}.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error calling email notification function for {class_name_detected}: {http_err} - Response: {http_err.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error calling email notification function for {class_name_detected}: {e}")
    except Exception as e:
        print(f"Unexpected error in trigger_email_notification for {class_name_detected}: {e}")

def create_settings_ui():
    ui_components = []
    with gr.Blocks() as settings_interface:
        gr.Markdown("## User Settings")
        email_comp = gr.Textbox(label="Notification Email", value=current_user_settings["notificationEmail"])
        ui_components.append(email_comp)
        loaded_settings_json_output = gr.JSON(label="Loaded Settings Preview")
        with gr.Accordion("Object Detection Preferences", open=False):
            for cls in COCO_CLASSES_FOR_UI:
                with gr.Row(): gr.Label(cls.title())
                with gr.Row():
                    watch_cb = gr.Checkbox(label=f"Watch for {cls}", value=current_user_settings["watchedObjects"].get(cls, False))
                    count_cb = gr.Checkbox(label=f"Count {cls}", value=current_user_settings["objectActions"].get(cls, {}).get("count", False))
                    notify_cb = gr.Checkbox(label=f"Notify on {cls} detect", value=current_user_settings["objectActions"].get(cls, {}).get("notifyOnDetect", False))
                    record_cb = gr.Checkbox(label=f"Record on {cls} detect", value=current_user_settings["objectActions"].get(cls, {}).get("recordOnDetect", False))
                    ui_components.extend([watch_cb, count_cb, notify_cb, record_cb])
        with gr.Row():
            load_button = gr.Button("Load Settings from Firebase")
            save_button = gr.Button("Save Settings to Firebase")
        status_message = gr.Textbox(label="Status", interactive=False)
        load_button.click(fn=load_user_settings_from_firebase, inputs=[], outputs=[loaded_settings_json_output])
        save_button.click(fn=save_user_settings_to_firebase, inputs=ui_components, outputs=[status_message])
    return settings_interface

if __name__ == "__main__":
    import numpy as np # Needed for np.zeros in detection if image is None
    video_interface = gr.Interface(
        fn=detection,
        inputs=[gr.Image(sources=["webcam"], type="numpy", streaming=True), conf_slider],
        outputs=gr.Image(type="numpy"),
        live=True,
        title="Live Object Detection Stream",
    )
    settings_ui = create_settings_ui()
    app = gr.TabbedInterface([video_interface, settings_ui], ["Object Detection Stream", "Settings"])
    app.launch()
