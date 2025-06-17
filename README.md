# FastRTC YOLOv10 Object Detection App

This application uses FastRTC and YOLOv10 to perform real-time object detection on a webcam feed or mobile camera, with user-configurable settings and (simulated) email notifications, designed for deployment on Google Cloud (Cloud Run and Firebase).

## Features

-   **Real-time Object Detection:** Utilizes YOLOv10n for efficient object detection on video streams.
-   **Web Interface:** Powered by Gradio and FastRTC for interactive webcam/mobile camera input.
-   **Adjustable Confidence Threshold:** Allows users to set the detection confidence level.
-   **User-Configurable Settings (via "Settings" tab):**
    *   Select specific objects to watch (e.g., person, car, dog, cat, bottle).
    *   Enable actions for watched objects:
        *   **Counting:** Display real-time counts of selected objects on the video feed.
        *   **Email Notification (Simulated):** Trigger a (simulated) email when specific objects are detected.
        *   **Recording (Placeholder):** Placeholder for future video recording functionality upon detection.
    *   Configure a notification email address.
    *   Settings are intended to be stored and retrieved from Firestore via Firebase Cloud Functions.
-   **(Simulated) Email Notifications:** Demonstrates triggering email alerts for detected objects.

## Architecture Overview

-   **Frontend & Video Processing (Gradio App):**
    *   A Python application using Gradio and FastRTC for the web interface and real-time video stream handling.
    *   Object detection is performed by a YOLOv10 ONNX model.
    *   This component is designed to be containerized using Docker and deployed on Google Cloud Run.
-   **Backend Logic (User Settings & Notifications):**
    *   Firebase Cloud Functions (Python runtime) are used for:
        *   Managing user-specific preferences (which objects to watch, actions to take), stored in Firestore.
        *   Handling (simulated) email notifications.
-   **Hosting & Routing:**
    *   Firebase Hosting serves as the main entry point for the application.
    *   It serves static content (like a landing page) and routes dynamic requests (e.g., to `/app/`) to the Gradio application running on Cloud Run.
-   **Authentication:**
    *   The architecture is designed for user-specific settings, implying user authentication.
    *   Firebase Authentication is the intended method. Currently, `app.py` uses a `SIMULATED_USER_ID_TOKEN` for demonstration. A full implementation would require client-side Firebase Auth to obtain a real ID token.
-   **Data Storage:**
    *   User preferences are stored in Firestore, accessed via Firebase Cloud Functions.

## Local Development & Running

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies for the Gradio App:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    (Note: For Firebase Functions local development, you'd `pip install -r firebase_functions/requirements.txt` within that directory or its virtual environment).

4.  **Running the Gradio Application:**
    ```bash
    python app.py
    ```
    The application will be available at a local URL (e.g., `http://127.0.0.1:7860`).

5.  **Note on Firebase Interactions:**
    *   When running locally, calls from `app.py` to Firebase Cloud Functions (for loading/saving settings, sending emails) will use placeholder URLs. These calls will print simulation messages to the console and will not interact with actual Firebase services unless you have the Firebase Emulator Suite running and have updated the URLs in `app.py` accordingly.

## Deployment to Firebase and Google Cloud

This section outlines the steps to deploy the application.

**Prerequisites:**

*   A Google Cloud Project with Billing enabled.
*   A Firebase Project linked to your Google Cloud Project.
*   Firebase CLI installed: `npm install -g firebase-tools` and configured: `firebase login`.
*   Google Cloud SDK (gcloud) installed and configured: `gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`.
*   Docker installed on your local machine.

**Step 1: Configure Project IDs and URLs**

*   **In `firebase.json`:**
    *   Replace `YOUR_GRADIO_CLOUD_RUN_SERVICE_ID` with the name you'll give your Cloud Run service (e.g., `yolov10-gradio-app`).
    *   Replace `YOUR_CLOUD_RUN_REGION` with the Google Cloud region where you'll deploy Cloud Run (e.g., `us-central1`).
*   **In `app.py`:**
    *   The placeholder URLs for `GET_PREFS_URL`, `SET_PREFS_URL`, and `SEND_EMAIL_URL` will be updated after deploying Firebase Functions (Step 2).

**Step 2: Deploy Firebase Functions**

1.  Firebase Functions are defined in the `firebase_functions` directory.
2.  Deploy them using the Firebase CLI from the root of your repository:
    ```bash
    firebase deploy --only functions
    ```
3.  After deployment, the Firebase CLI or Google Cloud Console will provide HTTP trigger URLs for your callable functions.
    *   Example URL format: `https_YOUR_REGION-YOUR_PROJECT_ID.cloudfunctions.net/functionName`
4.  **Update `app.py`:** Replace the placeholder values for `GET_PREFS_URL`, `SET_PREFS_URL`, and `SEND_EMAIL_URL` with these actual deployed function URLs.

**Step 3: Build and Push Docker Image for Gradio App**

1.  **Enable Artifact Registry API:** In your Google Cloud project, ensure the Artifact Registry API is enabled.
2.  **Create a Docker Repository:**
    ```bash
    gcloud artifacts repositories create YOUR_REPO_NAME --repository-format=docker \
        --location=YOUR_REGION --description="Docker repository for Gradio app"
    ```
    (e.g., `YOUR_REPO_NAME` could be `gradio-apps`, `YOUR_REGION` like `us-central1`).
3.  **Build the Docker Image:**
    From the root of your repository (where the `Dockerfile` is):
    ```bash
    docker build -t YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/YOUR_IMAGE_NAME:latest .
    ```
    (e.g., `YOUR_IMAGE_NAME` could be `yolov10-detector`).
4.  **Configure Docker Authentication:**
    ```bash
    gcloud auth configure-docker YOUR_REGION-docker.pkg.dev
    ```
5.  **Push the Docker Image:**
    ```bash
    docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/YOUR_IMAGE_NAME:latest
    ```

**Step 4: Deploy Gradio App to Cloud Run**

1.  Deploy the container image from Artifact Registry to Cloud Run:
    ```bash
    gcloud run deploy YOUR_GRADIO_SERVICE_NAME \
        --image YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/YOUR_IMAGE_NAME:latest \
        --platform managed \
        --region YOUR_CLOUD_RUN_REGION \
        --allow-unauthenticated \
        --port 7860
    ```
    *   `YOUR_GRADIO_SERVICE_NAME`: e.g., `yolov10-gradio-app`. This is the `serviceId` for `firebase.json`.
    *   `--allow-unauthenticated` makes the app publicly accessible. Adjust security as needed.
    *   `--port 7860`: Cloud Run sets a `PORT` environment variable. Gradio's `launch()` method (used by `app.launch()` if no port is specified) will automatically use this `PORT` if available, otherwise defaulting to 7860. The `EXPOSE 7860` in the Dockerfile informs Cloud Run which port the application *inside* the container listens on if `PORT` env var isn't heeded for some reason by the app's direct launch command.

**Step 5: Configure Firebase Hosting**

1.  **Update `firebase.json`:**
    *   Ensure `rewrites[0].run.serviceId` is set to `YOUR_GRADIO_SERVICE_NAME` (from Step 4).
    *   Ensure `rewrites[0].run.region` is set to `YOUR_CLOUD_RUN_REGION` (from Step 4).
2.  **Deploy Hosting Configuration:**
    ```bash
    firebase deploy --only hosting
    ```
    This will make your application accessible via your Firebase Hosting URL, with requests to `/app/` being routed to your Cloud Run service.

**Step 6: User Authentication (Important Note)**

*   The current `app.py` uses a `SIMULATED_USER_ID_TOKEN`. For a production system with user-specific settings:
    *   Implement a proper Firebase Authentication flow on the client-side (e.g., using JavaScript with the Firebase SDK, potentially integrated into a custom HTML page served by Firebase Hosting or within the Gradio interface if possible).
    *   The client would obtain a Firebase ID token after user login.
    *   This ID token must be securely sent to the Gradio backend (`app.py`).
    *   `app.py` would then use this real ID token in the `Authorization: Bearer <ID_TOKEN>` header when calling Firebase Cloud Functions. This allows the Cloud Functions to identify the authenticated user (`req.auth.uid`).

## Project Structure

-   `app.py`: Main Gradio application for video streaming, settings UI, and interaction logic.
-   `inference.py`: Contains the `YOLOv10` class for model loading and object detection, plus drawing utilities.
-   `requirements.txt`: Python dependencies for the Gradio application.
-   `Dockerfile`: Instructions to build the Docker container for the Gradio application.
-   `.dockerignore`: Specifies files to exclude from the Docker build context.
-   `firebase_functions/`: Directory containing the backend Firebase Cloud Functions.
    -   `firebase_functions/main.py`: Python code for Cloud Functions (user preferences, notifications).
    -   `firebase_functions/requirements.txt`: Python dependencies for the Cloud Functions.
    -   `firebase_functions/.python_version`: Specifies Python runtime for Cloud Functions.
-   `firebase.json`: Firebase project configuration for deploying Functions and Hosting.
-   `public/`: Directory for static assets served by Firebase Hosting.
    -   `public/index.html`: Basic landing page.
-   `README.md`: This file.
