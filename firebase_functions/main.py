import firebase_admin
from firebase_admin import credentials, firestore
from firebase_functions import https_fn, options

# It's good practice to initialize Firebase Admin SDK once per instance.
# However, in a serverless environment, this might run per invocation if not warm.
# For simplicity in this generated snippet, we assume it's handled.
# If firebase_admin.get_app() throws, then firebase_admin.initialize_app() is needed.
try:
    firebase_admin.get_app()
except ValueError:
    # Attempt to initialize with application default credentials,
    # which work well in Cloud Functions environment.
    # For local testing, GOOGLE_APPLICATION_CREDENTIALS env var would need to be set.
    firebase_admin.initialize_app()


# Global CORS options - Adjust origins for production
# These settings are often placed directly in the function deployment definition
# or handled by API Gateway / Firebase Hosting rewrites.
# For @https_fn.on_call, CORS is typically handled automatically.
# options.set_global_options(
#     region=options.SupportedRegion.EUROPE_WEST1, # Example
#     cors=options.CorsOptions(cors_origins=["http://localhost:7860", "YOUR_FIREBASE_HOSTING_URL"], cors_methods=["get", "post", "options"])
# )

@https_fn.on_call()
def get_user_preferences(req: https_fn.CallableRequest) -> any:
    """
    Retrieves user preferences from Firestore.
    User is authenticated via Firebase Auth context provided by Callable Functions.
    """
    if req.auth is None:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.UNAUTHENTICATED,
            message="Authentication required to retrieve preferences."
        )

    uid = req.auth.uid
    db = firestore.client()

    try:
        pref_doc_ref = db.collection("users").document(uid).collection("preferences").document("user_preferences")
        prefs_doc = pref_doc_ref.get()

        if prefs_doc.exists:
            return prefs_doc.to_dict()
        else:
            # Define and return default preferences if none exist
            default_prefs = {
                "watchedObjects": {}, # e.g., {"car": True, "person": False}
                "objectActions": {},  # e.g., {"car": {"count": True, "notify": False}}
                "notificationEmail": req.auth.token.get("email") # Default to user's email
            }
            return default_prefs
    except Exception as e:
        # Log the error for debugging
        print(f"Error fetching preferences for UID {uid}: {str(e)}")
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message="An internal error occurred while fetching preferences."
        )

@https_fn.on_call()
def set_user_preferences(req: https_fn.CallableRequest) -> any:
    """
    Sets or updates user preferences in Firestore.
    Data is passed in req.data.
    User is authenticated via Firebase Auth context.
    """
    if req.auth is None:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.UNAUTHENTICATED,
            message="Authentication required to set preferences."
        )

    uid = req.auth.uid
    data_to_set = req.data
    db = firestore.client()

    if not isinstance(data_to_set, dict) or not data_to_set:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Invalid data payload; expected a non-empty dictionary."
        )

    try:
        # Server-side timestamp for tracking updates
        data_to_set['lastUpdated'] = firestore.SERVER_TIMESTAMP

        # Reference to the user's main document (to store email, creation date)
        user_doc_ref = db.collection("users").document(uid)
        # Reference to the preferences document
        pref_doc_ref = user_doc_ref.collection("preferences").document("user_preferences")

        # Transaction to ensure atomicity if creating user doc and prefs doc
        @firestore.transactional
        def update_prefs_in_transaction(transaction, user_ref, prefs_ref, data):
            user_snapshot = user_ref.get(transaction=transaction)
            if not user_snapshot.exists:
                user_data = {
                    "uid": uid,
                    "createdAt": firestore.SERVER_TIMESTAMP
                }
                # Only set email if available in token (it usually is)
                if req.auth.token and req.auth.token.get("email"):
                    user_data["email"] = req.auth.token.get("email")
                transaction.set(user_ref, user_data)

            transaction.set(prefs_ref, data, merge=True) # merge=True to create or update

        update_prefs_in_transaction(db.transaction(), user_doc_ref, pref_doc_ref, data_to_set)

        return {"message": "Preferences updated successfully."}
    except Exception as e:
        # Log the error for debugging
        print(f"Error setting preferences for UID {uid}: {str(e)}")
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message="An internal error occurred while updating preferences."
        )

@https_fn.on_call()
def send_email_notification(req: https_fn.CallableRequest) -> any:
    """
    Simulates sending an email notification.
    Expects recipient_email, subject, and body in req.data.
    User should be authenticated.
    """
    if req.auth is None:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.UNAUTHENTICATED,
            message="Authentication required to send notifications."
        )

    recipient_email = req.data.get("recipient_email")
    subject = req.data.get("subject")
    body = req.data.get("body")

    if not recipient_email or not isinstance(recipient_email, str):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Missing or invalid 'recipient_email'."
        )
    if not subject or not isinstance(subject, str):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Missing or invalid 'subject'."
        )
    if not body or not isinstance(body, str):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Missing or invalid 'body'."
        )

    # Simulate sending email
    print("---- SIMULATING EMAIL SEND ----")
    print(f"TO: {recipient_email}")
    print(f"SUBJECT: {subject}")
    print(f"BODY: {body}")
    print("-------------------------------")

    # Placeholder: How to integrate with "Trigger Email" Firebase Extension
    # This extension typically works by creating a document in a specific Firestore collection.
    # Example:
    # db = firestore.client()
    # mail_doc_ref = db.collection("mail").add({
    #     "to": [recipient_email],
    #     "message": {
    #         "subject": subject,
    #         "html": body, # Or "text": body
    #     },
    #     # Optionally add more fields like 'template', 'cc', 'bcc', etc.
    #     # "createdAt": firestore.SERVER_TIMESTAMP, # Good for tracking
    # })
    # print(f"Mail document created with ID: {mail_doc_ref[1].id} (simulation only)")

    return {"message": f"Email simulation successful for: {recipient_email}"}

# For local testing with Firebase Emulator Suite:
# 1. Install Firebase CLI: `npm install -g firebase-tools`
# 2. Login: `firebase login`
# 3. Initialize Firebase in your project: `firebase init firestore` and `firebase init functions` (select Python)
# 4. Place this code in the functions directory (e.g., `functions/main.py`)
# 5. Install dependencies: `pip install firebase-admin firebase-functions` in `functions` dir.
# 6. Start emulators: `firebase emulators:start`
#
# To deploy:
# `firebase deploy --only functions`
