# Facelock: Face Recognition & Tracking System

Facelock is an advanced face recognition and tracking system designed for robotics and security applications. It utilizes high-performance algorithms to identify enrolled users and track their movements in real-time.

## System Description

The system employs a sophisticated multi-stage pipeline:
1.  **Face Detection**: Uses OpenCV Haar Cascades for initial face localization.
2.  **Landmark Estimation**: Leverages MediaPipe FaceMesh to extract 5 key facial landmarks (eyes, nose, and mouth corners).
3.  **Face Alignment**: Performs affine transformation to align the face into a consistent 112x112 representation (ArcFace standard).
4.  **Feature Embedding**: Uses an ONNX-optimized ArcFace model to generate a unique 512-dimensional vector representing the facial identity.
5.  **Tracking & Action Detection**: Once locked onto a target, the system tracks the face's center and detects behavioral actions such as eye blinks, smiles, and head rotation (Yaw).
6.  **Servo Control**: Calculates and publishes horizontal servo angles (0-180°) based on the target's position in the frame.

## MQTT Configuration

The system communicates movement data via the MQTT protocol to control external hardware (e.g., an ESP8266-based servo gimbal).

*   **Broker IP**: `157.173.101.159`
*   **Port**: `1883`
*   **Client ID**: `python_face_tracker_user398`
*   **Topic**: `vision/user398/movement`
*   **Username (VPS)**: `user398`
*   **Password (VPS)**: `5!mQ3@zT`

## Live Dashboard

The real-time status and movement logs can be monitored via the live dashboard:

*   **Dashboard URL**: http://157.173.101.159:8399/

---

## Installation & Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Model**:
    Run the setup script to download the ArcFace ONNX model:
    ```bash
    python download_model.py
    ```
3.  **Enroll Identities**:
    Use the enrollment tool to add authorized faces to the database:
    ```bash
    python -m src.enroll
    ```
4.  **Run Tracking**:
    Launch the main tracking system:
    ```bash
    python -m src.lock_and_recognize
    ```
