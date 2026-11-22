# Coffee Identification System

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import json
import time
import io
import hashlib

from pathlib import Path
from PIL import Image
from collections import Counter
from datetime import datetime
from ultralytics import YOLO


# Application configuration
st.set_page_config(
    page_title="PHOTO COFFEE",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Project and model paths
PROJECT_DIR = Path(
    r"D:\Isan-1\.Photo Coffee\Dataset\Photo Coffeeyolov8"
)

MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "PHOTO_COFFEE_YOLOv8_SEG"
    / "weights"
    / "best.pt"
)


# Detection history directory
HISTORY_DIR = PROJECT_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# Coffee classification labels
CLASS_NAMES = [
    "Arabika",
    "Ekselsa",
    "Liberika",
    "Robusta"
]


# Application color palette
COLORS = {
    "background": "#F5F1E8",
    "primary": "#79513A",
    "dark": "#34241B",
    "secondary": "#897968",
    "accent": "#C28A57",
    "success": "#667D60",
    "light": "#FFFCF6"
}


# Custom styling
st.markdown("""
<style>

.stApp {
    background-color: #F5F1E8;
}

[data-testid="stSidebar"] {
    background-color: #261A14;
}

[data-testid="stSidebar"] * {
    color: #F5F1E8;
}

[data-testid="stMetric"] {
    background-color: #FFFCF6;
    border: 1px solid #E5D9C9;
    padding: 15px;
    border-radius: 14px;
}

h1, h2, h3 {
    color: #34241B;
}

div.stButton > button {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# Initialize session state
# Session state stores temporary application data
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "analysis_image_id" not in st.session_state:
    st.session_state.analysis_image_id = None

if "history_saved_id" not in st.session_state:
    st.session_state.history_saved_id = None


# Load the YOLO model
# Cache the model to prevent repeated loading
@st.cache_resource(show_spinner="Loading PHOTO COFFEE AI Model...")
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    return model


# Check model availability
def check_model_status():

    return MODEL_PATH.exists()


# Create a unique identifier for an image
# Used to prevent showing results from a different image
def get_image_identifier(image):

    image_array = np.array(
        image.convert("RGB")
    )

    image_bytes = image_array.tobytes()

    return hashlib.md5(
        image_bytes
    ).hexdigest()


# Convert PIL image into downloadable PNG bytes
def convert_image_to_bytes(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# Extract detections from YOLO prediction results
def extract_detections(result, model_names):

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        class_id = int(box.cls.item())

        confidence = float(box.conf.item())

        # Retrieve the class name from the trained model
        if isinstance(model_names, dict):

            class_name = model_names.get(
                class_id,
                f"Class {class_id}"
            )

        else:

            if class_id < len(model_names):
                class_name = model_names[class_id]
            else:
                class_name = f"Class {class_id}"

        detections.append({
            "class_id": class_id,
            "class_name": str(class_name),
            "confidence": confidence
        })

    return detections


# Calculate detection statistics
def calculate_summary(detections):

    class_counts = Counter()

    confidence_values = []

    for detection in detections:

        class_counts[
            detection["class_name"]
        ] += 1

        confidence_values.append(
            detection["confidence"]
        )

    total_objects = len(detections)

    detected_classes = len(class_counts)

    average_confidence = (
        float(np.mean(confidence_values) * 100)
        if confidence_values
        else 0.0
    )

    return {
        "class_counts": class_counts,
        "total_objects": total_objects,
        "detected_classes": detected_classes,
        "average_confidence": average_confidence
    }


# Save detection results into history
def save_detection_history(
    annotated_image,
    detections,
    summary,
    inference_time,
    confidence_threshold,
    image_size
):

    current_time = datetime.now()

    # Generate a unique timestamp for each detection
    unique_id = current_time.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_filename = (
        f"PHOTO_COFFEE_{unique_id}.png"
    )

    json_filename = (
        f"PHOTO_COFFEE_{unique_id}.json"
    )

    image_path = HISTORY_DIR / image_filename

    json_path = HISTORY_DIR / json_filename

    # Save the segmentation result
    annotated_pil = Image.fromarray(
        annotated_image
    )

    annotated_pil.save(
        image_path,
        format="PNG"
    )

    # Prepare detection metadata
    history_record = {

        "application": "PHOTO COFFEE",

        "timestamp": current_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "total_objects": summary["total_objects"],

        "detected_classes": summary["detected_classes"],

        "average_confidence": summary["average_confidence"],

        "inference_time": inference_time,

        "confidence_threshold": confidence_threshold,

        "image_size": image_size,

        "class_counts": {
            name: summary["class_counts"].get(name, 0)
            for name in CLASS_NAMES
        },

        "detections": detections,

        "image_file": image_filename
    }

    # Save metadata as JSON
    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_record,
            file,
            indent=4,
            ensure_ascii=False
        )

    return unique_id


# Load detection history from JSON files
def read_detection_history():

    records = []

    json_files = list(
        HISTORY_DIR.glob("*.json")
    )

    # Sort history by newest modification time
    json_files.sort(
        key=lambda file: file.stat().st_mtime,
        reverse=True
    )

    for file in json_files:

        try:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as json_file:

                record = json.load(json_file)

                records.append(record)

        except Exception as error:

            print(
                f"Could not read history: {error}"
            )

    return records


# Sidebar navigation
with st.sidebar:

    st.title("☕ PHOTO COFFEE")

    st.caption(
        "Coffee Identification System"
    )

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Coffee Scanner",
            "Analytics",
            "Detection History",
            "About System"
        ]
    )

    st.divider()

    # Show model status
    if check_model_status():

        st.success("AI Model Online")

    else:

        st.error("AI Model Not Found")

    st.caption(
        "YOLOv8 Instance Segmentation"
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================
if selected_page == "Dashboard":

    st.title("☕ PHOTO COFFEE")

    st.subheader(
        "Coffee Identification System"
    )

    st.info(
        "An intelligent coffee bean identification "
        "system powered by computer vision and deep learning."
    )

    # Read history data for dashboard metrics
    records = read_detection_history()

    total_sessions = len(records)

    total_objects = sum(
        record.get("total_objects", 0)
        for record in records
    )

    # Dashboard metrics
    st.markdown("### System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Coffee Classes",
        "4",
        "Classification categories"
    )

    col2.metric(
        "Detection Sessions",
        total_sessions
    )

    col3.metric(
        "Total Objects",
        total_objects
    )

    col4.metric(
        "Model Architecture",
        "YOLOv8"
    )

    st.divider()

    # Display supported coffee varieties
    st.markdown("### Coffee Varieties")

    coffee_details = {
        "Arabika": "Arabica Coffee",
        "Ekselsa": "Excelsa Coffee",
        "Liberika": "Liberica Coffee",
        "Robusta": "Robusta Coffee"
    }

    class_columns = st.columns(4)

    for index, name in enumerate(CLASS_NAMES):

        with class_columns[index]:

            with st.container(border=True):

                st.markdown("## ☕")

                st.markdown(
                    f"### {name}"
                )

                st.caption(
                    coffee_details[name]
                )

    st.divider()

    # Display system workflow
    st.markdown("### System Workflow")

    workflow = [
        ("01", "Image Input", "Upload or capture an image"),
        ("02", "Detection", "AI object detection"),
        ("03", "Segmentation", "Object mask generation"),
        ("04", "Classification", "Coffee variety prediction")
    ]

    workflow_columns = st.columns(4)

    for index, item in enumerate(workflow):

        with workflow_columns[index]:

            with st.container(border=True):

                st.caption(item[0])

                st.markdown(
                    f"### {item[1]}"
                )

                st.write(item[2])

    st.divider()

    st.markdown("### Getting Started")

    st.write(
        "Navigate to Coffee Scanner from the sidebar "
        "to upload an image or capture a photo "
        "of coffee beans."
    )


# ============================================================
# COFFEE SCANNER PAGE
# ============================================================
elif selected_page == "Coffee Scanner":

    st.title("Coffee Scanner")

    st.caption(
        "Detect, segment, and classify coffee beans."
    )

    # Select image input method
    input_method = st.radio(
        "Image Input",
        [
            "Upload Image",
            "Camera"
        ],
        horizontal=True
    )

    selected_image = None

    # Upload image
    if input_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload coffee bean image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key="coffee_uploader"
        )

        if uploaded_file is not None:

            try:

                selected_image = Image.open(
                    uploaded_file
                ).convert("RGB")

            except Exception:

                st.error(
                    "The uploaded image could not be opened."
                )

    # Capture image using the camera
    elif input_method == "Camera":

        camera_file = st.camera_input(
            "Take a picture of coffee beans",
            key="coffee_camera"
        )

        if camera_file is not None:

            try:

                selected_image = Image.open(
                    camera_file
                ).convert("RGB")

            except Exception:

                st.error(
                    "The captured image could not be opened."
                )

    # Detection configuration
    st.markdown("### Detection Settings")

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Minimum confidence score required for a detection."
        )

    with config_col2:

        image_size = st.selectbox(
            "Image Size",
            [320, 480, 640, 960],
            index=2
        )

    # Preview the selected image
    if selected_image is not None:

        st.markdown("### Image Preview")

        st.image(
            selected_image,
            caption="Input Image",
            use_container_width=True
        )

        # Generate an ID to identify the current image
        current_image_id = get_image_identifier(
            selected_image
        )

        # Run YOLO detection
        if st.button(
            "🔍 RUN DETECTION",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "PHOTO COFFEE is analyzing your image..."
                ):

                    # Load trained model
                    model = load_model()

                    image_array = np.array(
                        selected_image.convert("RGB")
                    )

                    # Start inference timer
                    start_time = time.perf_counter()

                    # Run model prediction
                    results = model.predict(
                        source=image_array,
                        imgsz=image_size,
                        conf=confidence_threshold,
                        verbose=False
                    )

                    # Calculate inference time
                    inference_time = (
                        time.perf_counter()
                        - start_time
                    )

                    result = results[0]

                    # Extract detection results
                    detections = extract_detections(
                        result,
                        model.names
                    )

                    # Calculate statistics
                    summary = calculate_summary(
                        detections
                    )

                    # Generate segmentation visualization
                    plotted_image = result.plot()

                    # Convert BGR to RGB
                    annotated_image = cv2.cvtColor(
                        plotted_image,
                        cv2.COLOR_BGR2RGB
                    )

                    # Store analysis in session state
                    st.session_state.analysis = {
                        "detections": detections,
                        "summary": summary,
                        "inference_time": inference_time,
                        "confidence_threshold": confidence_threshold,
                        "image_size": image_size,
                        "annotated_image": annotated_image
                    }

                    st.session_state.analysis_image_id = (
                        current_image_id
                    )

                    # Save detection into history
                    history_id = save_detection_history(
                        annotated_image=annotated_image,
                        detections=detections,
                        summary=summary,
                        inference_time=inference_time,
                        confidence_threshold=confidence_threshold,
                        image_size=image_size
                    )

                    st.session_state.history_saved_id = (
                        history_id
                    )

                st.success(
                    "Detection completed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Detection error: {error}"
                )

                st.exception(error)

        # Display result only for the current image
        if (
            st.session_state.analysis is not None
            and st.session_state.analysis_image_id
            == current_image_id
        ):

            analysis = st.session_state.analysis

            summary = analysis["summary"]

            detections = analysis["detections"]

            annotated_image = analysis["annotated_image"]

            st.divider()

            st.markdown("## Detection Results")

            # Compare original image and segmentation output
            original_col, segmentation_col = st.columns(2)

            with original_col:

                st.markdown("### Original Image")

                st.image(
                    selected_image,
                    use_container_width=True
                )

            with segmentation_col:

                st.markdown("### Segmentation Result")

                st.image(
                    annotated_image,
                    use_container_width=True
                )

            st.divider()

            # Detection summary metrics
            st.markdown("### Detection Summary")

            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric(
                "Total Objects",
                summary["total_objects"]
            )

            metric2.metric(
                "Classes Found",
                summary["detected_classes"]
            )

            metric3.metric(
                "Average Confidence",
                f"{summary['average_confidence']:.2f}%"
            )

            metric4.metric(
                "Inference Time",
                f"{analysis['inference_time']:.3f} s"
            )

            st.divider()

            # Display class distribution
            st.markdown("### Coffee Variety Distribution")

            distribution_data = pd.DataFrame({
                "Coffee Variety": CLASS_NAMES,
                "Detected Objects": [
                    summary["class_counts"].get(
                        name,
                        0
                    )
                    for name in CLASS_NAMES
                ]
            })

            st.bar_chart(
                distribution_data.set_index(
                    "Coffee Variety"
                )
            )

            # Display object-level information
            st.markdown("### Detection Details")

            if detections:

                detection_table = pd.DataFrame(
                    [
                        {
                            "Object": f"Object {index + 1}",
                            "Coffee Variety": item["class_name"],
                            "Confidence": (
                                f"{item['confidence'] * 100:.2f}%"
                            )
                        }
                        for index, item in enumerate(
                            detections
                        )
                    ]
                )

                st.dataframe(
                    detection_table,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "No coffee beans were detected. "
                    "Try lowering the confidence threshold."
                )

            # Display model processing parameters
            st.markdown("### Processing Information")

            info_col1, info_col2, info_col3 = st.columns(3)

            info_col1.metric(
                "Confidence Threshold",
                f"{analysis['confidence_threshold']:.2f}"
            )

            info_col2.metric(
                "Image Size",
                analysis["image_size"]
            )

            info_col3.metric(
                "Inference Time",
                f"{analysis['inference_time']:.3f} seconds"
            )

            # Download segmentation result
            st.markdown("### Export Result")

            result_pil = Image.fromarray(
                annotated_image
            )

            st.download_button(
                label="Download Segmentation Image",
                data=convert_image_to_bytes(
                    result_pil
                ),
                file_name="PHOTO_COFFEE_Result.png",
                mime="image/png",
                use_container_width=True
            )

    else:

        st.info(
            "Upload an image or capture a photo "
            "to start coffee detection."
        )


# ============================================================
# ANALYTICS PAGE
# ============================================================
elif selected_page == "Analytics":

    st.title("Detection Analytics")

    st.caption(
        "Statistical overview of coffee bean detection results."
    )

    analysis = st.session_state.analysis

    # Check whether an analysis is available
    if analysis is None:

        st.info(
            "No analysis available. Run a detection "
            "from Coffee Scanner first."
        )

    else:

        summary = analysis["summary"]

        detections = analysis["detections"]

        # Display overall metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Objects",
            summary["total_objects"]
        )

        col2.metric(
            "Classes Found",
            summary["detected_classes"]
        )

        col3.metric(
            "Average Confidence",
            f"{summary['average_confidence']:.2f}%"
        )

        col4.metric(
            "Inference Time",
            f"{analysis['inference_time']:.3f} s"
        )

        st.divider()

        # Coffee class distribution
        st.markdown("### Coffee Variety Distribution")

        analytics_data = pd.DataFrame({
            "Coffee Variety": CLASS_NAMES,
            "Detected Objects": [
                summary["class_counts"].get(
                    name,
                    0
                )
                for name in CLASS_NAMES
            ]
        })

        st.bar_chart(
            analytics_data.set_index(
                "Coffee Variety"
            )
        )

        # Calculate percentage for each class
        st.markdown("### Percentage Distribution")

        if summary["total_objects"] > 0:

            percentage_data = pd.DataFrame({
                "Coffee Variety": CLASS_NAMES,
                "Count": [
                    summary["class_counts"].get(
                        name,
                        0
                    )
                    for name in CLASS_NAMES
                ],
                "Percentage": [
                    round(
                        summary["class_counts"].get(
                            name,
                            0
                        )
                        / summary["total_objects"]
                        * 100,
                        2
                    )
                    for name in CLASS_NAMES
                ]
            })

            st.dataframe(
                percentage_data,
                use_container_width=True,
                hide_index=True
            )

        # Confidence analysis for each detected object
        st.markdown("### Object Confidence Analysis")

        if detections:

            confidence_data = pd.DataFrame(
                [
                    {
                        "Object": f"Object {index + 1}",
                        "Coffee Variety": item["class_name"],
                        "Confidence (%)": round(
                            item["confidence"] * 100,
                            2
                        )
                    }
                    for index, item in enumerate(
                        detections
                    )
                ]
            )

            st.dataframe(
                confidence_data,
                use_container_width=True,
                hide_index=True
            )

            # Visualize confidence values
            st.markdown("### Confidence Distribution")

            st.bar_chart(
                confidence_data.set_index(
                    "Object"
                )["Confidence (%)"]
            )

        else:

            st.warning(
                "No detected objects are available for analysis."
            )


# ============================================================
# DETECTION HISTORY PAGE
# ============================================================
elif selected_page == "Detection History":

    st.title("Detection History")

    st.caption(
        "Browse previously saved detection sessions."
    )

    history_records = read_detection_history()

    # Display number of saved sessions
    st.metric(
        "Total Saved Sessions",
        len(history_records)
    )

    if not history_records:

        st.info(
            "No history available. Perform your first "
            "detection from the Coffee Scanner page."
        )

    else:

        st.divider()

        # Display saved records
        for index, record in enumerate(history_records):

            timestamp = record.get(
                "timestamp",
                "Unknown timestamp"
            )

            with st.expander(
                f"☕ Session {index + 1} | {timestamp}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### Session Summary")

                    st.write(
                        "Total Objects:",
                        record.get(
                            "total_objects",
                            0
                        )
                    )

                    st.write(
                        "Detected Classes:",
                        record.get(
                            "detected_classes",
                            0
                        )
                    )

                    st.write(
                        "Average Confidence:",
                        f"{record.get('average_confidence', 0):.2f}%"
                    )

                    st.write(
                        "Inference Time:",
                        f"{record.get('inference_time', 0):.3f} seconds"
                    )

                    st.markdown("### Class Counts")

                    class_counts = record.get(
                        "class_counts",
                        {}
                    )

                    for name in CLASS_NAMES:

                        st.write(
                            f"**{name}:** "
                            f"{class_counts.get(name, 0)}"
                        )

                with col2:

                    # Load saved segmentation image
                    image_filename = record.get(
                        "image_file",
                        ""
                    )

                    image_path = (
                        HISTORY_DIR
                        / image_filename
                    )

                    if image_path.exists():

                        history_image = Image.open(
                            image_path
                        ).convert("RGB")

                        st.image(
                            history_image,
                            caption="Saved Segmentation Result",
                            use_container_width=True
                        )

                        # Download historical image
                        st.download_button(
                            label="Download Image",
                            data=image_path.read_bytes(),
                            file_name=image_filename,
                            mime="image/png",
                            key=f"history_download_{index}"
                        )

                # Show individual object detections
                saved_detections = record.get(
                    "detections",
                    []
                )

                if saved_detections:

                    st.markdown("### Detection Details")

                    history_table = pd.DataFrame(
                        [
                            {
                                "Object": f"Object {i + 1}",
                                "Coffee Variety": item.get(
                                    "class_name",
                                    "Unknown"
                                ),
                                "Confidence": (
                                    f"{item.get('confidence', 0) * 100:.2f}%"
                                )
                            }
                            for i, item in enumerate(
                                saved_detections
                            )
                        ]
                    )

                    st.dataframe(
                        history_table,
                        use_container_width=True,
                        hide_index=True
                    )


# ============================================================
# ABOUT SYSTEM PAGE
# ============================================================
elif selected_page == "About System":

    st.title("About PHOTO COFFEE")

    st.caption(
        "Coffee Identification System"
    )

    st.info(
        "PHOTO COFFEE is an AI-powered application "
        "for coffee bean detection, segmentation, "
        "and variety classification."
    )

    st.divider()

    # System description
    st.markdown("### System Overview")

    st.write(
        """
        PHOTO COFFEE is a computer vision application
        developed to identify coffee bean varieties
        through digital image processing and deep learning.

        The system uses a trained YOLOv8 Instance
        Segmentation model to locate coffee beans,
        generate segmentation masks, and classify
        detected objects into their corresponding varieties.
        """
    )

    # Supported coffee varieties
    st.markdown("### Classification Categories")

    class_columns = st.columns(4)

    coffee_details = {
        "Arabika": "Arabica Coffee",
        "Ekselsa": "Excelsa Coffee",
        "Liberika": "Liberica Coffee",
        "Robusta": "Robusta Coffee"
    }

    for index, name in enumerate(CLASS_NAMES):

        with class_columns[index]:

            with st.container(border=True):

                st.markdown("## ☕")

                st.markdown(
                    f"### {name}"
                )

                st.caption(
                    coffee_details[name]
                )

    st.divider()

    # Model information
    st.markdown("### Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Architecture",
        "YOLOv8"
    )

    col2.metric(
        "Task",
        "Instance Segmentation"
    )

    col3.metric(
        "Classes",
        "4 Varieties"
    )

    # System workflow
    st.markdown("### System Workflow")

    system_steps = [
        "Image Acquisition",
        "Image Input Processing",
        "Coffee Bean Detection",
        "Instance Segmentation",
        "Variety Classification",
        "Result Visualization"
    ]

    for index, step in enumerate(system_steps):

        st.write(
            f"**{index + 1}.** {step}"
        )

    st.divider()

    # Technologies used
    st.markdown("### Technologies")

    st.write(
        """
        - Python
        - Streamlit
        - Ultralytics YOLOv8
        - OpenCV
        - NumPy
        - Pandas
        - Pillow
        """
    )

    # Application purpose
    st.markdown("### Application Purpose")

    st.write(
        """
        PHOTO COFFEE provides an interactive interface
        for testing, analyzing, and visualizing coffee
        bean classification results produced by a
        trained computer vision model.
        """
    )


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "PHOTO COFFEE | Coffee Identification System | "
    "YOLOv8 Instance Segmentation"
)