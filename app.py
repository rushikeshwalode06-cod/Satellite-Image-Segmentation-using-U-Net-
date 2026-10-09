import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatelliteSeg AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS ONLY
# No HTML div elements are used in the application UI.
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 85% 10%,
            rgba(14,165,233,0.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #06101f 0%,
            #08182d 55%,
            #061b20 100%
        );
}


/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #07152a 0%,
        #06111f 100%
    );

    border-right: 1px solid
    rgba(59,130,246,0.25);
}

section[data-testid="stSidebar"] h1 {
    color: #f8fafc !important;
    font-size: 25px !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: #cbd5e1 !important;
}


/* =========================
   MAIN HEADINGS
========================= */

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label {
    color: #cbd5e1 !important;
}


/* =========================
   HEADER
========================= */

.hero-container {
    padding: 26px 30px;
    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(20,61,130,0.90),
            rgba(5,54,63,0.90)
        );

    border: 1px solid
    rgba(59,130,246,0.45);

    box-shadow:
        0 20px 50px rgba(0,0,0,0.30);

    margin-bottom: 22px;
}

.hero-title {
    font-size: 37px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 16px;
    color: #bfdbfe;
    margin-top: 5px;
}


/* =========================
   STATUS
========================= */

.status-box {
    padding: 10px 16px;
    border-radius: 30px;

    background: rgba(
        16,
        185,
        129,
        0.15
    );

    border: 1px solid
    rgba(16,185,129,0.35);

    color: #6ee7b7;

    font-weight: 700;
}


/* =========================
   SECTION
========================= */

.section-box {
    background: rgba(
        7,
        30,
        54,
        0.72
    );

    border: 1px solid
    rgba(59,130,246,0.25);

    border-radius: 20px;

    padding: 20px;

    margin-bottom: 18px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.20);
}


/* =========================
   METRICS
========================= */

[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            rgba(13,42,76,0.95),
            rgba(7,25,45,0.95)
        );

    border: 1px solid
    rgba(59,130,246,0.25);

    border-radius: 16px;

    padding: 16px;

    box-shadow:
        0 10px 25px rgba(0,0,0,0.20);
}

[data-testid="stMetricLabel"] {
    color: #93c5fd !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}


/* =========================
   UPLOADER
========================= */

[data-testid="stFileUploader"] {

    background:
        rgba(5,24,45,0.85);

    border: 1px dashed
    #38bdf8;

    border-radius: 17px;

    padding: 12px;
}


/* =========================
   BUTTON
========================= */

.stButton > button {

    width: 100%;

    min-height: 52px;

    border-radius: 14px;

    border: 1px solid
    rgba(56,189,248,0.35);

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #0891b2
        );

    color: white;

    font-size: 16px;

    font-weight: 800;

    box-shadow:
        0 10px 25px
        rgba(37,99,235,0.25);
}

.stButton > button:hover {

    background:
        linear-gradient(
            90deg,
            #1d4ed8,
            #0284c7
        );

    border-color: #38bdf8;

    color: white;
}


/* =========================
   DOWNLOAD
========================= */

.stDownloadButton > button {

    width: 100%;

    min-height: 48px;

    border-radius: 13px;

    background:
        rgba(8,47,73,0.9);

    color: #7dd3fc;

    border: 1px solid
    rgba(56,189,248,0.35);

    font-weight: 700;
}


/* =========================
   IMAGES
========================= */

[data-testid="stImage"] img {

    border-radius: 15px;

    border: 1px solid
    rgba(59,130,246,0.25);
}


/* =========================
   DIVIDER
========================= */

hr {
    border-color:
    rgba(148,163,184,0.12);
}


/* =========================
   SLIDER
========================= */

.stSlider [data-baseweb="slider"] {
    padding-top: 10px;
}


/* =========================
   FOOTER
========================= */

.footer-text {
    text-align: center;
    color: #64748b;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "unet_model.keras"

IMAGE_SIZE = (256, 256)


# ============================================================
# CUSTOM METRICS / LOSS
# ============================================================

def dice_coefficient(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.reshape(
        y_true,
        [-1]
    )

    y_pred = tf.reshape(
        y_pred,
        [-1]
    )

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
        +
        smooth
    )


def iou_metric(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.reshape(
        y_true,
        [-1]
    )

    y_pred = tf.reshape(
        y_pred,
        [-1]
    )

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    union = (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
        -
        intersection
    )

    return (
        intersection + smooth
    ) / (
        union + smooth
    )


def dice_loss(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.reshape(
        y_true,
        [-1]
    )

    y_pred = tf.reshape(
        y_pred,
        [-1]
    )

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
        +
        smooth
    )

    return 1.0 - dice


def combined_loss(y_true, y_pred):

    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
    )

    return bce + dice_loss(
        y_true,
        y_pred
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient,
            "iou_metric": iou_metric
        },
        compile=False
    )


try:

    model = load_model()

    model_loaded = True

except Exception as e:

    model = None

    model_loaded = False

    model_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛰️ SatelliteSeg AI")

    st.caption(
        "AI-powered satellite image "
        "segmentation dashboard"
    )

    st.divider()

    st.subheader("🧠 Model Information")

    st.markdown(
        """
        **Architecture**  
        🔹 U-Net

        **Encoder**  
        🔹 MobileNetV2

        **Task**  
        🔹 Building Segmentation

        **Input Size**  
        🔹 256 × 256

        **Output**  
        🔹 Binary Mask
        """
    )

    st.divider()

    st.subheader("⚙️ Processing Pipeline")

    st.markdown(
        """
        🔵 **1** — Upload satellite image

        🔵 **2** — Resize to 256 × 256

        🔵 **3** — U-Net prediction

        🔵 **4** — Generate building mask

        🔵 **5** — Create overlay
        """
    )

    st.divider()

    st.subheader("📁 Supported Images")

    st.write("🖼️ JPG")
    st.write("🖼️ JPEG")
    st.write("🖼️ PNG")

    st.divider()

    if model_loaded:

        st.success(
            "🟢 U-Net Model Loaded"
        )

    else:

        st.error(
            "🔴 Model Not Loaded"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-container">
    """,
    unsafe_allow_html=True
)

st.markdown(
    "🛰️ Satellite Image Segmentation"
)

st.caption(
    "AI-powered building detection using "
    "U-Net + MobileNetV2"
)

if model_loaded:

    st.success(
        "🟢 MODEL READY  •  DEEP LEARNING"
    )

else:

    st.error(
        "🔴 MODEL NOT READY"
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# MODEL ERROR
# ============================================================

if not model_loaded:

    st.error(
        "The model could not be loaded."
    )

    st.warning(
        "Place unet_model.keras in the "
        "same folder as app.py."
    )

    st.code(model_error)

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.header("📤 Upload Satellite Image")

uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(image):

    original_size = image.size

    image = image.convert(
        "RGB"
    )

    resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    array = np.array(
        resized,
        dtype=np.float32
    )

    array = array / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array, original_size


# ============================================================
# PREDICTION
# ============================================================

def predict_segmentation(image):

    processed_image, original_size = (
        preprocess_image(image)
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    if prediction.ndim == 4:

        prediction = prediction[0]

    if prediction.shape[-1] == 1:

        probability = prediction[:, :, 0]

    else:

        probability = np.max(
            prediction,
            axis=-1
        )

    return probability


# ============================================================
# CREATE MASK
# ============================================================

def create_mask(
    probability,
    threshold
):

    return (
        probability >= threshold
    ).astype(np.uint8)


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    mask
):

    original = np.array(
        image.convert("RGB")
    )

    mask_image = Image.fromarray(
        (
            mask * 255
        ).astype(np.uint8)
    )

    mask_image = mask_image.resize(
        image.size,
        Image.Resampling.NEAREST
    )

    mask_array = np.array(
        mask_image
    )

    result = original.copy()

    building = (
        mask_array > 127
    )

    # Red highlight for detected buildings
    result[
        building,
        0
    ] = 255

    result[
        building,
        1
    ] = (
        result[
            building,
            1
        ] * 0.25
    ).astype(np.uint8)

    result[
        building,
        2
    ] = (
        result[
            building,
            2
        ] * 0.25
    ).astype(np.uint8)

    overlay = (
        0.65 * original
        +
        0.35 * result
    ).astype(np.uint8)

    return overlay


# ============================================================
# IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Image error: {e}"
        )

        st.stop()


    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    st.divider()

    st.header("📊 Image Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Image Width",
            f"{image.width}px"
        )

    with c2:

        st.metric(
            "Image Height",
            f"{image.height}px"
        )

    with c3:

        st.metric(
            "Model Input",
            "256 × 256"
        )

    with c4:

        st.metric(
            "Model",
            "U-Net"
        )


    # ========================================================
    # INPUT IMAGE
    # ========================================================

    st.divider()

    st.header("🌍 Input Satellite Image")

    st.image(
        image,
        caption="Uploaded Satellite Image",
        use_container_width=True
    )


    # ========================================================
    # SETTINGS
    # ========================================================

    st.divider()

    st.header("⚙️ Segmentation Settings")

    threshold = st.slider(
        "Building Detection Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05
    )


    # ========================================================
    # RUN BUTTON
    # ========================================================

    st.divider()

    if st.button(
        "🚀 RUN U-NET SEGMENTATION",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 AI is analyzing the satellite image..."
        ):

            probability = predict_segmentation(
                image
            )

            mask = create_mask(
                probability,
                threshold
            )

            overlay = create_overlay(
                image,
                mask
            )

        st.session_state["mask"] = mask
        st.session_state["overlay"] = overlay
        st.session_state["image"] = image

        st.success(
            "✅ Segmentation completed successfully!"
        )


# ============================================================
# RESULTS
# ============================================================

if "mask" in st.session_state:

    mask = st.session_state[
        "mask"
    ]

    overlay = st.session_state[
        "overlay"
    ]


    # ========================================================
    # ANALYTICS
    # ========================================================

    st.divider()

    st.header("🎯 Segmentation Results")

    total_pixels = mask.size

    segmented_pixels = int(
        np.sum(mask)
    )

    coverage = (
        segmented_pixels /
        total_pixels
    ) * 100


    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Total Pixels",
            f"{total_pixels:,}"
        )

    with m2:

        st.metric(
            "Segmented Pixels",
            f"{segmented_pixels:,}"
        )

    with m3:

        st.metric(
            "Building Coverage",
            f"{coverage:.2f}%"
        )

    with m4:

        st.metric(
            "Architecture",
            "U-Net"
        )


    # ========================================================
    # TWO RESULT COLUMNS
    # ========================================================

    st.divider()

    left_result, right_result = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # MASK
    # ========================================================

    with left_result:

        st.subheader(
            "🎭 Segmentation Mask"
        )

        display_mask = (
            mask * 255
        ).astype(np.uint8)

        st.image(
            display_mask,
            caption="Predicted Building Mask",
            use_container_width=True
        )


    # ========================================================
    # OVERLAY
    # ========================================================

    with right_result:

        st.subheader(
            "🔎 Segmentation Overlay"
        )

        st.image(
            overlay,
            caption="Satellite Image + Building Segmentation",
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.divider()

    st.header("💾 Download Results")

    d1, d2 = st.columns(2)

    with d1:

        mask_buffer = io.BytesIO()

        Image.fromarray(
            display_mask
        ).save(
            mask_buffer,
            format="PNG"
        )

        st.download_button(
            "⬇️ Download Segmentation Mask",
            data=mask_buffer.getvalue(),
            file_name="segmentation_mask.png",
            mime="image/png",
            use_container_width=True
        )

    with d2:

        overlay_buffer = io.BytesIO()

        Image.fromarray(
            overlay
        ).save(
            overlay_buffer,
            format="PNG"
        )

        st.download_button(
            "⬇️ Download Overlay",
            data=overlay_buffer.getvalue(),
            file_name="segmentation_overlay.png",
            mime="image/png",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛰️ SatelliteSeg AI  •  "
    "U-Net + MobileNetV2  •  "
    "Deep Learning Image Segmentation"
)