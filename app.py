"""DrishtiAI: accessible AI-assisted diabetic retinopathy screening prototype."""

import base64
import os
from datetime import datetime

import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

from utils.gradcam import GradCAMError, make_gradcam_heatmap, overlay_heatmap
from utils.prediction import ModelLoadError, load_class_mapping, load_model, predict, validate_image
from utils.quality_check import technical_quality_warning
from utils.report import build_printable_report
from utils.translations import LANGUAGES, t

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "final_DR_model.keras")
CLASS_MAP_PATH = os.path.join(APP_DIR, "models", "class_mapping.json")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.60
DATASET = {"Dataset images": "1,608", "Test images": "326", "Classes": "3", "Input": "224 x 224"}
METRICS = {"Accuracy": "74.85%", "Precision": "75.32%", "Recall": "74.85%", "F1 score": "74.48%", "QWK": "74.36%"}
AUC = {"Normal": 0.9008, "NPDR": 0.8173, "PDR": 0.9472}
COMPARISON = {
    "ResNet50": (0.748466, 0.753222, 0.748466, 0.744844, 0.743574),
    "EfficientNetB0": (0.634969, 0.656511, 0.634969, 0.605289, 0.643970),
    "MobileNetV3Small": (0.625767, 0.685979, 0.625767, 0.564205, 0.674860),
    "ConvNeXtTiny": (0.619632, 0.652718, 0.619632, 0.619949, 0.560628),
}
COLORS = {"Normal": "#23856b", "NPDR": "#c8872e", "PDR": "#c45258"}

st.set_page_config(page_title="DrishtiAI | Retinal screening", page_icon="D", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#142b3a;--muted:#667a83;--paper:#f4f8f6;--white:#fff;--line:#dce8e3;--teal:#23856b;--navy:#123248;--coral:#c45258;--amber:#c8872e}
.stApp{background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif}.stApp h1,.stApp h2,.stApp h3{font-family:'Space Grotesk',sans-serif;letter-spacing:0;color:var(--navy)}
[data-testid="stSidebar"]{background:var(--navy)}[data-testid="stSidebar"] *{color:#effaf6!important}
.hero{background:linear-gradient(125deg,#123248 0%,#1b4a53 58%,#23856b 100%);color:#fff;border-radius:22px;padding:2.5rem 2.7rem;margin:.5rem 0 1.2rem;box-shadow:0 18px 40px #12324824}.hero h1{color:#fff;font-size:clamp(2rem,4vw,3.5rem);line-height:1.03;max-width:780px;margin:.2rem 0 .8rem}.hero p{color:#d7ece6;max-width:620px;font-size:1.06rem;margin:0}.eyebrow{color:#68c8a8;font-size:.72rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.2rem 1.35rem;box-shadow:0 7px 22px #1232480d;height:100%}.card-title{font-family:'Space Grotesk';font-weight:700;color:var(--navy);font-size:1.05rem;margin-bottom:.7rem}.muted{color:var(--muted);font-size:.9rem}.notice{background:#fff8e9;border:1px solid #ecdcae;border-left:4px solid var(--amber);border-radius:11px;padding:.9rem 1rem;color:#634f29;font-size:.9rem}.privacy{background:#e9f5ef;border:1px solid #b9dece;border-radius:12px;padding:1rem;color:#245b49}.dropzone{background:linear-gradient(135deg,#fff,#eef8f3);border:1px dashed #70b69f;border-radius:17px;padding:2.2rem;text-align:center}.metric{background:#fff;border:1px solid var(--line);border-radius:13px;padding:1rem 1.1rem}.metric strong{font-family:'Space Grotesk';font-size:1.7rem;color:var(--navy)}.metric span{display:block;color:var(--muted);font-size:.82rem;margin-top:.2rem}.pill{display:inline-block;color:#fff;border-radius:99px;padding:.38rem .8rem;font-weight:700}.step{background:#eff7f3;border:1px solid #d0e8dd;border-radius:12px;padding:.95rem}.step b{display:block;color:var(--teal);font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.35rem}.stButton>button{border-radius:9px;border:0;background:var(--teal);color:#fff;font-weight:700;min-height:2.7rem}.stButton>button:hover{background:#176b56;border:0}.section-rule{border-top:1px solid var(--line);margin:1.7rem 0}.small-caps{font-size:.76rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700;color:var(--teal)}
+@media(max-width:700px){.hero{padding:1.5rem}.hero h1{font-size:2rem}.card{padding:1rem}.dropzone{padding:1.5rem}}
+</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def resources():
    return load_model(MODEL_PATH), load_class_mapping(CLASS_MAP_PATH)


def metric_cards(values):
    cols = st.columns(len(values))
    for col, (label, value) in zip(cols, values.items()):
        with col:
            st.markdown(f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>', unsafe_allow_html=True)


def translated(key, fallback):
    return t(key, st.session_state.get("language", "English"), fallback)


def header(title, subtitle):
    st.markdown(f'<div class="eyebrow">DrishtiAI / {title}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="muted">{subtitle}</p>', unsafe_allow_html=True)


def urgency_content(name):
    return {
        "Normal": ("ROUTINE SCREENING", "Continue regular eye screening and follow your healthcare professional's recommendations, especially if you have diabetes.", "#23856b"),
        "NPDR": ("EYE-CARE FOLLOW-UP", "Arrange an evaluation with an eye-care professional. The AI result should be confirmed through an appropriate clinical examination.", "#c8872e"),
        "PDR": ("PROMPT EYE-CARE EVALUATION", "Seek evaluation by an eye-care professional promptly. Do not rely on this AI result alone.", "#c45258"),
    }.get(name, ("EYE-CARE FOLLOW-UP", "Please discuss this screening result with a qualified healthcare professional.", "#c8872e"))


def explanation(name):
    return {
        "Normal": "The AI model did not identify patterns associated with diabetic retinopathy in this image. This does not guarantee that your eyes are disease-free. Continue recommended diabetes and eye screening.",
        "NPDR": "The AI model identified possible patterns associated with non-proliferative diabetic retinopathy. This can affect small blood vessels at the back of the eye and should be evaluated by an eye-care professional.",
        "PDR": "The AI model identified possible patterns associated with proliferative diabetic retinopathy. This can represent a more advanced pattern and should be evaluated promptly by an eye-care professional.",
    }.get(name, "The model produced a screening category that needs professional review.")


def render_voice(result):
    name = result["predicted_class_name"]
    urgency, next_step, _ = urgency_content(name)
    text = f"AI screening result: {name}. Confidence: {result['confidence']:.1f} percent. {explanation(name)} Suggested follow-up: {urgency}. {next_step} This is an experimental screening prediction, not a medical diagnosis."
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    components.html(f"""<button aria-label="Listen to result" onclick="speak()" style="background:#123248;color:white;border:0;border-radius:9px;padding:11px 15px;font-weight:700;cursor:pointer">&#128266; Listen to Result</button><script>function speak(){{const text=atob('{encoded}'); window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));}}</script>""", height=48)


def dashboard():
    st.markdown('<div class="hero"><div class="eyebrow">Accessible screening support</div><h1>Understand the result. Know the next step.</h1><p>Upload a retinal image, understand the AI screening result, and learn what steps to consider next.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><strong>Research & Educational Prototype</strong><br>This AI system provides an experimental screening prediction based on a retinal image. It does not provide a medical diagnosis. AI predictions may be incorrect and should be confirmed by a qualified healthcare professional.</div>', unsafe_allow_html=True)
    st.write("")
    metric_cards(DATASET)
    st.write("")
    if st.button("Start New Screening", type="primary", use_container_width=False):
        st.session_state.page = "Screening"
        st.rerun()
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.subheader("A clear path from image to action")
    cols = st.columns(4)
    for col, number, title, body in zip(cols, ["01", "02", "03", "04"], ["Upload", "AI screen", "Understand", "Act"], ["Add a retinal fundus image", "See probabilities from the supplied model", "Read plain-language context", "Review an appropriate next step"]):
        col.markdown(f'<div class="step"><b>{number}</b><strong>{title}</strong><p class="muted">{body}</p></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="card"><div class="card-title">Model snapshot</div><p class="muted">The supplied ResNet50 model classifies retinal images into three research categories. These figures are reported evaluation results, not a guarantee for any individual image.</p>', unsafe_allow_html=True)
        metric_cards({"Accuracy": "74.85%", "F1 score": "74.48%", "QWK": "74.36%"})
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="privacy"><strong>Privacy first</strong><p>Images are processed locally by this app and are not sent to an external AI or voice service. Temporary results live only in this session.</p></div>', unsafe_allow_html=True)


def screening():
    header(translated("screening", "Retinal image screening"), "A local, experimental screening pass for one retinal fundus image.")
    st.markdown('<div class="notice"><strong>Before you begin:</strong> Your image is processed for AI screening. Follow-up guidance is informational and does not replace an eye specialist.</div>', unsafe_allow_html=True)
    low_bandwidth = st.session_state.get("low_bandwidth", False)
    uploaded = st.file_uploader("Upload retinal fundus image", type=["png", "jpg", "jpeg"], accept_multiple_files=False, help="JPG, JPEG, or PNG. Maximum 15 MB. On supported phones, use the camera option in the file picker.")
    if uploaded is None:
        st.markdown('<div class="dropzone"><h3>Upload a retinal image</h3><p class="muted">JPG, JPEG, or PNG · local processing · no permanent storage by default</p></div>', unsafe_allow_html=True)
        return
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("This file is larger than 15 MB. Please choose a smaller image.")
        return
    try:
        image = Image.open(uploaded)
        image.load()
    except Exception:
        st.error("The file could not be read as an image. Please upload a valid PNG or JPG.")
        return
    if not validate_image(image):
        st.error("This image is too small to analyze. Please upload an image at least 32 x 32 pixels.")
        return
    warning = technical_quality_warning(image)
    if warning:
        st.warning(f"Technical image-quality warning: {warning} This is not a clinical image-quality assessment.")
    st.image(image, caption="Original retinal image", use_container_width=True)
    if st.button(translated("analyze", "Analyze Image"), type="primary"):
        try:
            model, mapping = resources()
            with st.spinner("Running local inference and preparing an explanation..."):
                result = predict(model, mapping, image)
                heatmap, _, _ = make_gradcam_heatmap(result["preprocessed_batch"], model)
                original = result["preprocessed_batch"][0].astype("uint8")
                heatmap_color, overlay = overlay_heatmap(original, heatmap)
                st.session_state.result = result
                st.session_state.gradcam = (original, heatmap_color, overlay)
                st.session_state.screening_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        except (ModelLoadError, GradCAMError, ValueError, RuntimeError) as exc:
            st.error("Analysis could not be completed. Check the supplied model files or try another image.")
            if st.session_state.get("debug_mode"):
                st.caption(str(exc))
    render_result(low_bandwidth)


def render_result(low_bandwidth=False):
    result = st.session_state.get("result")
    if not result:
        return
    name = result["predicted_class_name"]
    confidence = result["confidence"] / 100
    urgency, next_step, urgency_color = urgency_content(name)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-caps">{translated("model_result", "AI screening result")}</div>', unsafe_allow_html=True)
    left, right = st.columns([.9, 1.4])
    with left:
        st.markdown(f'<div class="card"><div class="card-title">Predicted class</div><span class="pill" style="background:{COLORS.get(name, "#123248")}">{name}</span><h2 style="margin:.8rem 0 0">{result["confidence"]:.1f}%</h2><p class="muted">Model confidence</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="card-title">Class probabilities</div>', unsafe_allow_html=True)
        for class_name, probability in result["probabilities"].items():
            st.progress(min(max(probability / 100, 0), 1), text=f"{class_name}  {probability:.1f}%")
        st.markdown('<p class="muted">Confidence reflects the model output probabilities. It does not represent medical certainty.</p></div>', unsafe_allow_html=True)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(f"Uncertain AI prediction: the engineering confidence threshold is {LOW_CONFIDENCE_THRESHOLD:.0%}. Please seek professional evaluation rather than relying on this result. This threshold is not clinically validated.")
    st.write("")
    st.markdown(f'<div class="card"><div class="card-title">{translated("what_means", "What does this mean?")}</div><p>{explanation(name)}</p><p class="muted">This interpretation is based on an AI screening prediction and is not a confirmed diagnosis.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="card" style="border-left:5px solid {urgency_color}"><div class="card-title">{translated("what_next", "What should I do now?")}</div><p class="small-caps" style="color:{urgency_color}">{urgency}</p><p>{next_step}</p><p class="muted">Suggested follow-up based on the AI screening category. This is informational guidance, not medical triage.</p></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1])
    with left:
        render_voice(result)
    with right:
        report = build_printable_report(result, st.session_state.get("screening_time", ""), st.session_state.get("gradcam"), urgency, next_step)
        st.download_button(translated("report", "Download Screening Report"), data=report, file_name="drishtiai-screening-report.html", mime="text/html")
    st.markdown('<div class="card"><div class="card-title">Need professional help?</div><p class="muted">Connect this screening result to real care through a configured healthcare directory or local health worker. No clinic or doctor is invented by this prototype.</p>', unsafe_allow_html=True)
    st.button("Find Nearby Eye-Care Facility", disabled=True, help="Healthcare directory integration is not configured.")
    st.button("Request Specialist Review", disabled=True, help="Requires an actual healthcare-provider integration.")
    st.caption("Future integration: healthcare directory and tele-ophthalmology review pathway.")
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Red-flag symptoms that need urgent medical attention"):
        st.write("Seek urgent medical attention if you experience sudden vision loss, a sudden increase in floaters, flashes of light, a dark curtain or shadow over vision, severe eye pain, or another concerning acute symptom.")
        st.caption("These symptoms require professional medical assessment regardless of the AI prediction.")
    if not low_bandwidth:
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.subheader("Why did the AI make this prediction?")
        st.caption("Grad-CAM highlights regions that influenced the model prediction. It helps us inspect model behavior, but it is not a clinical diagnostic map.")
        images = st.session_state.get("gradcam")
        if images:
            for col, image, title in zip(st.columns(3), images, ["Original image", "Grad-CAM heatmap", "Grad-CAM overlay"]):
                with col:
                    st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)


def performance():
    header("Model performance", "Reported evaluation results for the supplied final model on the held-out test set.")
    st.caption("These figures describe the model in general. They are not computed from an uploaded image and do not represent clinical validation.")
    metric_cards(METRICS)
    st.write("")
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown('<div class="card"><div class="card-title">ROC-AUC by class</div>', unsafe_allow_html=True)
        for name, value in AUC.items():
            st.progress(value, text=f"{name}  {value:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="card-title">Evaluation scope</div><p class="muted">DR_512: 1,608 images total, with 1,141 training, 163 validation, and 326 test images. These are image counts, not patient counts.</p><p><strong>Selected model:</strong> ResNet50</p><p class="muted">No ROC curve or confusion matrix asset is present in the supplied assets folder, so neither is fabricated here.</p></div>', unsafe_allow_html=True)
    st.subheader("Model comparison")
    st.caption("Provided results, shown as proportions. ResNet50 is the selected final model.")
    for model_name, values in COMPARISON.items():
        st.markdown(f"**{model_name}** {'· Selected final model' if model_name == 'ResNet50' else ''}")
        cols = st.columns(5)
        for col, label, value in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "QWK"], values):
            col.metric(label, f"{value:.3f}")


def impact():
    header("Potential impact", "Designed around the gap between screening access, understanding, and professional care.")
    st.markdown('<div class="notice"><strong>Potential impact</strong><br>The statements below describe intended benefits of the design. They are not measured real-world outcomes.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, title, body in zip(cols * 2, ["Earlier awareness", "Accessible screening support", "Understandable AI results", "Regional language support", "Low-bandwidth design", "Connection to professional care"], ["Help users recognize when follow-up may matter.", "Support screening camps and primary healthcare workflows.", "Translate model categories into plain-language context.", "Start with English and Hindi; keep reviewed translations extensible.", "Prioritize essential results when connections are slow.", "Create a clear handoff without pretending a doctor reviewed the image."]):
        col.markdown(f'<div class="card"><div class="card-title">{title}</div><p class="muted">{body}</p></div>', unsafe_allow_html=True)
    st.subheader("The hackathon story")
    st.markdown('<div class="pipeline"><span>Access gap</span><b>→</b><span>AI screening</span><b>→</b><span>Plain language</span><b>→</b><span>Trust signals</span><b>→</b><span>Next step</span><b>→</b><span>Professional care</span></div>', unsafe_allow_html=True)


def about():
    header("How DrishtiAI works", "A transparent academic prototype for explainable retinal image screening.")
    st.markdown('<div class="card"><div class="card-title">Current workflow</div><div class="pipeline"><span>User</span><b>→</b><span>Upload</span><b>→</b><span>224 x 224 preprocessing</span><b>→</b><span>ResNet50</span><b>→</b><span>Probabilities</span><b>→</b><span>Grad-CAM</span><b>→</b><span>Suggested follow-up</span></div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card"><div class="card-title">Privacy first</div><p class="muted">Uploaded images are processed locally by this Streamlit process. The app does not send retinal images or personal information to external AI or voice APIs, and it does not persist images by default. A production deployment should document its storage, access, and retention policy explicitly.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="notice"><strong>Research & Educational Prototype</strong><br>This application is an academic/hackathon prototype for AI-assisted retinal image screening. It is not a medical diagnostic device and should not be used to make clinical decisions. AI predictions may be incorrect and should be confirmed by a qualified healthcare professional.</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# DRISHTIAI")
    st.caption("AI-assisted retinal screening")
    selected_language = st.selectbox("Language", LANGUAGES, index=LANGUAGES.index(st.session_state.get("language", "English")))
    st.session_state.language = selected_language
    st.divider()
    pages = ["Dashboard", "Screening", "Model Performance", "Potential Impact", "How It Works"]
    current = st.session_state.get("page", "Dashboard")
    page = st.radio("Navigate", pages, index=pages.index(current) if current in pages else 0)
    st.session_state.page = page
    st.toggle("Low Bandwidth Mode", key="low_bandwidth", help="Reduces non-essential visual output and prioritizes the result.")
    st.divider()
    st.caption("LOCAL INFERENCE")
    st.caption("Images stay in this app session and are not sent to external AI services.")

page = st.session_state.get("page", "Dashboard")
if page == "Dashboard":
    dashboard()
elif page == "Screening":
    screening()
elif page == "Model Performance":
    performance()
elif page == "Potential Impact":
    impact()
else:
    about()
