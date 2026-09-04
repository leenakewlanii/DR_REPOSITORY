# DrishtiAI

AI-assisted diabetic retinopathy screening for accessible, explainable screening support.

DrishtiAI is an academic and hackathon prototype for people who may have limited access to ophthalmologists or difficulty interpreting medical terminology. It turns a retinal image into an experimental AI screening prediction, plain-language context, suggested follow-up guidance, and a Grad-CAM inspection view.

> **Important:** This is not a medical diagnostic device. It must not be used to make clinical decisions. AI predictions may be incorrect and should be confirmed by a qualified healthcare professional.

## Current capabilities

- Local inference with the supplied `final_DR_model.keras` model
- Three research categories: `Normal`, `NPDR`, and `PDR`
- Dynamic confidence and class probabilities from the model output
- Low-confidence warning using a configurable engineering threshold
- Technical image-quality warning for obvious dark, bright, low-contrast, or low-resolution inputs
- Grad-CAM original, heatmap, and overlay views
- Plain-language explanation and category-based informational next-step guidance
- English and reviewed Hindi translation surface, ready for additional languages
- Browser/device text-to-speech without an external voice API
- Low Bandwidth Mode that hides non-essential visual explanation output
- Self-contained printable HTML report, which can be printed to PDF from the browser
- Model performance dashboard with only the supplied metrics
- Disabled healthcare-directory and specialist-review controls clearly marked as future integrations

## Project structure

```text
diabetic_retinopathy_app/
├── app.py
├── requirements.txt
├── README.md
├── assets/                         # Optional supplied evaluation assets
├── models/
│   ├── final_DR_model.keras
│   └── class_mapping.json
└── utils/
    ├── __init__.py
    ├── prediction.py
    ├── gradcam.py
    ├── quality_check.py
    ├── report.py
    └── translations.py
```

The supplied `assets/` directory is currently empty. The app therefore does not fabricate or display a ROC curve image or confusion matrix.

## Verified model behavior

The saved model was inspected locally and has:

- Input shape: `224 x 224 x 3`
- Output shape: `3` softmax probabilities
- Outer layers: inference-safe augmentation layers, nested `resnet50`, global average pooling, dropout, and dense classification head
- Nested Grad-CAM layer: `conv5_block3_out`

The model graph contains ResNet50 caffe-style preprocessing. `utils/prediction.py` therefore feeds raw RGB pixels in the `0-255` range after resizing and does not apply `preprocess_input` a second time. The model is loaded for inference only; it is never retrained or modified.

## Reported evaluation results

These are fixed reported results for the supplied model, not metrics calculated from an uploaded image:

| Metric | Result |
|---|---:|
| Accuracy | 74.85% |
| Precision | 75.32% |
| Recall | 74.85% |
| F1 score | 74.48% |
| Quadratic Weighted Kappa | 74.36% |

| Class | ROC-AUC |
|---|---:|
| Normal | 0.9008 |
| NPDR | 0.8173 |
| PDR | 0.9472 |

Dataset image counts: 1,608 total, 1,141 training, 163 validation, and 326 test. These are image counts, not patient counts.

## Windows setup

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the project with the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended for TensorFlow compatibility.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

For a headless smoke test on another port:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8502
Invoke-WebRequest http://localhost:8502 -UseBasicParsing
```

## User flow

1. Open **Start New Screening**.
2. Upload a JPG, JPEG, or PNG retinal image. On supported phones, the browser file picker can offer the camera.
3. Review the original image and choose **Analyze Image**. Inference is not automatic.
4. Review the dynamic prediction, confidence, class probabilities, plain-language explanation, and suggested follow-up category.
5. Use **Listen to Result** for browser/device speech.
6. Review Grad-CAM, unless Low Bandwidth Mode is enabled.
7. Download the self-contained HTML report and print it to PDF if needed.

## Safety boundaries

The interface uses terms such as **AI screening result**, **model prediction**, and **suggested follow-up**. It does not claim confirmed diagnosis, clinical validation, doctor review, or guaranteed absence of disease.

The routine, follow-up, and prompt categories are informational guidance based on the model category, not medically validated triage. The low-confidence cutoff is an engineering setting and is not clinically validated. Technical image checks are not clinical image-quality assessment.

The app does not prescribe treatment or medication changes. It reminds users to seek urgent professional assessment for symptoms such as sudden vision loss, a sudden increase in floaters, flashes, a dark curtain or shadow, severe eye pain, or other concerning acute symptoms, regardless of the AI prediction.

## Privacy

Images are processed by the local Streamlit process and are not sent to external AI or voice APIs. Uploaded image data and results remain in the current Streamlit session and are not stored permanently by this prototype. A production deployment must define and document its storage, access, retention, and deletion policy.

## Deployment

For Streamlit Community Cloud, push the project to a repository, select `app.py` as the entry point, and use `requirements.txt`. Keep the model file available to the deployment; large model files may need a release asset or managed storage with a documented access policy.

### Render deployment

This project includes `render.yaml`, which defines a Render web service named `drishtiai`. To publish it:

1. Create a GitHub repository and push this project folder, including `models/final_DR_model.keras` and `models/class_mapping.json`.
2. In Render, choose **New + → Blueprint** and connect the GitHub repository.
3. Select the repository's `render.yaml` and apply the Blueprint.
4. Wait for the TensorFlow build to finish, then open the generated `onrender.com` URL.

The Blueprint uses Python 3.11, installs `requirements.txt`, and binds Streamlit to Render's `$PORT`. The free plan may sleep after inactivity and has limited CPU/RAM; model startup can take a few minutes.

For a container deployment, a minimal command is:

```text
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Use HTTPS and a documented privacy/retention policy before handling real patient data. This prototype is not a substitute for clinical governance, validation, or regulatory review.

## Troubleshooting

- **Model not found:** Confirm `models/final_DR_model.keras` exists and is readable.
- **Class mapping error:** Confirm `models/class_mapping.json` contains keys `0`, `1`, and `2`.
- **Analysis fails:** Confirm TensorFlow is installed, try a valid RGB/JPG/PNG image, and check that the file is under 15 MB.
- **Grad-CAM fails:** The implementation expects the verified nested model named `resnet50` and layer `conv5_block3_out`; do not rename model layers without updating `utils/gradcam.py`.
- **Speech does not play:** Check browser permissions and device volume. Speech uses the browser Web Speech API and does not upload text.
- **Hindi copy needs review:** Add reviewed strings to `utils/translations.py` before exposing additional translated medical language.
