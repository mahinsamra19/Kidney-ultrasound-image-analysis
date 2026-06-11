# Renal AI Analysis System for Kidney Ultrasound Classification and Explainable Diagnosis

An AI-powered diagnostic support tool and explainable ultrasound classification system designed to classify kidney ultrasound scans and generate localized visual heatmaps. The system detects multiple pathologies (stones, hydronephrosis, cysts, tumours) and provides a streamlined patient-reporting workflow for clinical environments.

## Project Information

Project Title:
Renal AI Analysis System for Kidney Ultrasound Classification and Explainable Diagnosis

Academic Year:
2025–2026

Team Members:
- Manisha Kumari (24251A05U7)
- N. Srinivitha (24251A05Y8)
- Samra Mahin (25255A0529)

Guide:
K Sindhura

---

## 🌟 Key Features

### Advanced Diagnostic Engine
* **Multi-Class Classification**: Accurately classifies ultrasound images into: `stone`, `hydronephrosis`, `cyst`, `tumour`, or `normal`.
* **Image Enhancement Pipeline**: Uploaded scans are preprocessed using Non-Local Means (NLM) denoising to suppress speckle noise, followed by Contrast Limited Adaptive Histogram Equalisation (CLAHE) to improve local contrast before classification.
* **Custom Masking Logic (`detect_kidney_mask`)**: Intelligently identifies the actual ultrasound scan cone. This restricts all diagnostic analysis and visual overlays exclusively to anatomical regions, effectively eliminating background noise, text, and UI artifacts from the original scan.
* **Explainable Heatmaps (`build_pathology_heatmap`)**: Goes beyond basic Grad-CAM by utilizing advanced image processing techniques (blob detection, compactness, and variance analysis). This generates precise hotspots that are strictly localized to the physical finding.
* **Smart Normalization**: Automatically skips heatmap generation for healthy (normal) cases, presenting a clean interface.

### Professional Clinical Interface
* **Modern UI/UX**: Features a sleek, professional light theme designed for extended clinical use.
* **Comprehensive Biomarker Dashboard**: Replaces generic probability boxes with a detailed, full-width section showing dynamic confirmation percentages and specific clinical features for the detected pathology.
* **Patient-Reporting Workflow**: Built-in functionality to input patient details (Name, Age, Sex) and instantly generate/print a clean, professional clinical report free of distracting AI confidence metrics.

---

## 📊 Dataset

The model was trained using a kidney ultrasound image dataset containing five classes:

- Normal
- Stone
- Cyst
- Hydronephrosis
- Tumour

Images were resized to 224×224 pixels and processed through the image enhancement pipeline before training.

---

## 🤖 Model

**Base Model:** MobileNetV2

Transfer learning was used with a custom classification head for five-class kidney ultrasound classification.

---

## 📤 Results & System Outputs

The application generates:

- Predicted kidney condition
- Confidence score breakdown
- Grad-CAM heatmap
- Severity index
- Clinical severity level
- Biomarker analysis
- Automated clinical impression
- Printable patient report

---

## 🏗 Folder Structure

```
renal-ai/
├── backend/
│   ├── main.py                  ← FastAPI application entrypoint
│   ├── requirements.txt         ← Python dependencies
│   ├── models/
│   │   ├── inference.py         ← Core diagnostic logic, custom masking & heatmaps
│   │   └── weights/             ← Trained PyTorch checkpoints (.pth files)
│   └── routes/
│       └── analyze.py           ← REST API endpoints (/api/analyze)
└── frontend/
    └── templates/
        └── index.html           ← Single-page clinical dashboard interface
```

---

## 🚀 Installation & Execution

### 1. Install Dependencies

Ensure you have Python 3.10+ installed.

```bash
cd backend
pip install -r requirements.txt
```

### 2. Prepare Model Weights

Place your trained PyTorch checkpoints into the `backend/models/weights` directory. 

```bash
mkdir -p backend/models/weights
# Copy your trained checkpoints:
cp /path/to/mobilenetv2_renal.pth backend/models/weights/
```
*Note: If no weights are present, the application will run in **demo mode using a deterministic image fingerprinting pipeline, which produces consistent diagnostic outputs without requiring trained weights**.*

### 3. Run the Server

Start the FastAPI backend server:

```bash
cd backend
python main.py
# Alternatively, use uvicorn directly:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch the App

Open **http://localhost:8000** in your modern web browser to access the Renal AI Dashboard.

*Interactive API documentation is automatically generated and available at **http://localhost:8000/docs**.*

---

## ☁️ Deployment

The application is deployed on Render at [your-render-url].
It is linked to this GitHub repository and automatically rebuilds
on every push to the main branch. No GPU is required — all inference
runs on CPU.

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI, Uvicorn |
| **Machine Learning** | PyTorch (CPU inference; CUDA auto-detected if available locally) |
| **Computer Vision** | OpenCV (`cv2`), NumPy, SciPy (Blob Detection, Masking) |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript |

---

## 📸 Screenshots

### Upload Interface

[Image]

### Classification Result

[Image]

### Heatmap Visualization

[Image]

---

## 🧠 Technical Deep Dive: Custom Masking & Heatmaps

The precision of Renal AI comes from its post-processing pipeline:
1. **Inference**: The image passes through MobileNetV2, a lightweight five-class fine-tuned CNN to yield class probabilities and raw activation maps.
2. **Scan Cone Detection**: `detect_kidney_mask` uses thresholding and morphological operations to isolate the fan-shaped ultrasound area from the rest of the image.
3. **Refinement**: `build_pathology_heatmap` applies variance analysis and compactness filtering to the raw activations, ensuring that the generated heatmap perfectly aligns with physical abnormalities (like a calculus) rather than bleeding into the background.
4. **Overlay**: The resulting refined mask is blended back over the original image using a pathology-specific colormap (e.g., red for stones, blue for cysts) only within the detected scan cone.
5. **Severity Scoring**: The predicted class is mapped to a severity index (0–100) displayed on a five-point Clinical Severity Scale (Normal to Critical), alongside four condition-specific biomarkers with confirmation probabilities and an Automated Clinical Impression paragraph.

---

## 📝 Usage Notes

* **Data Privacy**: All uploaded images are processed in-memory. No patient scans are written to or stored on the disk.
* **Hardware Acceleration**: The system automatically detects and utilizes CUDA-enabled GPUs if available (`torch.cuda.is_available()`), falling back to CPU otherwise.
* **Printing Reports**: Use the "Generate Report" button in the UI to prompt for patient info and trigger the browser's print dialog, optimized specifically for A4 paper.

---

## 👩‍💻 Team

Developed by Manisha Kumari, N. Srinivitha, and Samra Mahin
Department of Computer Science and Engineering
G. Narayanamma Institute of Technology and Science, Hyderabad
Academic Year 2025–2026
