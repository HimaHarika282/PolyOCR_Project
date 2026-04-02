# 🚀 PolyOCR – Multilingual Printed & Handwritten OCR Web App

## 🚀 Overview

PolyOCR is a multilingual OCR web application that extracts both printed and handwritten text from images using a hybrid pipeline powered by EasyOCR and Microsoft TrOCR.

The system supports 8+ languages and provides structured, clean text output along with bounding box visualization — making it useful for digitizing documents, notes, and multilingual content.

This repository showcases the full-stack web interface and system integration layer of the project.

---

## 👩‍💻 My Contributions

* Designed and implemented the complete frontend using Flask, HTML, CSS, and JavaScript
* Integrated EasyOCR and TrOCR outputs into the web interface for seamless user interaction
* Built image upload, preview, and result visualization pipeline
* Implemented error handling for incorrect OCR mode selection and invalid inputs

---

## ⚙️ System Architecture

User Upload → Flask Backend → OCR Processing (EasyOCR / TrOCR)
→ Post-processing → Language Detection → UI Display

---

## ✨ Features

* Multilingual OCR (8+ languages including Hindi, Arabic, Spanish, French, German, etc.)
* Printed + Handwritten text recognition
* Structured paragraph-style text output
* Bounding box visualization on detected text
* Language detection using FastText
* Error handling for incorrect OCR selection
* Light/Dark mode toggle
* Copy extracted text to clipboard

---

## 🧠 OCR Engine

* **EasyOCR**: Used for printed text detection and recognition
* **TrOCR**: Transformer-based model used for handwritten text recognition

---

## 📸 Demo

### 🔹 Printed Text Detection

![Output 1](assets/output1.png)

### 🔹 Handwritten Text Recognition

![Output 2](assets/output2.png)

### 🔹 Multilingual Detection (Arabic)

![Output 3](assets/output3.png)

🔗 **Live Demo:** https://huggingface.co/spaces/HimaHarika2827/polyocr

These examples demonstrate the system’s ability to accurately extract and structure text across different languages and formats.

---

## 🛠 Tech Stack

### Frontend

* HTML5, CSS3
* JavaScript

### Backend

* Python (Flask)
* EasyOCR (Printed OCR)
* TrOCR (Handwritten OCR)
* FastText (Language Detection)
* OpenCV & PIL (Image Processing)

---

## 📁 Folder Structure

```
polyocr/
├── static/
│   ├── frontend_files/
│   ├── uploads/
│   ├── annotated/
│   └── json_output/
├── templates/
│   └── base.html
├── main.py
├── ocr_processing.py
├── handwritten_ocr.py
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/HimaHarika282/PolyOCR_Project.git
cd PolyOCR_Project/polyocr_ui

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

Visit: http://127.0.0.1:5000/

---

## ⚡ Running with GPU (Optional)

For faster OCR processing, run the backend on Google Colab with GPU and connect it to the frontend using ngrok.

---

## 📌 Notes

* Hugging Face deployment may run on CPU and can be slow
* GPU execution significantly improves performance for both EasyOCR and TrOCR

---

## 📄 License

This project is licensed under the MIT License.
