# PolyOCR_Project
HEY EVERYONE! This is our polyocr project. I have created this repo so thatwe can work on integration of our individual parts

UI PART:
# PolyOCR – Multilingual Printed & Handwritten OCR Web App

**PolyOCR** is a web-based Optical Character Recognition (OCR) system capable of detecting and extracting printed and handwritten text from images in multiple languages. It uses EasyOCR for printed text and Microsoft's TrOCR for handwritten text, supporting languages like Hindi, Arabic, Spanish, French, Dutch, Portuguese, Swedish, German and English.

>  This repository contains the **Web UI and frontend/backend integration** component of the PolyOCR project.

---

##  Deployed Application

🔗 Live Demo (Hugging Face Space): [https://huggingface.co/spaces/HimaHarika2827/polyocr](https://huggingface.co/spaces/HimaHarika2827/polyocr)

---

##  Features

-  Upload images with **printed or handwritten** text.
-  Supports **multiple languages**.
-  Smart backend detection (if implemented) to warn user when the wrong OCR type is selected.
-  Output displays:
  - Clean extracted text
  - Detected language
  - Annotated image with bounding boxes
-  Light/Dark mode toggle
-  Copy extracted text to clipboard
-  Displays friendly error messages for unsupported files or missing text

---

##  Tech stacks used

### Frontend
- HTML5, CSS3
- JavaScript (vanilla)
- Responsive layout with clean styling

### Backend
- Python (Flask) 
- EasyOCR for Printed Text
- TrOCR for Handwritten Text (Transformer-based OCR)
- FastText for language detection
- OpenCV & PIL for image processing

---

##  Folder Structure
  polyocr/
  ├── static/
  │ ├── frontend_files/
  │ │ └── base.css
  │ │ └── base.js
  │ ├── uploads/
  | | |__uploaded_image.png  # Uploaded images
  │ └── annotated/ 
  | | |__ annotated_output.png # Images with bounding boxes
  | |__ json_output/
  |   |__ output.json #json output with text, confidence, language 
  ├── templates/
  │ └── base.html # Main frontend page
  ├── web.py # Flask backend
  ├── ocr_processing.py # Printed OCR logic (EasyOCR)
  ├── handwritten_ocr.py # Handwritten OCR logic (TrOCR)
  └── README.md # This file

### For faster results
##Run Faster Using Google Colab (GPU)
Both printed (EasyOCR) and handwritten (TrOCR) processing can be slow on CPU.
For faster results, run the OCR backend using Google Colab with GPU, and connect it to the web UI using ngrok or similar.


 

#Installation

### Clone the Repository
```bash
  git clone https://github.com/Bhavika-17/PolyOCR_Project.git
  cd PolyOCR_Project/polyocr_ui
  ```
  
  #Create a Virtual Environment (Optional but recommended)
  ``` bash
  python -m venv venv
  source venv/bin/activate   # On Windows: venv\Scripts\activate
  ```

  #Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```
  Note: In the requirements file there is link to download files required for TrOCR. Make sure      to keep them in a folder named ./trocr_model/ after downloading

  #Start the Flask Web App
  ```bash
  python web.py
   ```

  #Access in Browser
  Visit: http://127.0.0.1:5000/
  **You will see an upload page where you can select an image and choose either Printed OCR or         Handwritten OCR.
  
  #Example:
  * Choose the OCR type.

  * Upload an image containing text of your choosen ocr type.(note: This is must for better         results because models are different for both types.
  
  * Click "Detect Text".
  
   * The app displays:
        Detected text:
        Language:

### Colab Setup Instructions (for both OCRs):
  1. Create a new Colab notebook and enable GPU:
     Runtime → Change runtime type → GPU
  
  2. Upload:
  ocr_processing.py
  handwritten_ocr.py
  web.py  
  Any required models (like trocr_model/)
  Your HTML/CSS/JS files if needed
  
  3.Install dependencies:
  ```bash
  !pip install flask easyocr transformers fasttext opencv-python
   ```
  4.Start Flask app in Colab cell:
  ```python
  !python web.py
   ```
  5.Expose it via ngrok:
  ```python
  !pip install pyngrok
  from pyngrok import ngrok
  public_url = ngrok.connect(5000)
  print(public_url)
  ```
 6.Update your frontend to use the public_url for API calls if needed.

📝 Now your web UI will talk to a GPU-accelerated backend running in Colab.

** Notes
 Hugging Face Spaces may run on CPU and can be slow. Use Colab+GPU for actual usage/testing.
 
 Printed OCR (ocr_processing.py) and handwritten OCR (handwritten_ocr.py) both support GPU       processing in Colab.

#License
This project is licensed under the MIT License.


