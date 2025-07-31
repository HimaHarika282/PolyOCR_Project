import cv2
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
from huggingface_hub import snapshot_download
import math
import langid
import json
import os


local_path = ".trocr_handwritten"


# Load pre-trained TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function: Segment lines using OpenCV
image_path = "english.png"
def segment_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation to merge characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours for each line
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_images = []
    boxes = []

    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x, y, w, h = cv2.boundingRect(cnt)
        line_img = img[y:y+h, x:x+w]
        line_images.append(line_img)
        boxes.append((x, y, w, h))
    
    return line_images, boxes

# Function: Recognize text using TrOCR
def recognize_line_images(line_images):
    results = []
    for line in line_images:
        pil_img = Image.fromarray(line)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append(text)
    return results  

def detect_language(text):
    lang, log_prob = langid.classify(text)
    try:
        confidence = math.exp(log_prob)  # Convert log probability to [0, 1]
    except OverflowError:
        confidence = 0.0
    return lang, round(confidence, 4)


def save_to_json(output_path, data):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"JSON saved to {output_path}")

#  Modified Main function
def process_paragraph_image(image_path):
    line_images, boxes = segment_lines(image_path)
    texts = recognize_line_images(line_images)

    results = []

    for text, (x, y, w, h) in zip(texts, boxes):

        if not text.strip():
           continue  # skip blank or whitespace-only lines
        lang, confidence = detect_language(text)
        result = {
            "bounding_box": {"x": x, "y": y, "w": w, "h": h},
            "recognized_text": text,
            "language": lang,
            "confidence": confidence
        }
        results.append(result)

        # 🖨 Optional print for clarity (without line numbers)
        print(f"Text: {text}")
        print(f"Language: {lang} (Confidence: {confidence:.2f})\n")

        

     #  Detect overall language from all recognized lines
    full_text = " ".join([item["recognized_text"] for item in results]).strip()
    if full_text:
       overall_lang, overall_conf = detect_language(full_text)
    else:
       overall_lang, overall_conf = "unknown", 0.0


    #  Final output structure including overall language
    output_data = {
        "overall_language": {
            "language": overall_lang,
            "confidence": overall_conf
        },
        "lines": results
    }

    # Save result to JSON
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_json_path = f"{base}_lang_output.json"
    save_to_json(output_json_path, output_data)

process_paragraph_image("english.png")
def convert_to_icdar_format(json_path, output_icdar_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_icdar_path, "w", encoding="utf-8") as out_file:
        for item in data["lines"]:
            x = item["bounding_box"]["x"]
            y = item["bounding_box"]["y"]
            w = item["bounding_box"]["w"]
            h = item["bounding_box"]["h"]
            text = item["recognized_text"]

            # ICDAR format: x1,y1,x2,y1,x2,y2,x1,y2,text
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            icdar_line = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text}"
            out_file.write(icdar_line + "\n")

    print(f" ICDAR file saved to {output_icdar_path}")

# Use it after your OCR pipeline
json_file = "english_lang_output.json"
icdar_file = "english_icdar_output.txt"
convert_to_icdar_format(json_file, icdar_file)
