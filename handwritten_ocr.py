import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
from huggingface_hub import snapshot_download
import langid
langid.set_languages(['hi', 'ar', 'en', 'fr', 'es', 'nl', 'de', 'it', 'pt', 'sv', 'ko'])
import json
import os

LANGUAGE_CODE_MAP = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'hi': 'Hindi',
    'zh': 'Chinese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'de': 'German',
    'it': 'Italian',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'sv': 'Swedish'   
}

# local_path = "./trocr_model"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


STATIC_FOLDER = "static"
JSON_OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, "json_output")
os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)

def detect_language(text):
    lang, prob = langid.classify(text)
    return lang, prob

def save_to_json(output_path, data):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ JSON saved to {output_path}")

def segment_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_images = []
    boxes = []

    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x, y, w, h = cv2.boundingRect(cnt)
        line_img = img[y:y+h, x:x+w]
        line_images.append(line_img)
        boxes.append((x, y, w, h))

    return line_images, boxes

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


def process_paragraph_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image.copy()
    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:
            line_crop = image[y:y+h, x:x+w]
            if line_crop.ndim == 2:
                line_crop = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2RGB)
            elif line_crop.shape[2] == 1:
                line_crop = cv2.merge([line_crop]*3)
            line_pil = Image.fromarray(line_crop)
            lines.append(((x, y, w, h), line_pil))

    lines_sorted = sorted(lines, key=lambda b: (b[0][1], b[0][0]))

    result_data = []
    for i, (box, line_image) in enumerate(lines_sorted):
        pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if not text.strip():
            continue
        lang, confidence = detect_language(text)
        if confidence < 0.8:
            lang = "unknown"
        result = {
            "line_number": i + 1,
            "bounding_box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
            "recognized_text": text,
            "language": lang,
            "confidence": confidence
        }
        result_data.append(result)

        cv2.rectangle(annotated, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv2.putText(annotated, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    annotated_path = 'static/annotated/annotated_output.png'
    cv2.imwrite(annotated_path, annotated)
    
    json_path = 'static/json_output/output.json'
    save_to_json(json_path, result_data)
    
    if result_data:
        top_result = max(result_data, key=lambda x: x["confidence"])
        top_language = top_result["language"]
    else:
        top_language = "unknown"

    top_language_full = LANGUAGE_CODE_MAP.get(top_language, "Unknown")

    return {
        "text": "\n".join([d["recognized_text"] for d in result_data]),
        "language": top_language_full,
        "json_path": json_path
        }