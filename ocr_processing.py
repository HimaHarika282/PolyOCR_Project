import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import langid
import torch

langid.set_languages(['hi', 'ar', 'en', 'fr', 'es', 'nl', 'de', 'it', 'pt', 'sv', 'ko'])

LANGUAGE_CODE_MAP = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'hi': 'Hindi',
    'ch_sim': 'Chinese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'de': 'German',
    'it': 'Italian',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'sv': 'Swedish'  
    # Add more as needed
}
use_gpu = torch.cuda.is_available()
print("🔁 Loading global EasyOCR readers...")
reader_hi = easyocr.Reader(['hi', 'en'], gpu=use_gpu)  # Set gpu=True if available
reader_ar = easyocr.Reader(['ar', 'en'], gpu=use_gpu)
reader_ko = easyocr.Reader(['ko', 'en'], gpu=use_gpu)
reader_es = easyocr.Reader(['es', 'en', 'fr', 'de', 'it', 'pt', 'nl', 'sv'], gpu=use_gpu)



def is_overlapping(box1, box2, threshold=0.5):
    box1 = np.array(box1, dtype=np.int32)
    box2 = np.array(box2, dtype=np.int32)

    if box1.size == 0 or box2.size == 0:
        return False

    rect1 = cv2.boundingRect(box1)
    rect2 = cv2.boundingRect(box2)

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou > threshold

def get_weighted_language(boxes):
    lang_scores = {}
    for box in boxes:
        lang = box["language"]
        weight = box["ocr_confidence"]
        lang_scores[lang] = lang_scores.get(lang, 0) + weight
    if lang_scores:
        return max(lang_scores, key=lang_scores.get)
    return "un"

def process_printed_image_with_easyocr(image_path):
    image = cv2.imread(image_path)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)

    results = (
        reader_hi.readtext(sharpened) +
        reader_ar.readtext(sharpened) +
        reader_ko.readtext(sharpened) +
        reader_es.readtext(sharpened)
    )  
    
    results.sort(key=lambda x: x[2], reverse=True)

    final_results = []
    seen_boxes = []

    for bbox, text, conf in results:
        if not any(is_overlapping(bbox, seen) for seen in seen_boxes):
            seen_boxes.append(bbox)
            final_results.append((bbox, text, conf))

    
    final_results.sort(key=lambda x: min([pt[1] for pt in x[0]]))

    
    paragraph_lines = []
    line_text = ""
    prev_bottom = 0

    for bbox, text, conf in final_results:
        top = min([pt[1] for pt in bbox])
        if abs(top - prev_bottom) > 30:
            if line_text:
                paragraph_lines.append(line_text)
            line_text = text
        else:
            line_text += " " + text
        prev_bottom = max([pt[1] for pt in bbox])
    if line_text:
        paragraph_lines.append(line_text)

   
    for bbox, text, _ in final_results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        top_left = tuple(pts[0])
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    

    annotated_path = 'static/annotated/annotated_output.png'
    json_path = 'static/json_output/output.json'
    cv2.imwrite(annotated_path, image)

    
    output_data = []
    for bbox, text, conf in final_results:
        text_clean = text.strip()
        if len(text_clean) < 2:
            lang_code = "un"
            lang_conf = 0.0
        else:
            lang_code, _ = langid.classify(text_clean)
            lang_conf = langid.rank(text_clean)[0][1]

        (top_left, top_right, bottom_right, bottom_left) = bbox
        x = int(top_left[0])
        y = int(top_left[1])
        w = int(top_right[0] - top_left[0])
        h = int(bottom_left[1] - top_left[1])

        output_data.append({
            "text": text_clean,
            "bounding_box": [x, y, x + w, y + h],
            "language": lang_code,
            "lang_confidence": round(lang_conf, 4),
            "ocr_confidence": round(conf, 4)
        })
    final_lang_code = get_weighted_language(output_data)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "text": "\n".join(paragraph_lines),
            "language": LANGUAGE_CODE_MAP.get(final_lang_code, "Unknown"),
            "boxes": output_data
        }, f, ensure_ascii=False, indent=2)   

    return {
        "text": "\n".join(paragraph_lines),
        "language": LANGUAGE_CODE_MAP.get(final_lang_code, "Unknown") ,
        "json_path": json_path,
        "annotated_image_path": annotated_path
        }