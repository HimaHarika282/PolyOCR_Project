import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import langid
langid.set_languages(['hi', 'ar', 'en', 'fr', 'es', 'nl', 'de', 'it', 'pt', 'sv'])


# Load image and sharpen it
image_path = 'image.png'
image = cv2.imread(image_path)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)

# Initialize multilingual readers
reader_ar = easyocr.Reader(['ar', 'en'])
reader_ko = easyocr.Reader(['ko','en'])
reader_hi = easyocr.Reader(['hi', 'en'])
reader_es = easyocr.Reader(['es', 'en', 'fr','de','it','pt','nl','sv'])

# Read texts
results_ar = reader_ar.readtext(sharpened)
results_ko = reader_ko.readtext(sharpened)
results_hi = reader_hi.readtext(sharpened)
results_es = reader_es.readtext(sharpened)

# Combine and sort all results by confidence (high to low)
all_results = results_ar + results_ko + results_hi + results_es 
all_results.sort(key=lambda x: x[2], reverse=True)

# Keep only best result per region (avoid duplicate detections)
final_results = []
seen_boxes = []

def is_overlapping(box1, box2, threshold=0.5):
    # Check IOU (intersection over union) between two boxes
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
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou > threshold

for bbox, text, conf in all_results:
    if not any(is_overlapping(bbox, seen) for seen in seen_boxes):
        seen_boxes.append(bbox)
        final_results.append((bbox, text, conf))

# Sort by top-left Y to simulate paragraph flow
final_results.sort(key=lambda x: min([pt[1] for pt in x[0]]))

# Combine into paragraph-like lines
print("  Recognized Text :\n")
line_text = ""
prev_bottom = 0
for bbox, text, conf in final_results:
    top = min([pt[1] for pt in bbox])
    if abs(top - prev_bottom) > 30:  # New line if gap is large
        if line_text:
            print(line_text)
            print(bbox)
            
        line_text = text
    else:
        line_text += " " + text
    prev_bottom = max([pt[1] for pt in bbox])

if line_text:
    print(line_text)
    print(bbox)

# Optional: Draw bounding boxes
for bbox, text, conf in final_results:
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    top_left = tuple(pts[0])
    cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 0, 0), 2)

plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


output_data = []

# Step 1: Collect all lines of text
full_text = ""

for (bbox, text, confidence) in final_results:
    full_text += text.strip() + " "  # accumulate all text lines

    # Line-wise processing
    if len(text.strip()) < 2:
        lang_code = "un"
        lang_score = 0.0
    else:
        try:
            ranked = langid.rank(text)
            top_lang, log_prob = ranked[0]
            lang_code = top_lang
             # Convert log-probability to normal confidence (0–1)
            lang_score = math.exp(log_prob)

            # Safety clamp: if result is extremely small, set to 0
            if lang_score < 0:
                lang_score = 0.0
            else:
                lang_score = round(lang_score, 8)    
        except Exception:
            lang_code = "un"
            lang_score = 0.0

    (top_left, top_right, bottom_right, bottom_left) = bbox
    x = int(top_left[0])
    y = int(top_left[1])
    w = int(top_right[0] - top_left[0])
    h = int(bottom_left[1] - top_left[1])

    output_data.append({
        "text": text,
        "bounding_box": [x, y, x + w, y + h],
        "language": lang_code,
        "lang_confidence": round(lang_score, 8),
        "ocr_confidence": round(confidence, 4)
    })

# Step 2: Detect overall language
try:
    ranked = langid.rank(full_text)
    overall_lang, log_prob = ranked[0]
    
    # Convert log probability to normal probability
    overall_score = math.exp(log_prob)

    # Optional: clamp very low values to 0.0
    if overall_score < 0:
        overall_score = 0.0
    else:
        overall_score = round(overall_score, 8)

except Exception:
    overall_lang = "un"
    overall_score = 0.0

# Step 3: Add overall language info at the end
result = {
    "overall_language": overall_lang,
    "overall_lang_confidence": round(overall_score, 8),
    "lines": output_data
}

# Step 4: Save to JSON
with open("ocr_results_with_overall_language.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(" JSON saved with overall language as ocr_results_with_overall_language.json")

#  ICDAR Format: res_output_icdar.txt

icdar_lines = []

for item in output_data:
    x1, y1, x2, y2 = item["bounding_box"]
    text = item["text"]
    # ICDAR typically expects comma-separated coordinates and the transcription
    line = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text}"
    icdar_lines.append(line)

# Save to file
icdar_filename = "res_output_icdar.txt"
with open(icdar_filename, "w", encoding="utf-8") as f:
    for line in icdar_lines:
        f.write(line + "\n")

print(f"ICDAR result file saved as {icdar_filename}")

