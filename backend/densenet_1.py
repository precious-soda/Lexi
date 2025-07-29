import os
import json
import uuid
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.models as models
import pandas as pd
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
from segmentation.main_1 import process_image, Binarization

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# Configuration classes
class MainCFG:
    IMAGE_SIZE = 52
    NUM_CLASSES = 587
    MODEL_ARCHITECTURE = 'densenet121'
    PRETRAINED = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    MODEL_CHECKPOINT_DIR = os.path.join('models', 'densenet', 'main', 'checkpoints', 'kannada_densenet121_attention')
    MODEL_CHECKPOINT_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'kannada_densenet121_attention_best.pth')
    LABEL_CSV_PATH = os.path.join('label-main.csv')

class OttaksharaCFG:
    IMAGE_SIZE = 52
    NUM_CLASSES = 34
    MODEL_ARCHITECTURE = 'densenet121'
    PRETRAINED = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    MODEL_CHECKPOINT_DIR = os.path.join('models', 'densenet', 'ottaksharas', 'checkpoints', 'kannada_densenet121_attention')
    MODEL_CHECKPOINT_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'kannada_densenet121_attention_best.pth')
    LABEL_CSV_PATH = os.path.join('label-ottaksharas.csv')

UPLOAD_FOLDER = './uploads'
STATIC_FOLDER = './static'
FONT_PATH = './model/NotoSansKannada-VariableFont_wdthwght.ttf'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
TARGET_SIZE = MainCFG.IMAGE_SIZE
CROP_PADDING = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Seed for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Attention modules for DenseNet
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        out = avg_out + max_out
        out = self.fc(out).view(b, c, 1, 1)
        return x * out

class GlobalAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // reduction, 1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, h * w)
        v = self.value_conv(x).view(b, -1, h * w)
        attention = self.softmax(torch.bmm(q, k))
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out

class DenseNetWithAttention(nn.Module):
    def __init__(self, densenet, num_classes):
        super().__init__()
        self.features = densenet.features
        self.ca1 = ChannelAttention(256)
        self.ca2 = ChannelAttention(512)
        self.ca3 = ChannelAttention(1024)
        self.ca4 = ChannelAttention(1024)
        self.global_attn = GlobalAttention(1024)
        num_ftrs = densenet.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        x = self.features.denseblock1(x)
        x = self.ca1(x)
        x = self.features.transition1(x)
        x = self.features.denseblock2(x)
        x = self.ca2(x)
        x = self.features.transition2(x)
        x = self.features.denseblock3(x)
        x = self.ca3(x)
        x = self.features.transition3(x)
        x = self.features.denseblock4(x)
        x = self.ca4(x)
        x = self.features.norm5(x)
        x = self.global_attn(x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model(cfg):
    if cfg.MODEL_ARCHITECTURE == 'densenet121':
        densenet = models.densenet121(weights='DenseNet121_Weights.DEFAULT' if cfg.PRETRAINED else None)
        original_conv0 = densenet.features.conv0
        densenet.features.conv0 = nn.Conv2d(1, original_conv0.out_channels,
                                          kernel_size=original_conv0.kernel_size,
                                          stride=original_conv0.stride,
                                          padding=original_conv0.padding,
                                          bias=original_conv0.bias)
        if cfg.PRETRAINED:
            densenet.features.conv0.weight.data = original_conv0.weight.data.mean(dim=1, keepdim=True)
        model = DenseNetWithAttention(densenet, cfg.NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL_ARCHITECTURE}")
    return model

def load_model_and_stats(checkpoint_path, cfg):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model = build_model(cfg)
    try:
        import torch.serialization
        from torch.optim.lr_scheduler import OneCycleLR
        torch.serialization.add_safe_globals([getattr, OneCycleLR])
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    if 'train_mean' not in checkpoint or 'train_std' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'train_mean' or 'train_std'.")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise RuntimeError(f"Failed to load model state dict: {e}")
    model.to(cfg.DEVICE)
    model.eval()
    train_mean = checkpoint['train_mean']
    train_std = checkpoint['train_std']
    print(f"Loaded model from {checkpoint_path}")
    print(f"Using normalization stats: Mean={train_mean:.4f}, Std={train_std:.4f}")
    return model, train_mean, train_std

def load_class_names(csv_path, num_classes):
    class_names = [str(i) for i in range(num_classes)]
    try:
        if os.path.exists(csv_path):
            df_labels = pd.read_csv(csv_path, header=None, names=['Folder_Index', 'Kannada_Character'])
            df_labels = df_labels.sort_values('Folder_Index')
            loaded_names = df_labels['Kannada_Character'].tolist()
            if len(loaded_names) == num_classes:
                class_names = loaded_names
                print(f"Loaded {len(class_names)} class names from {csv_path}")
            else:
                print(f"Warning: Number of names in {csv_path} ({len(loaded_names)}) does not match NUM_CLASSES ({num_classes}).")
        else:
            print(f"Warning: Label CSV not found at {csv_path}. Using numeric labels.")
    except Exception as e:
        print(f"Error loading class names from {csv_path}: {e}. Using numeric labels.")
    return class_names

def load_mappings(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON mappings: {e}")
        return None

def get_character_group(index, mappings):
    if not mappings:
        return None
    if index in mappings.get('vowels', []):
        return 'vowels'
    elif index in mappings.get('consonants', []):
        return 'consonants'
    elif index in mappings.get('consonant_vowel_combinations', []):
        return 'consonant_vowel_combinations'
    elif index in mappings.get('consonant_modifiers', []):
        return 'consonant_modifiers'
    elif index in mappings.get('conjuncts', []):
        return 'conjuncts'
    elif index in mappings.get('numerals', []):
        return 'numerals'
    elif index in mappings.get('special_characters', []):
        return 'special_characters'
    else:
        return None

def extract_base_consonant(main_index, main_chars, mappings):
    if not mappings:
        return None, None
    if main_index in mappings.get('consonants', []):
        return main_chars[main_index], ''
    if main_index in mappings.get('consonant_vowel_combinations', []) or main_index in mappings.get('consonant_modifiers', []):
        block_start = (main_index - 15) // 16 * 16 + 15
        base_consonant_index = block_start + 1
        base_consonant = main_chars[base_consonant_index] if base_consonant_index < len(main_chars) else ''
        if main_index in mappings.get('consonant_vowel_combinations', []):
            vowel_index = (main_index - block_start - 2) % 12
            vowel_signs = ['ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ']
            vowel_sign = vowel_signs[vowel_index] if vowel_index < len(vowel_signs) else ''
        elif main_index in mappings.get('consonant_modifiers', []):
            modifier_index = (main_index - block_start - 14) % 2
            modifier_signs = ['ಂ', 'ಃ']
            vowel_sign = modifier_signs[modifier_index] if modifier_index < len(modifier_signs) else ''
        else:
            vowel_sign = ''
        return base_consonant, vowel_sign
    if main_index in mappings.get('conjuncts', []):
        return main_chars[main_index], ''
    return None, None

def combine_characters(main_index, ottakshara_index, main_chars, ott_class_names, mappings):
    group = get_character_group(main_index, mappings)
    if group is None or main_index >= len(main_chars):
        return main_chars[main_index] if main_index < len(main_chars) else f"Unknown ({main_index})"
    if ottakshara_index >= len(ott_class_names):
        return main_chars[main_index] if main_index < len(main_chars) else f"Unknown ({main_index})"
    main_char = main_chars[main_index]
    ottakshara_char = ott_class_names[ottakshara_index]
    if group in ['vowels', 'numerals', 'special_characters']:
        return main_char
    base_consonant, vowel_sign = extract_base_consonant(main_index, main_chars, mappings)
    if base_consonant is None:
        return main_char
    virama = '್'
    conjunct = f"{base_consonant}{virama}{ottakshara_char}"
    if vowel_sign:
        conjunct += vowel_sign
    return conjunct

def prepare_segmented_image(roi, padding=10):
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masked_roi = np.where(mask == 255, roi, 255)
    padded_img = cv2.copyMakeBorder(masked_roi, padding, padding, padding, padding, 
                                    cv2.BORDER_CONSTANT, value=255)
    return padded_img

def preprocess_for_prediction(roi, is_segmented=False):
    try:
        if not is_segmented:
            gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 15
            )
            open_kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
            close_kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            processed_binary = closed
            contours, _ = cv2.findContours(processed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("Warning: No contours found.")
                return None
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < 50:
                print(f"Warning: Largest contour area is too small ({cv2.contourArea(largest_contour)}).")
                return None
            x, y, w, h = cv2.boundingRect(largest_contour)
            y1 = max(0, y - CROP_PADDING)
            y2 = min(processed_binary.shape[0], y + h + CROP_PADDING)
            x1 = max(0, x - CROP_PADDING)
            x2 = min(processed_binary.shape[1], x + w + CROP_PADDING)
            cropped_char = processed_binary[y1:y2, x1:x2]
            if cropped_char.shape[0] == 0 or cropped_char.shape[1] == 0:
                print("Warning: Cropped character has zero dimension.")
                return None
        else:
            cropped_char = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, cropped_char = cv2.threshold(cropped_char, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ch, cw = cropped_char.shape
        if ch <= TARGET_SIZE and cw <= TARGET_SIZE:
            sharp_char_canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            top = (TARGET_SIZE - ch) // 2
            left = (TARGET_SIZE - cw) // 2
            end_row = min(top + ch, TARGET_SIZE)
            end_col = min(left + cw, TARGET_SIZE)
            img_h = end_row - top
            img_w = end_col - left
            sharp_char_canvas[top:end_row, left:end_col] = cropped_char[:img_h, :img_w]
        else:
            scale = (TARGET_SIZE - CROP_PADDING) / max(ch, cw)
            new_w = min(TARGET_SIZE, max(1, int(cw * scale)))
            new_h = min(TARGET_SIZE, max(1, int(ch * scale)))
            sharp_char_canvas = cv2.resize(cropped_char, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            delta_w = TARGET_SIZE - new_w
            delta_h = TARGET_SIZE - new_h
            top = delta_h // 2
            left = delta_w // 2
            canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            end_row = min(top + new_h, TARGET_SIZE)
            end_col = min(left + new_w, TARGET_SIZE)
            img_h = end_row - top
            img_w = end_col - left
            canvas[top:end_row, left:end_col] = sharp_char_canvas[:img_h, :img_w]
            sharp_char_canvas = canvas
        return sharp_char_canvas
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        traceback.print_exc()
        return None

def predict_image(model, roi, train_mean, train_std, class_names, device, is_segmented=False):
    processed_np = preprocess_for_prediction(roi, is_segmented=is_segmented)
    if processed_np is None:
        return None
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[train_mean], std=[train_std])
    ])
    input_tensor = transform(processed_np)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    pred_index = predicted_idx.item()
    pred_confidence = float(confidence.item())
    pred_class_name = class_names[pred_index] if 0 <= pred_index < len(class_names) else f"Unknown Index ({pred_index})"
    return {
        "index": int(pred_index),
        "class_name": pred_class_name,
        "confidence": pred_confidence
    }

def save_predicted_text(segmentation_result, output_dir, session_id):
    try:
        output_file = os.path.join(output_dir, session_id, "predicted_text.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for li in sorted(segmentation_result["lines"].keys(), key=int):
                line_text = ""
                for wi in sorted(segmentation_result["lines"][str(li)]["words"].keys(), key=int):
                    word_text = ""
                    for char in segmentation_result["lines"][str(li)]["words"][str(wi)]["characters"]:
                        combined_char = char.get("combined_char", "Unknown")
                        word_text += combined_char
                    line_text += word_text + " "
                f.write(line_text.strip() + "\n")
        print(f"Saved predicted text to {output_file}")
        return f"/static/{session_id}/predicted_text.txt"
    except Exception as e:
        print(f"Error saving predicted text: {e}")
        traceback.print_exc()
        return None

def process_image_with_predictions(image_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std, main_class_names, ott_class_names, output_dir, session_id):
    mappings = load_mappings(os.path.join('kannada_mappings.json'))
    if mappings is None:
        print("Warning: Proceeding without character mappings. Ottaksharas will be ignored.")
    
    # Call the segmentation pipeline from segmentation.main_1
    segmentation_result = process_image(image_path, os.path.join(output_dir, session_id))
    if segmentation_result is None:
        raise ValueError(f"Segmentation failed for image at {image_path}")
    
    # Convert integer keys to strings for compatibility
    segmentation_result_str = {"lines": {}}
    for li in segmentation_result["lines"]:
        segmentation_result_str["lines"][str(li)] = {
            "line_img": segmentation_result["lines"][li]["line_img"],
            "bbox": [int(x) for x in segmentation_result["lines"][li]["bbox"]],
            "words": {}
        }
        for wi in segmentation_result["lines"][li]["words"]:
            segmentation_result_str["lines"][str(li)]["words"][str(wi)] = {
                "word_img": segmentation_result["lines"][li]["words"][wi]["word_img"],
                "bbox": [int(x) for x in segmentation_result["lines"][li]["words"][wi]["bbox"]],
                "characters": []
            }
            for char in segmentation_result["lines"][li]["words"][wi]["characters"]:
                char_data = {
                    "char_img": char["char_img"],
                    "bbox": [int(x) for x in char["bbox"]],
                    "type": char["type"],
                    "ottaksharas": []
                }
                for ott in char.get("ottaksharas", []):
                    char_data["ottaksharas"].append({
                        "char_img": ott["char_img"],
                        "bbox": [int(x) for x in ott["bbox"]],
                        "type": ott["type"]
                    })
                segmentation_result_str["lines"][str(li)]["words"][str(wi)]["characters"].append(char_data)
    
    # Load original image for drawing bounding boxes
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image at {image_path}")
    
    # Create directories for saving bounding box images
    lines_dir = os.path.join(output_dir, session_id, "lines")
    words_dir = os.path.join(output_dir, session_id, "words")
    chars_dir = os.path.join(output_dir, session_id, "characters")
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(words_dir, exist_ok=True)
    os.makedirs(chars_dir, exist_ok=True)
    
    # Process predictions and save bounding box images
    for li in segmentation_result_str["lines"]:
        # Save line bounding box image
        line_bbox = segmentation_result_str["lines"][li]["bbox"]
        line_bbox_img = image.copy()
        x, y, w, h = line_bbox
        cv2.rectangle(line_bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        line_bbox_path = os.path.join(lines_dir, f"line_{int(li):03d}_bbox.png")
        cv2.imwrite(line_bbox_path, line_bbox_img)
        segmentation_result_str["lines"][li]["bbox_url"] = f"/static/{session_id}/lines/line_{int(li):03d}_bbox.png"
        
        for wi in segmentation_result_str["lines"][li]["words"]:
            # Save word bounding box image
            word_bbox = segmentation_result_str["lines"][li]["words"][wi]["bbox"]
            word_bbox_img = cv2.cvtColor(segmentation_result_str["lines"][li]["line_img"], cv2.COLOR_GRAY2BGR).copy()
            wx, wy, ww, wh = word_bbox
            cv2.rectangle(word_bbox_img, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 2)
            word_bbox_path = os.path.join(words_dir, f"line_{int(li):03d}_word_{int(wi):03d}_bbox.png")
            cv2.imwrite(word_bbox_path, word_bbox_img)
            segmentation_result_str["lines"][li]["words"][wi]["bbox_url"] = f"/static/{session_id}/words/line_{int(li):03d}_word_{int(wi):03d}_bbox.png"
            
            for ci, char in enumerate(segmentation_result_str["lines"][li]["words"][wi]["characters"]):
                # Save character bounding box image
                char_bbox = char["bbox"]
                char_bbox_img = cv2.cvtColor(segmentation_result_str["lines"][li]["words"][wi]["word_img"], cv2.COLOR_GRAY2BGR).copy()
                cx, cy, cw, ch = char_bbox
                cv2.rectangle(char_bbox_img, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
                char_bbox_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_main_bbox.png")
                cv2.imwrite(char_bbox_path, char_bbox_img)
                char["bbox_url"] = f"/static/{session_id}/characters/line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_main_bbox.png"
                
                # Save enhanced character image
                enhanced_char = prepare_segmented_image(char["char_img"])
                char_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_main.png")
                cv2.imwrite(char_path, enhanced_char)
                
                # Predict main character
                char_pred = predict_image(main_model, enhanced_char, main_mean, main_std, main_class_names, MainCFG.DEVICE, is_segmented=True)
                char["prediction"] = char_pred
                
                # Update combined_char
                ottakshara_chars = []
                for oi, ott in enumerate(char["ottaksharas"]):
                    # Save ottakshara bounding box image
                    ott_bbox = ott["bbox"]
                    ott_bbox_img = cv2.cvtColor(segmentation_result_str["lines"][li]["words"][wi]["word_img"], cv2.COLOR_GRAY2BGR).copy()
                    ox, oy, ow, oh = ott_bbox
                    cv2.rectangle(ott_bbox_img, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
                    ott_bbox_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_ottakshara_{oi:02d}_bbox.png")
                    cv2.imwrite(ott_bbox_path, ott_bbox_img)
                    ott["bbox_url"] = f"/static/{session_id}/characters/line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_ottakshara_{oi:02d}_bbox.png"
                    
                    # Save enhanced ottakshara image
                    enhanced_ott = prepare_segmented_image(ott["char_img"])
                    ott_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_ottakshara_{oi:02d}.png")
                    cv2.imwrite(ott_path, enhanced_ott)
                    
                    # Predict ottakshara
                    ott_pred = predict_image(ott_model, enhanced_ott, ott_mean, ott_std, ott_class_names, OttaksharaCFG.DEVICE, is_segmented=True)
                    ott["prediction"] = ott_pred
                    if ott_pred:
                        ottakshara_chars.append(ott_pred)
                
                # Combine characters if necessary
                if char_pred and ottakshara_chars and mappings:
                    combined = combine_characters(
                        char_pred["index"],
                        ottakshara_chars[0]["index"],
                        main_class_names,
                        ott_class_names,
                        mappings
                    )
                    char["combined_char"] = combined
                elif char_pred:
                    char["combined_char"] = char_pred["class_name"]
    
    predicted_text_url = save_predicted_text(segmentation_result_str, output_dir, session_id)
    return segmentation_result_str, predicted_text_url

def compute_metrics(segmentation_result, session_id):
    total_lines = len(segmentation_result["lines"])
    total_words = 0
    total_chars = 0
    total_ottaksharas = 0
    areas = []
    predictions = []
    for li in segmentation_result["lines"]:
        for wi in segmentation_result["lines"][li]["words"]:
            total_words += 1
            for ci, char in enumerate(segmentation_result["lines"][li]["words"][wi]["characters"]):
                total_chars += 1
                x, y, w, h = char["bbox"]
                areas.append(w * h)
                pred = char["prediction"]
                combined_char = char.get("combined_char", "Unknown")
                if pred:
                    predictions.append({
                        "line": int(li),
                        "word": int(wi),
                        "char": int(ci),
                        "type": "main",
                        "label": pred["class_name"],
                        "confidence": float(pred["confidence"]),
                        "char_url": f"/static/{session_id}/characters/line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_main.png",
                        "combined_char": combined_char
                    })
                for oi, ott in enumerate(char["ottaksharas"]):
                    total_ottaksharas += 1
                    x, y, w, h = ott["bbox"]
                    areas.append(w * h)
                    ott_pred = ott["prediction"]
                    if ott_pred:
                        predictions.append({
                            "line": int(li),
                            "word": int(wi),
                            "char": int(ci),
                            "ottakshara": int(oi),
                            "type": "ottakshara",
                            "label": ott_pred["class_name"],
                            "confidence": float(ott_pred["confidence"]),
                            "char_url": f"/static/{session_id}/characters/line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_ottakshara_{oi:02d}.png",
                            "combined_char": combined_char
                        })
    avg_area = float(np.mean(areas)) if areas else 0.0
    avg_confidence = float(np.mean([p["confidence"] for p in predictions])) if predictions else 0.0
    return {
        "total_lines": int(total_lines),
        "total_words": int(total_words),
        "total_chars": int(total_chars),
        "total_ottaksharas": int(total_ottaksharas),
        "avg_char_area": avg_area,
        "total_predictions": int(len(predictions)),
        "avg_confidence": avg_confidence
    }, predictions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            session_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            
            # Process image with predictions
            segmentation_result, predicted_text_url = process_image_with_predictions(
                file_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std,
                main_class_names, ott_class_names, app.config['STATIC_FOLDER'], session_id
            )
            metrics, predictions = compute_metrics(segmentation_result, session_id)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            # Remove non-serializable image data from response
            for li in segmentation_result["lines"]:
                segmentation_result["lines"][li]["line_img"] = None
                for wi in segmentation_result["lines"][li]["words"]:
                    segmentation_result["lines"][li]["words"][wi]["word_img"] = None
                    for char in segmentation_result["lines"][li]["words"][wi]["characters"]:
                        char["char_img"] = None
                        for ott in char["ottaksharas"]:
                            ott["char_img"] = None
            
            return jsonify({
                "segmentation": segmentation_result,
                "predictions": predictions,
                "metrics": metrics,
                "session_id": session_id,
                "predicted_text_url": predicted_text_url
            })
        else:
            return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        print(f"Error processing request: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/static/<session_id>/<path:path>')
def serve_static(session_id, path):
    return send_from_directory(os.path.join(app.config['STATIC_FOLDER'], session_id), path)

if __name__ == "__main__":
    seed_everything(MainCFG.SEED)
    try:
        main_class_names = load_class_names(MainCFG.LABEL_CSV_PATH, MainCFG.NUM_CLASSES)
        ott_class_names = load_class_names(OttaksharaCFG.LABEL_CSV_PATH, OttaksharaCFG.NUM_CLASSES)
        main_model, main_mean, main_std = load_model_and_stats(MainCFG.MODEL_CHECKPOINT_FILE, MainCFG)
        ott_model, ott_mean, ott_std = load_model_and_stats(OttaksharaCFG.MODEL_CHECKPOINT_FILE, OttaksharaCFG)
    except Exception as e:
        print(f"Error loading resources: {e}")
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=5000)