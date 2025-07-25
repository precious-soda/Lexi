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
from scipy.ndimage import interpolation

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

UPLOAD_FOLDER = './Uploads'
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

def radon_transform_skew(image, angle_range=(-10, 10), angle_step=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    angles = np.arange(angle_range[0], angle_range[1], angle_step)
    sinogram = np.zeros((binary.shape[0], len(angles)))
    for i, angle in enumerate(angles):
        rotated = interpolation.rotate(binary, angle, reshape=False, mode='nearest')
        projection = np.sum(rotated, axis=1)
        sinogram[:, i] = projection
    variances = np.var(sinogram, axis=0)
    best_angle = angles[np.argmax(variances)]
    return best_angle

def deskew_image(image):
    skew_angle = radon_transform_skew(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def sort_components(stats, method="left-to-right"):
    reverse = False
    i = 0
    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1
    indices = np.argsort(stats[1:, i])[::-1 if reverse else 1] + 1
    return indices

def segment_sentence(image):
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(original_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_projection = np.sum(binary, axis=1)
    lines = []
    in_line = False
    start_y = 0
    threshold = np.max(horizontal_projection) * 0.1
    for i, projection in enumerate(horizontal_projection):
        if projection > threshold and not in_line:
            start_y = i
            in_line = True
        elif projection <= threshold and in_line:
            end_y = i
            if end_y - start_y > 10:
                lines.append((start_y, end_y))
            in_line = False
    if in_line:
        lines.append((start_y, len(horizontal_projection)))
    sentences = []
    for start_y, end_y in lines:
        line_region = binary[start_y:end_y, :]
        cols_with_content = np.sum(line_region, axis=0)
        content_cols = np.where(cols_with_content > 0)[0]
        if len(content_cols) > 0:
            left_x = content_cols[0]
            right_x = content_cols[-1]
            padding = 5
            left_x = max(0, left_x - padding)
            right_x = min(original_gray.shape[1] - 1, right_x + padding)
            start_y = max(0, start_y - padding)
            end_y = min(original_gray.shape[0] - 1, end_y + padding)
            roi = original_gray[start_y:end_y, left_x:right_x+1]
            bbox = (left_x, start_y, right_x - left_x + 1, end_y - start_y)
            sentences.append((roi, bbox))
    return sentences

def segment_word(sentence_roi):
    blurred_sentence = cv2.medianBlur(sentence_roi, 3)
    blurred_sentence = cv2.GaussianBlur(blurred_sentence, (5, 5), 0)
    edges = cv2.Canny(blurred_sentence, 50, 150)
    ret, thresh_inv = cv2.threshold(blurred_sentence, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(edges, thresh_inv)
    line_height = sentence_roi.shape[0]
    kernel_width = max(10, int(line_height * 0.5))
    kernel = np.ones((5, kernel_width), np.uint8)
    img_closing = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_closing, connectivity=8)
    sorted_indices = sort_components(stats, "left-to-right")
    areas = [stats[idx, cv2.CC_STAT_AREA] for idx in sorted_indices]
    area_threshold = np.median(areas) * 0.1 if areas else 1000
    words = []
    prev_x_end = -float('inf')
    gap_threshold = int(line_height * 0.3)
    for idx in sorted_indices:
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < area_threshold:
            continue
        gap = x - prev_x_end
        if gap > gap_threshold and prev_x_end != -float('inf'):
            pass
        roi = sentence_roi[y:y+h, x:x+w]
        words.append((roi, (x, y, w, h)))
        prev_x_end = x + w
    return words

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def segment_character(word_roi):
    row, col = word_roi.shape
    processing_roi = word_roi
    edges = cv2.Canny(processing_roi, 50, 150)
    ret, thresh_inv = cv2.threshold(processing_roi, 127, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(edges, thresh_inv)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    sorted_indices = sort_components(stats, "left-to-right")
    characters = []
    ottaksharas = []
    avg_char_height = np.mean([stats[idx, cv2.CC_STAT_HEIGHT] for idx in sorted_indices if stats[idx, cv2.CC_STAT_AREA] > 100])
    for idx in sorted_indices:
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 100:
            continue
        roi = word_roi[y:y+h, x:x+w]
        bbox = (x, y, w, h)
        if (y > row * 0.7) or (h < avg_char_height * 0.6 and area < np.median([stats[i, cv2.CC_STAT_AREA] for i in sorted_indices if stats[i, cv2.CC_STAT_AREA] > 100])):
            ottaksharas.append((roi, bbox))
        else:
            characters.append((roi, bbox, []))
    for ott_roi, ott_bbox in ottaksharas:
        ott_x, ott_y, ott_w, ott_h = ott_bbox
        ott_center = ott_x + ott_w / 2
        min_distance = float('inf')
        associated_char_idx = -1
        max_iou = 0
        for i, (_, char_bbox, _) in enumerate(characters):
            char_x, char_y, char_w, char_h = char_bbox
            char_center = char_x + char_w / 2
            is_below = ott_y > char_y + char_h * 0.5
            is_diagonal_right = ott_x >= char_x - ott_w * 0.5 and ott_x + ott_w <= char_x + char_w * 1.5
            iou = compute_iou(ott_bbox, char_bbox)
            distance = abs(ott_center - char_center) + abs(ott_y - (char_y + char_h)) * 0.5
            if (is_below or is_diagonal_right) and (iou > 0.1 or distance < min_distance):
                if iou > max_iou or (iou == max_iou and distance < min_distance):
                    max_iou = iou
                    min_distance = distance
                    associated_char_idx = i
        if associated_char_idx >= 0:
            characters[associated_char_idx][2].append((ott_roi, ott_bbox))
    return characters

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

def process_image(image_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std, main_class_names, ott_class_names, output_dir, session_id):
    mappings = load_mappings(os.path.join('kannada_mappings.json'))
    if mappings is None:
        print("Warning: Proceeding without character mappings. Ottaksharas will be ignored.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image at {image_path}")
    image = deskew_image(image)
    segmentation_result = {"lines": {}}
    
    lines_dir = os.path.join(output_dir, session_id, "lines")
    words_dir = os.path.join(output_dir, session_id, "words")
    chars_dir = os.path.join(output_dir, session_id, "characters")
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(words_dir, exist_ok=True)
    os.makedirs(chars_dir, exist_ok=True)
    
    sentences = segment_sentence(image)
    for li, (line_img, line_bbox) in enumerate(sentences):
        enhanced_line = prepare_segmented_image(line_img)
        line_path = os.path.join(lines_dir, f"line_{int(li):03d}.png")
        cv2.imwrite(line_path, enhanced_line)
        
        line_bbox_img = image.copy()
        x, y, w, h = line_bbox
        cv2.rectangle(line_bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        line_bbox_path = os.path.join(lines_dir, f"line_{int(li):03d}_bbox.png")
        cv2.imwrite(line_bbox_path, line_bbox_img)
        
        segmentation_result["lines"][str(li)] = {
            "line_img": enhanced_line,
            "bbox": [int(x) for x in line_bbox],
            "bbox_url": f"/static/{session_id}/lines/line_{int(li):03d}_bbox.png",
            "words": {}
        }
        
        words = segment_word(line_img)
        for wi, (word_img, word_bbox) in enumerate(words):
            enhanced_word = prepare_segmented_image(word_img)
            word_path = os.path.join(words_dir, f"line_{int(li):03d}_word_{int(wi):03d}.png")
            cv2.imwrite(word_path, enhanced_word)
            
            word_bbox_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR).copy()
            wx, wy, ww, wh = word_bbox
            cv2.rectangle(word_bbox_img, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 2)
            word_bbox_path = os.path.join(words_dir, f"line_{int(li):03d}_word_{int(wi):03d}_bbox.png")
            cv2.imwrite(word_bbox_path, word_bbox_img)
            
            segmentation_result["lines"][str(li)]["words"][str(wi)] = {
                "word_img": enhanced_word,
                "bbox": [int(x) for x in word_bbox],
                "bbox_url": f"/static/{session_id}/words/line_{int(li):03d}_word_{int(wi):03d}_bbox.png",
                "characters": []
            }
            
            characters = segment_character(word_img)
            for ci, (char_roi, char_bbox, ottaksharas) in enumerate(characters):
                enhanced_char = prepare_segmented_image(char_roi)
                char_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_main.png")
                cv2.imwrite(char_path, enhanced_char)
                
                char_bbox_img = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR).copy()
                cx, cy, cw, ch = char_bbox
                cv2.rectangle(char_bbox_img, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
                char_bbox_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_main_bbox.png")
                cv2.imwrite(char_bbox_path, char_bbox_img)
                
                char_pred = predict_image(main_model, enhanced_char, main_mean, main_std, main_class_names, MainCFG.DEVICE, is_segmented=True)
                char_data = {
                    "char_img": enhanced_char,
                    "bbox": [int(x) for x in char_bbox],
                    "bbox_url": f"/static/{session_id}/characters/line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_main_bbox.png",
                    "type": "main",
                    "prediction": char_pred,
                    "ottaksharas": [],
                    "combined_char": char_pred["class_name"] if char_pred else "Unknown"
                }
                
                ottakshara_chars = []
                for oi, (ott_roi, ott_bbox) in enumerate(ottaksharas):
                    enhanced_ott = prepare_segmented_image(ott_roi)
                    ott_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_ottakshara_{oi:02d}.png")
                    cv2.imwrite(ott_path, enhanced_ott)
                    
                    ott_bbox_img = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR).copy()
                    ox, oy, ow, oh = ott_bbox
                    cv2.rectangle(ott_bbox_img, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
                    ott_bbox_path = os.path.join(chars_dir, f"line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_ottakshara_{oi:02d}_bbox.png")
                    cv2.imwrite(ott_bbox_path, ott_bbox_img)
                    
                    ott_pred = predict_image(ott_model, enhanced_ott, ott_mean, ott_std, ott_class_names, OttaksharaCFG.DEVICE, is_segmented=True)
                    char_data["ottaksharas"].append({
                        "char_img": enhanced_ott,
                        "bbox": [int(x) for x in ott_bbox],
                        "bbox_url": f"/static/{session_id}/characters/line_{int(li):03d}_word_{int(wi):03d}_char_{ci:02d}_ottakshara_{oi:02d}_bbox.png",
                        "type": "ottakshara",
                        "prediction": ott_pred
                    })
                    if ott_pred:
                        ottakshara_chars.append(ott_pred)
                
                if char_pred and ottakshara_chars and mappings:
                    combined = combine_characters(
                        char_pred["index"],
                        ottakshara_chars[0]["index"],
                        main_class_names,
                        ott_class_names,
                        mappings
                    )
                    char_data["combined_char"] = combined
                elif char_pred:
                    char_data["combined_char"] = char_pred["class_name"]
                
                segmentation_result["lines"][str(li)]["words"][str(wi)]["characters"].append(char_data)
    
    predicted_text_url = save_predicted_text(segmentation_result, output_dir, session_id)
    return segmentation_result, predicted_text_url

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
            
            segmentation_result, predicted_text_url = process_image(file_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std, main_class_names, ott_class_names, app.config['STATIC_FOLDER'], session_id)
            metrics, predictions = compute_metrics(segmentation_result, session_id)
            
            os.remove(file_path)
            
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