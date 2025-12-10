import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.models as models
import pandas as pd
import random
from tqdm import tqdm
import traceback

# Configuration classes
class MainCFG:
    IMAGE_SIZE = 52
    NUM_CLASSES = 587
    MODEL_ARCHITECTURE = 'densenet121'
    PRETRAINED = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    MODEL_CHECKPOINT_DIR = os.path.join('main', 'checkpoints', 'kannada_densenet121_attention')
    MODEL_CHECKPOINT_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'kannada_densenet121_attention_best.pth')
    LABEL_CSV_PATH = os.path.join('handwritten-kannada-characters-main', 'label.csv')

class OttaksharaCFG:
    IMAGE_SIZE = 52
    NUM_CLASSES = 34
    MODEL_ARCHITECTURE = 'densenet121'
    PRETRAINED = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    MODEL_CHECKPOINT_DIR = os.path.join('output', 'checkpoints', 'kannada_densenet121_attention')
    MODEL_CHECKPOINT_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'kannada_densenet121_attention_best.pth')
    LABEL_CSV_PATH = os.path.join('handwritten-kannada-characters-ottaksharas', 'label.csv')

TARGET_SIZE = MainCFG.IMAGE_SIZE
CROP_PADDING = 10
VIRAMA = '\u0CCD'  # Kannada virama character (್)

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
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
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

# Updated Segmentation Functions
def sort_components(stats, method="left-to-right"):
    """Sort CCA components based on bounding box coordinates."""
    reverse = False
    i = 0
    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1
    indices = np.argsort(stats[1:, i])[::-1 if reverse else 1] + 1
    return indices

def segment_sentence(image):
    """Segment image into lines using horizontal projection method."""
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
    """Segment sentence into words using Canny + CCA."""
    blurred_sentence = cv2.GaussianBlur(sentence_roi, (5, 5), 0)
    edges = cv2.Canny(blurred_sentence, 50, 150)
    ret, thresh_inv = cv2.threshold(blurred_sentence, 127, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(edges, thresh_inv)
    kernel = np.ones((5, 40), np.uint8)
    img_dilation = cv2.dilate(combined, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_dilation, connectivity=8)
    sorted_indices = sort_components(stats, "left-to-right")
    words = []
    for idx in sorted_indices:
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 1000:
            continue
        roi = sentence_roi[y:y+h, x:x+w]
        words.append((roi, (x, y, w, h)))
    return words

def segment_character(word_roi):
    """Segment word into characters and ottaksharas using Canny + CCA with advanced ottakshara relation logic."""
    row, col = word_roi.shape
    processing_roi = word_roi
    edges = cv2.Canny(processing_roi, 50, 150)
    ret, thresh_inv = cv2.threshold(processing_roi, 127, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(edges, thresh_inv)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    sorted_indices = sort_components(stats, "left-to-right")
    characters = []
    ottaksharas = []
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
        if y > (row / 2):
            ottaksharas.append((roi, bbox))
        else:
            characters.append((roi, bbox, []))
    for ott_roi, ott_bbox in ottaksharas:
        ott_x, _, ott_w, _ = ott_bbox
        ott_center = ott_x + ott_w / 2
        min_distance = float('inf')
        associated_char_idx = -1
        for i, (_, char_bbox, _) in enumerate(characters):
            char_x, _, char_w, _ = char_bbox
            char_center = char_x + char_w / 2
            distance = abs(ott_center - char_center)
            if (ott_x < char_x + char_w and ott_x + ott_w > char_x) or distance < min_distance:
                min_distance = distance
                associated_char_idx = i
        if associated_char_idx >= 0:
            characters[associated_char_idx][2].append((ott_roi, ott_bbox))
    return characters

def prepare_segmented_image(roi, padding=10):
    """Enhance segmented image by applying Otsu's thresholding and adding padding."""
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
    pred_confidence = confidence.item()
    pred_class_name = class_names[pred_index] if 0 <= pred_index < len(class_names) else f"Unknown Index ({pred_index})"
    return {
        "index": pred_index,
        "class_name": pred_class_name,
        "confidence": pred_confidence
    }

def process_image(image_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std, main_class_names, ott_class_names, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image at {image_path}")
    segmentation_result = {"lines": {}}
    sentences = segment_sentence(image)
    print(f"Found {len(sentences)} lines")
    for li, (line_img, line_bbox) in enumerate(sentences):
        segmentation_result["lines"][li] = {"line_img": line_img, "bbox": line_bbox, "words": {}}
        words = segment_word(line_img)
        print(f"Line {li}: Found {len(words)} words")
        for wi, (word_img, word_bbox) in enumerate(words):
            segmentation_result["lines"][li]["words"][wi] = {"word_img": word_img, "bbox": word_bbox, "characters": []}
            characters = segment_character(word_img)
            for ci, (char_roi, char_bbox, ottaksharas) in enumerate(characters):
                enhanced_char = prepare_segmented_image(char_roi)
                char_pred = predict_image(main_model, enhanced_char, main_mean, main_std, main_class_names, MainCFG.DEVICE, is_segmented=True)
                char_data = {
                    "char_img": enhanced_char,
                    "bbox": char_bbox,
                    "type": "main",
                    "prediction": char_pred,
                    "ottaksharas": [],
                    "combined_char": char_pred["class_name"] if char_pred else "Unknown"
                }
                ottakshara_chars = []
                for oi, (ott_roi, ott_bbox) in enumerate(ottaksharas):
                    enhanced_ott = prepare_segmented_image(ott_roi)
                    ott_pred = predict_image(ott_model, enhanced_ott, ott_mean, ott_std, ott_class_names, OttaksharaCFG.DEVICE, is_segmented=True)
                    char_data["ottaksharas"].append({
                        "char_img": enhanced_ott,
                        "bbox": ott_bbox,
                        "type": "ottakshara",
                        "prediction": ott_pred
                    })
                    if ott_pred:
                        ottakshara_chars.append(ott_pred["class_name"])
                if ottakshara_chars and char_pred:
                    char_data["combined_char"] = char_pred["class_name"] + VIRAMA + (VIRAMA.join(ottakshara_chars))
                segmentation_result["lines"][li]["words"][wi]["characters"].append(char_data)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lines_dir = os.path.join(output_dir, "lines")
    words_dir = os.path.join(output_dir, "words")
    chars_dir = os.path.join(output_dir, "characters")
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(words_dir, exist_ok=True)
    os.makedirs(chars_dir, exist_ok=True)
    
    print(f"Saving segmented images to: {output_dir}")
    for li in segmentation_result["lines"]:
        line_img = prepare_segmented_image(segmentation_result["lines"][li]["line_img"])
        cv2.imwrite(os.path.join(lines_dir, f"line_{li:03d}.png"), line_img)
        for wi in segmentation_result["lines"][li]["words"]:
            word_img = prepare_segmented_image(segmentation_result["lines"][li]["words"][wi]["word_img"])
            cv2.imwrite(os.path.join(words_dir, f"line_{li:03d}_word_{wi:03d}.png"), word_img)
            for ci, char in enumerate(segmentation_result["lines"][li]["words"][wi]["characters"]):
                cv2.imwrite(os.path.join(chars_dir, f"line_{li:03d}_word_{wi:03d}_char_{ci:02d}_main.png"), char["char_img"])
                for oi, ott in enumerate(char["ottaksharas"]):
                    cv2.imwrite(os.path.join(chars_dir, f"line_{li:03d}_word_{wi:03d}_char_{ci:02d}_ottakshara_{oi:02d}.png"), ott["char_img"])
    print("Segmented images saved successfully!")
    return segmentation_result

def export_to_txt(segmentation_result, output_dir, filename="recognized_text.txt"):
    """Export recognized text to a .txt file."""
    if not segmentation_result:
        print("No segmentation result to export.")
        return
    txt_path = os.path.join(output_dir, filename)
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for li in sorted(segmentation_result["lines"].keys()):
                f.write(f"Line {li}:\n")
                for wi in sorted(segmentation_result["lines"][li]["words"].keys()):
                    word_text = ""
                    for char in segmentation_result["lines"][li]["words"][wi]["characters"]:
                        combined_char = char.get("combined_char", "Unknown")
                        word_text += combined_char
                    f.write(f"  Word {wi}: {word_text}\n")
                f.write("\n")
        print(f"Recognized text exported to: {txt_path}")
    except Exception as e:
        print(f"Error exporting to TXT file: {e}")
        traceback.print_exc()

def compute_metrics(segmentation_result):
    total_lines = len(segmentation_result["lines"])
    total_words = 0
    total_chars = 0
    total_ottaksharas = 0
    areas = []
    predictions_summary = []
    for li in segmentation_result["lines"]:
        for wi in segmentation_result["lines"][li]["words"]:
            total_words += 1
            for char in segmentation_result["lines"][li]["words"][wi]["characters"]:
                total_chars += 1
                x, y, w, h = char["bbox"]
                areas.append(w * h)
                pred = char["prediction"]
                combined_char = char.get("combined_char", "Unknown")
                if pred:
                    predictions_summary.append({
                        "type": "main",
                        "line": li,
                        "word": wi,
                        "char": total_chars - 1,
                        "class_name": pred["class_name"],
                        "confidence": pred["confidence"],
                        "model": "main",
                        "combined_char": combined_char
                    })
                for ott_idx, ott in enumerate(char["ottaksharas"]):
                    total_ottaksharas += 1
                    x, y, w, h = ott["bbox"]
                    areas.append(w * h)
                    ott_pred = ott["prediction"]
                    if ott_pred:
                        predictions_summary.append({
                            "type": "ottakshara",
                            "line": li,
                            "word": wi,
                            "char": total_chars - 1,
                            "ottakshara": ott_idx,
                            "class_name": ott_pred["class_name"],
                            "confidence": ott_pred["confidence"],
                            "model": "ottakshara",
                            "combined_char": combined_char
                        })
    avg_area = np.mean(areas) if areas else 0
    print("=" * 40)
    print("SEGMENTATION METRICS")
    print("=" * 40)
    print(f"Total Lines: {total_lines}")
    print(f"Total Words: {total_words}")
    print(f"Total Characters: {total_chars}")
    print(f"Total Ottaksharas: {total_ottaksharas}")
    print(f"Average Character Area: {avg_area:.2f}")
    print("=" * 40)
    print("\nPrediction Summary:")
    for pred in predictions_summary:
        if pred["type"] == "main":
            print(f"Line {pred['line']}, Word {pred['word']}, Char {pred['char']}, Model {pred['model']}: "
                  f"Predicted '{pred['class_name']}' (Confidence: {pred['confidence']:.4f}), Combined: '{pred['combined_char']}'")
        else:
            print(f"Line {pred['line']}, Word {pred['word']}, Char {pred['char']}, Ottakshara {pred['ottakshara']}, Model {pred['model']}: "
                  f"Predicted '{pred['class_name']}' (Confidence: {pred['confidence']:.4f}), Combined: '{pred['combined_char']}'")
    return predictions_summary

if __name__ == "__main__":
    seed_everything(MainCFG.SEED)
    image_path = input("Enter image file path: ")
    output_dir = input("Enter output directory: ")
    try:
        main_class_names = load_class_names(MainCFG.LABEL_CSV_PATH, MainCFG.NUM_CLASSES)
        ott_class_names = load_class_names(OttaksharaCFG.LABEL_CSV_PATH, OttaksharaCFG.NUM_CLASSES)
        main_model, main_mean, main_std = load_model_and_stats(MainCFG.MODEL_CHECKPOINT_FILE, MainCFG)
        ott_model, ott_mean, ott_std = load_model_and_stats(OttaksharaCFG.MODEL_CHECKPOINT_FILE, OttaksharaCFG)
        print("Processing image...")
        seg_result = process_image(image_path, main_model, ott_model, main_mean, main_std, ott_mean, ott_std, main_class_names, ott_class_names, output_dir)
        compute_metrics(seg_result)
        export_to_txt(seg_result, output_dir)
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()