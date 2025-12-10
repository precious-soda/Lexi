import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.models as models
import pandas as pd
import random
import traceback

class CFG:
    IMAGE_SIZE = 52
    NUM_CLASSES = 587
    MODEL_ARCHITECTURE = 'densenet121'
    PRETRAINED = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42  # For reproducibility
    
    MODEL_CHECKPOINT_DIR = os.path.join('output2', 'checkpoints', 'kannada_densenet121_attention')
    MODEL_CHECKPOINT_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'kannada_densenet121_attention_best.pth')
    LABEL_CSV_PATH = os.path.join('handwritten-kannada-characters_old', 'label.csv')

# Constants for preprocessing
TARGET_SIZE = CFG.IMAGE_SIZE
ADAPTIVE_THRESH_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
ADAPTIVE_THRESH_BLOCK_SIZE = 31
ADAPTIVE_THRESH_C = 15
MORPH_OPEN_KERNEL_SIZE = (3, 3)
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_KERNEL_SIZE = (3, 3)
MORPH_CLOSE_ITERATIONS = 1
CROP_PADDING = 10
BACKGROUND_NOISE_STDDEV = 45
BACKGROUND_CLIP_MAX_VALUE = 50
BACKGROUND_BLUR_KERNEL_SIZE = (9, 9)
BACKGROUND_BLUR_SIGMA = 4.0
MASK_THRESHOLD = 128
CANNY_LOW = 50
CANNY_HIGH = 150
MIN_CONTOUR_AREA = 100
DILATION_KERNEL = np.ones((5, 5), np.uint8)
DILATION_ITERATIONS = 1

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

def preprocess_for_prediction(input_path, min_contour_area=100):
    """
    Preprocess an image to extract multiple characters and ottaksharas using CCA and Canny for segmentation,
    but apply adaptive thresholding for final character images.
    Returns lists of preprocessed character images, their bounding boxes, and ottakshara associations.
    """
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Failed to load image {input_path}")
            return None, None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use Canny + CCA for segmentation
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred_gray, CANNY_LOW, CANNY_HIGH)
        _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(edges, thresh_inv)
        img_dilation = cv2.dilate(combined, DILATION_KERNEL, iterations=DILATION_ITERATIONS)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_dilation, connectivity=8)
        sorted_indices = sort_components(stats, "left-to-right")

        processed_chars = []
        bounding_boxes = []
        ottakshara_assignments = []
        characters = []
        ottaksharas = []

        # Segment characters and ottaksharas
        img_height = gray.shape[0]
        for idx in sorted_indices:
            x = stats[idx, cv2.CC_STAT_LEFT]
            y = stats[idx, cv2.CC_STAT_TOP]
            w = stats[idx, cv2.CC_STAT_WIDTH]
            h = stats[idx, cv2.CC_STAT_HEIGHT]
            area = stats[idx, cv2.CC_STAT_AREA]

            if area < min_contour_area:
                continue

            y1 = max(0, y - CROP_PADDING)
            y2 = min(gray.shape[0], y + h + CROP_PADDING)
            x1 = max(0, x - CROP_PADDING)
            x2 = min(gray.shape[1], x + w + CROP_PADDING)
            cropped_char = gray[y1:y2, x1:x2]
            if cropped_char.shape[0] == 0 or cropped_char.shape[1] == 0:
                print(f"Warning: Cropped component at ({x1},{y1}) has zero dimension. Skipping.")
                continue

            # Determine if it's an ottakshara
            if y > img_height / 2:
                ottaksharas.append((cropped_char, (x1, y1, x2, y2)))
            else:
                characters.append((cropped_char, (x1, y1, x2, y2), []))

        # Associate ottaksharas with main characters
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

        # Process main characters with previous preprocessing pipeline
        for char_roi, char_bbox, ottakshara_list in characters:
            # Apply adaptive thresholding and morphological operations
            binary = cv2.adaptiveThreshold(
                char_roi, 255, ADAPTIVE_THRESH_METHOD,
                cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
            )
            open_kernel = np.ones(MORPH_OPEN_KERNEL_SIZE, np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=MORPH_OPEN_ITERATIONS)
            close_kernel = np.ones(MORPH_CLOSE_KERNEL_SIZE, np.uint8)
            processed_binary = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=MORPH_CLOSE_ITERATIONS)

            # Include ottaksharas in the character image
            x1, y1, x2, y2 = char_bbox
            char_img = processed_binary
            for ott_roi, ott_bbox in ottakshara_list:
                ox1, oy1, ox2, oy2 = ott_bbox
                # Adjust ottakshara coordinates relative to char_roi
                rel_ox1 = max(0, ox1 - x1)
                rel_oy1 = max(0, oy1 - y1)
                rel_ox2 = min(ox2 - x1, char_img.shape[1])
                rel_oy2 = min(oy2 - y1, char_img.shape[0])
                if rel_ox2 > rel_ox1 and rel_oy2 > rel_oy1:
                    ott_binary = cv2.adaptiveThreshold(
                        ott_roi, 255, ADAPTIVE_THRESH_METHOD,
                        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
                    )
                    char_img[rel_oy1:rel_oy2, rel_ox1:rel_ox2] = np.maximum(
                        char_img[rel_oy1:rel_oy2, rel_ox1:rel_ox2], ott_binary[:rel_oy2-rel_oy1, :rel_ox2-rel_ox1]
                    )

            ch, cw = char_img.shape
            scale = (TARGET_SIZE - CROP_PADDING) / max(ch, cw)
            new_w = min(TARGET_SIZE, max(1, int(cw * scale)))
            new_h = min(TARGET_SIZE, max(1, int(ch * scale)))
            resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            sharp_char_canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            delta_w = TARGET_SIZE - new_w
            delta_h = TARGET_SIZE - new_h
            top, left = delta_h // 2, delta_w // 2
            end_row = min(top + new_h, TARGET_SIZE)
            end_col = min(left + new_w, TARGET_SIZE)
            img_h = end_row - top
            img_w = end_col - left
            sharp_char_canvas[top:end_row, left:end_col] = resized[:img_h, :img_w]

            background_canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
            noise = np.random.normal(0, BACKGROUND_NOISE_STDDEV, (TARGET_SIZE, TARGET_SIZE))
            noisy_background = background_canvas + noise
            clipped_dark_noise = np.clip(noisy_background, 0, BACKGROUND_CLIP_MAX_VALUE)
            diffused_dark_background = cv2.GaussianBlur(
                clipped_dark_noise, BACKGROUND_BLUR_KERNEL_SIZE, BACKGROUND_BLUR_SIGMA
            )
            diffused_dark_background_uint8 = diffused_dark_background.astype(np.uint8)

            _, char_mask = cv2.threshold(sharp_char_canvas, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
            final_image_np = np.where(char_mask > 0, sharp_char_canvas, diffused_dark_background_uint8)

            processed_chars.append(final_image_np)
            bounding_boxes.append(char_bbox)
            ottakshara_assignments.append([(ott_roi, ott_bbox) for ott_roi, ott_bbox in ottakshara_list])

        if not processed_chars:
            print(f"Warning: No valid characters found in {input_path} after filtering.")
            return None, None, None

        # Save intermediate images for debugging
        debug_dir = "debug_preprocessed"
        os.makedirs(debug_dir, exist_ok=True)
        for i, char_img in enumerate(processed_chars):
            cv2.imwrite(os.path.join(debug_dir, f"char_{i:03d}.png"), char_img)

        return processed_chars, bounding_boxes, ottakshara_assignments

    except Exception as e:
        print(f"Error during preprocessing image {input_path}: {e}")
        traceback.print_exc()
        return None, None, None

def load_class_names(csv_path):
    class_names = [str(i) for i in range(CFG.NUM_CLASSES)]
    try:
        if os.path.exists(csv_path):
            df_labels = pd.read_csv(csv_path, header=None, names=['Folder_Index', 'Kannada_Character'])
            df_labels = df_labels.sort_values('Folder_Index')
            loaded_names = df_labels['Kannada_Character'].tolist()
            if len(loaded_names) == CFG.NUM_CLASSES:
                class_names = loaded_names
                print(f"Loaded {len(class_names)} class names from {csv_path}")
            else:
                print(f"Warning: Number of names in {csv_path} ({len(loaded_names)}) does not match NUM_CLASSES ({CFG.NUM_CLASSES}). Using numeric labels.")
        else:
            print(f"Warning: Label CSV not found at {csv_path}. Using numeric labels.")
    except Exception as e:
        print(f"Error loading class names from {csv_path}: {e}. Using numeric labels.")
    return class_names

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
        raise RuntimeError(f"Failed to load model state dict: {e}. Ensure checkpoint matches DenseNetWithAttention architecture.")

    model.to(cfg.DEVICE)
    model.eval()

    train_mean = checkpoint['train_mean']
    train_std = checkpoint['train_std']

    print(f"Loaded model from {checkpoint_path}")
    print(f"Using normalization stats: Mean={train_mean:.4f}, Std={train_std:.4f}")

    return model, train_mean, train_std

def predict_multiple_characters(model, image_path, train_mean, train_std, class_names, device, min_contour_area=100):
    """
    Predict classes for multiple characters in a single image, including ottaksharas.
    Returns a list of predictions with bounding box and ottakshara information.
    """
    processed_chars, bounding_boxes, ottakshara_assignments = preprocess_for_prediction(image_path, min_contour_area)
    if processed_chars is None or bounding_boxes is None:
        print("Preprocessing failed or no valid characters found.")
        return None

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[train_mean], std=[train_std])
    ])
    input_tensors = [transform(char) for char in processed_chars]
    input_batch = torch.stack(input_tensors).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted_indices = torch.max(probabilities, 1)

    results = []
    for i, (pred_idx, confidence, bbox, ottaksharas) in enumerate(zip(predicted_indices, confidences, bounding_boxes, ottakshara_assignments)):
        pred_index = pred_idx.item()
        pred_confidence = confidence.item()
        pred_class_name = class_names[pred_index] if 0 <= pred_index < len(class_names) else f"Unknown Index ({pred_index})"
        ottakshara_info = [(ott_bbox, None) for _, ott_bbox in ottaksharas]
        results.append({
            "index": pred_index,
            "class_name": pred_class_name,
            "confidence": pred_confidence,
            "bounding_box": bbox,
            "ottaksharas": ottakshara_info
        })

    return results

if __name__ == "__main__":
    seed_everything(CFG.SEED)
    
    IMAGE_TO_PREDICT = "../../test/k64.jpeg"  # Replace with your image path
    
    try:
        class_names = load_class_names(CFG.LABEL_CSV_PATH)
        model, train_mean, train_std = load_model_and_stats(CFG.MODEL_CHECKPOINT_FILE, CFG)
    except Exception as e:
        print(f"Error loading resources: {e}")
        exit(1)

    if not os.path.exists(IMAGE_TO_PREDICT):
        print(f"Error: Image to predict not found at '{IMAGE_TO_PREDICT}'")
    else:
        print(f"\nPredicting characters in image: {IMAGE_TO_PREDICT}")
        prediction_results = predict_multiple_characters(
            model, IMAGE_TO_PREDICT, train_mean, train_std, class_names, CFG.DEVICE
        )

        if prediction_results:
            print("\n--- Prediction Results ---")
            for i, result in enumerate(prediction_results, 1):
                print(f"\nCharacter {i}:")
                print(f"  Predicted Index:      {result['index']}")
                print(f"  Predicted Class Name: '{result['class_name']}'")
                print(f"  Confidence:           {result['confidence']:.4f}")
                print(f"  Bounding Box:         (x1={result['bounding_box'][0]}, y1={result['bounding_box'][1]}, "
                      f"x2={result['bounding_box'][2]}, y2={result['bounding_box'][3]})")
                if result["ottaksharas"]:
                    print("  Associated Ottaksharas:")
                    for j, (ott_bbox, _) in enumerate(result["ottaksharas"], 1):
                        print(f"    Ottakshara {j}: Bounding Box (x1={ott_bbox[0]}, y1={ott_bbox[1]}, "
                              f"x2={ott_bbox[2]}, y2={ott_bbox[3]})")
            print("-------------------------")
            
            # Visualize results
            img = cv2.imread(IMAGE_TO_PREDICT)
            for result in prediction_results:
                x1, y1, x2, y2 = result['bounding_box']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, result['class_name'], (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for ott_bbox, _ in result['ottaksharas']:
                    ox1, oy1, ox2, oy2 = ott_bbox
                    cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (255, 0, 0), 1)
            cv2.imwrite("output_with_boxes.jpg", img)
            print("Saved visualization to 'output_with_boxes.jpg'")
        else:
            print("Prediction failed or no characters detected.")