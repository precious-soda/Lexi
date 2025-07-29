import os
import cv2
import numpy as np
from scipy.ndimage import interpolation
from pathlib import Path

class Binarization:
    def __init__(self):
        self.hr = 0
        self.cei = None
        self.cei_bin = None
        self.eg_bin = None
        self.tli_erosion = None
        self.ldi = None
    
    def binarize(self, image):
        """Binarization using Sauvola method with light distribution correction."""
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale = self.light_distribution(grayscale)
        winy = int((2.0 * grayscale.shape[0] - 1) / 3)
        winx = min(grayscale.shape[1] - 1, winy)
        if winx > 127:
            winx = winy = 127
        return self.sauvola_threshold(grayscale, winx, winy)
    
    def sauvola_threshold(self, im, winx, winy):
        k = 0.1
        dR = 128
        wxh = winx // 2
        wyh = winy // 2
        x_first_th = wxh
        x_last_th = im.shape[1] - wxh - 1
        y_last_th = im.shape[0] - wyh - 1
        y_first_th = wyh
        map_m = np.zeros(im.shape, dtype=np.float32)
        map_s = np.zeros(im.shape, dtype=np.float32)
        self.calc_local_stats(im, map_m, map_s, winx, winy)
        th_surf = np.zeros(im.shape, dtype=np.float32)
        for j in range(y_first_th, y_last_th + 1):
            for i in range(im.shape[1] - winx + 1):
                m = map_m[j, i + wxh]
                s = map_s[j, i + wxh]
                th = m * (1 + k * (s / dR - 1))
                th_surf[j, i + wxh] = th
                if i == 0:
                    th_surf[j, :x_first_th + 1] = th
                    if j == y_first_th:
                        th_surf[:y_first_th, :x_first_th + 1] = th
                    if j == y_last_th:
                        th_surf[y_last_th + 1:, :x_first_th + 1] = th
                if j == y_first_th:
                    th_surf[:y_first_th, i + wxh] = th
                if j == y_last_th:
                    th_surf[y_last_th + 1:, i + wxh] = th
            th_surf[j, x_last_th:] = th
            if j == y_first_th:
                th_surf[:y_first_th, x_last_th:] = th
            if j == y_last_th:
                th_surf[y_last_th + 1:, x_last_th:] = th
        return np.where(im >= th_surf, 255, 0).astype(np.uint8)
    
    def calc_local_stats(self, im, map_m, map_s, winx, winy):
        im_sum = cv2.integral(im.astype(np.float64))
        im_sum_sq = cv2.integral((im.astype(np.float64)) ** 2)
        wxh = winx // 2
        wyh = winy // 2
        x_first_th = wxh
        y_first_th = wyh
        y_last_th = im.shape[0] - wyh - 1
        win_area = winx * winy
        for j in range(y_first_th, y_last_th + 1):
            for i in range(im.shape[1] - winx + 1):
                y1, y2 = j - wyh, j - wyh + winy
                x1, x2 = i, i + winx
                sum_val = (im_sum[y2, x2] + im_sum[y1, x1] - 
                           im_sum[y1, x2] - im_sum[y2, x1])
                sum_sq = (im_sum_sq[y2, x2] + im_sum_sq[y1, x1] - 
                          im_sum_sq[y1, x2] - im_sum_sq[y2, x1])
                m = sum_val / win_area
                s = np.sqrt((sum_sq - m * sum_val) / win_area)
                map_m[j, i + x_first_th] = m
                map_s[j, i + x_first_th] = s
    
    def light_distribution(self, grayscale):
        grayscale = grayscale.astype(np.float32)
        self.get_histogram(grayscale)
        self.get_cei(grayscale)
        self.get_edge(grayscale)
        self.get_tli(grayscale)
        int_img = self.cei.copy()
        for y in range(int_img.shape[1]):
            for x in range(int_img.shape[0]):
                if self.tli_erosion[x, y] == 0:
                    head = x
                    end = x
                    while (end < self.tli_erosion.shape[0] and 
                           self.tli_erosion[end, y] == 0):
                        end += 1
                    end -= 1
                    n = end - head + 1
                    if n <= 30:
                        mpv_h = []
                        mpv_e = []
                        for k in range(5):
                            if head - k >= 0:
                                mpv_h.append(self.cei[head - k, y])
                            if end + k < self.cei.shape[0]:
                                mpv_e.append(self.cei[end + k, y])
                        if mpv_h and mpv_e:
                            max_h = np.max(mpv_h)
                            max_e = np.max(mpv_e)
                            for m in range(n):
                                int_img[head + m, y] = max_h + (m + 1) * ((max_e - max_h) / n)
        kernel = np.ones((11, 11), np.float32) / 121
        self.ldi = cv2.filter2D(self.scale(int_img), -1, kernel)
        grayscale = (self.cei / self.ldi) * 260
        for y in range(self.tli_erosion.shape[0]):
            for x in range(self.tli_erosion.shape[1]):
                if self.tli_erosion[y, x] != 0:
                    grayscale[y, x] *= 1.5
        grayscale = cv2.GaussianBlur(grayscale, (3, 3), 2)
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
        return grayscale
    
    def get_histogram(self, image):
        hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [30], [0, 300])
        sqrt_hw = np.sqrt(image.shape[0] * image.shape[1])
        self.hr = 0
        for i in range(hist.shape[0]):
            if hist[i, 0] > sqrt_hw:
                self.hr = i * 10
                break
    
    def get_cei(self, grayscale):
        cei = (grayscale - (self.hr + 50 * 0.4)) * 2
        self.cei = cv2.normalize(cei, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
        _, self.cei_bin = cv2.threshold(self.cei, 59, 255, cv2.THRESH_BINARY_INV)
    
    def get_edge(self, grayscale):
        m1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        m2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32)
        m3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        m4 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
        eg1 = np.abs(cv2.filter2D(grayscale, cv2.CV_32F, m1))
        eg2 = np.abs(cv2.filter2D(grayscale, cv2.CV_32F, m2))
        eg3 = np.abs(cv2.filter2D(grayscale, cv2.CV_32F, m3))
        eg4 = np.abs(cv2.filter2D(grayscale, cv2.CV_32F, m4))
        eg_avg = self.scale((eg1 + eg2 + eg3 + eg4) / 4)
        _, self.eg_bin = cv2.threshold(eg_avg, 30, 255, cv2.THRESH_BINARY)
    
    def get_tli(self, grayscale):
        tli = np.ones((grayscale.shape[0], grayscale.shape[1]), dtype=np.float32) * 255
        tli -= self.eg_bin
        tli -= self.cei_bin
        _, tli = cv2.threshold(tli, 0, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.float32)
        self.tli_erosion = cv2.erode(tli, kernel)
        _, self.tli_erosion = cv2.threshold(self.tli_erosion, 0, 255, cv2.THRESH_BINARY)
    
    def scale(self, image):
        min_val, max_val = np.min(image), np.max(image)
        if max_val - min_val == 0:
            return np.zeros_like(image)
        res = image / (max_val - min_val)
        min_val = np.min(res)
        res = (res - min_val) * 255
        return res

def radon_transform_skew(image, angle_range=(-10, 10), angle_step=0.5):
    """Detect skew angle using Radon Transform."""
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
    """Deskew image using Radon Transform."""
    skew_angle = radon_transform_skew(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

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

def segment_word(sentence_roi):
    """Segment sentence into words using Canny + CCA with enhanced techniques."""
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
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
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
    """Segment word into characters and ottaksharas with improved spatial relationship analysis."""
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

def compute_metrics(segmentation_result):
    """Compute and print segmentation metrics."""
    if not segmentation_result:
        print("No segmentation result to compute metrics for.")
        return
        
    total_lines = len(segmentation_result["lines"])
    total_words = 0
    total_chars = 0
    total_ottaksharas = 0
    areas = []
    
    for li in segmentation_result["lines"]:
        for wi in segmentation_result["lines"][li]["words"]:
            total_words += 1
            for char in segmentation_result["lines"][li]["words"][wi]["characters"]:
                total_chars += 1
                x, y, w, h = char["bbox"]
                areas.append(w * h)
                if "ottaksharas" in char:
                    total_ottaksharas += len(char["ottaksharas"])
                    for ott in char["ottaksharas"]:
                        x, y, w, h = ott["bbox"]
                        areas.append(w * h)
    
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

def prepare_segmented_image(roi, padding=10):
    """Enhance segmented image by applying Otsu's thresholding and adding padding."""
    _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masked_roi = np.where(mask == 255, roi, 255)
    padded_img = cv2.copyMakeBorder(masked_roi, padding, padding, padding, padding, 
                                    cv2.BORDER_CONSTANT, value=255)
    return padded_img

def save_segmented_images(segmentation_result, output_dir):
    """Save enhanced segmented images to output directory."""
    if not segmentation_result:
        print("No segmentation result to save.")
        return
        
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
        line_img = segmentation_result["lines"][li]["line_img"]
        enhanced_line_img = prepare_segmented_image(line_img)
        line_path = os.path.join(lines_dir, f"line_{li:03d}.png")
        cv2.imwrite(line_path, enhanced_line_img)
        
        for wi in segmentation_result["lines"][li]["words"]:
            word_img = segmentation_result["lines"][li]["words"][wi]["word_img"]
            enhanced_word_img = prepare_segmented_image(word_img)
            word_path = os.path.join(words_dir, f"line_{li:03d}_word_{wi:03d}.png")
            cv2.imwrite(word_path, enhanced_word_img)
            
            for ci, char in enumerate(segmentation_result["lines"][li]["words"][wi]["characters"]):
                char_img = char["char_img"]
                enhanced_char_img = prepare_segmented_image(char_img)
                ctype = char["type"]
                char_path = os.path.join(chars_dir, f"line_{li:03d}_word_{wi:03d}_char_{ci:02d}_{ctype}.png")
                cv2.imwrite(char_path, enhanced_char_img)
                
                if "ottaksharas" in char:
                    for oi, ott in enumerate(char["ottaksharas"]):
                        ott_img = ott["char_img"]
                        enhanced_ott_img = prepare_segmented_image(ott_img)
                        ott_type = ott["type"]
                        ott_path = os.path.join(chars_dir, f"line_{li:03d}_word_{wi:03d}_char_{ci:02d}_{ott_type}_{oi:02d}.png")
                        cv2.imwrite(ott_path, enhanced_ott_img)
    
    print("Segmented images saved successfully!")

def process_image(image_path, output_dir):
    """Process the image and build segmentation result, starting with binarization."""
    # Check if input file exists
    if not Path(image_path).exists():
        print(f"Error: Input file '{image_path}' not found!")
        return None
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Binarize the image
    binarizer = Binarization()
    try:
        binary_image = binarizer.binarize(image)
    except Exception as e:
        print(f"Error during binarization: {e}")
        return None
    
    # Save binarized image
    binary_output_path = os.path.join(output_dir, "binary_image.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    success = cv2.imwrite(binary_output_path, binary_image)
    if not success:
        print(f"Error: Could not save binarized image to '{binary_output_path}'")
        return None
    print(f"Binarized image saved as '{binary_output_path}'")
    
    # Convert binary image to 3-channel for compatibility with deskewing
    image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    # Deskew the binarized image
    image = deskew_image(image)
    
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
                char_data = {"char_img": char_roi, "bbox": char_bbox, "type": "main"}
                char_data["ottaksharas"] = [{"char_img": ott_roi, "bbox": ott_bbox, "type": "ottakshara"} 
                                            for ott_roi, ott_bbox in ottaksharas]
                segmentation_result["lines"][li]["words"][wi]["characters"].append(char_data)
    
    return segmentation_result

if __name__ == "__main__":
    image_path = input("Enter image file path: ")
    output_dir = input("Enter output directory: ")
    
    print("Processing image...")
    seg_result = process_image(image_path, output_dir)
    
    if seg_result:
        compute_metrics(seg_result)
        save_segmented_images(seg_result, output_dir)
    else:
        print("Failed to process image. Please check the file path.")

