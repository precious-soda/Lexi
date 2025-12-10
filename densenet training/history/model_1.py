import os
import glob
import random
import time
import warnings
from collections import Counter
import torchmetrics
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.v2 as T 
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

class CFG:
    DATASET_DIR = 'handwritten-kannada-characters'
    CHARACTERS_FOLDER = os.path.join(DATASET_DIR, 'Characters')
    OUTPUT_DIR = 'output1'
    MODEL_NAME = 'kannada_densenet121_attention'
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs', MODEL_NAME)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints', MODEL_NAME)
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations', MODEL_NAME)
    
    IMAGE_SIZE = 52 
    NUM_CLASSES = 621
    VALID_SPLIT = 0.10
    TEST_SPLIT = 0.10 
    
    MODEL_ARCHITECTURE = 'densenet121' 
    PRETRAINED = True 
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 20
    BATCH_SIZE = 128
    NUM_WORKERS = 4 
    PIN_MEMORY = True if DEVICE == torch.device("cuda") else False
    LEARNING_RATE_MAX = 5e-4
    WEIGHT_DECAY = 0.01
    OPTIMIZER = 'AdamW'
    SCHEDULER = 'OneCycleLR'
    USE_AMP = True 
    
    METRICS_AVERAGE = 'weighted'
    SEED = 42  # Added for reproducibility
    TSNE_SAMPLE_SIZE = 1000  # Added for t-SNE sampling
    N_MISCLASSIFIED_EXAMPLES = 16  # Added for misclassified examples
    N_AUGMENTATION_EXAMPLES = 16  # Added for augmentation visualization

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
def get_data_paths_and_labels(characters_dir):
    if not os.path.exists(characters_dir):
        raise FileNotFoundError(f"Dataset directory {characters_dir} does not exist")
    
    samples = []
    class_folders = sorted(glob.glob(os.path.join(characters_dir, '[0-9]*')),
                           key=lambda x: int(os.path.basename(x)))
    expected_indices = set(range(CFG.NUM_CLASSES))
    found_indices = set()

    for class_folder in class_folders:
        try:
            class_index = int(os.path.basename(class_folder))
            found_indices.add(class_index)
        except ValueError:
            print(f"Warning: Non-integer folder name found: {class_folder}. Skipping.")
            continue

        if not 0 <= class_index < CFG.NUM_CLASSES:
            print(f"Warning: Folder index {class_index} out of expected range [0, {CFG.NUM_CLASSES-1}]. Skipping.")
            continue

        image_paths = glob.glob(os.path.join(class_folder, '*.jpg'))
        if not image_paths:
            print(f"Warning: No JPG images found in folder: {class_folder}")
            continue

        for img_path in image_paths:
            samples.append((img_path, class_index))
    
    if len(found_indices) != CFG.NUM_CLASSES:
        missing = expected_indices - found_indices
        extra = found_indices - expected_indices
        error_msg = ""
        if missing:
            error_msg += f"Expected class indices missing: {sorted(list(missing))}\n"
        if extra:
            error_msg += f"Unexpected class indices found: {sorted(list(extra))}\n"
        raise ValueError(f"Expected {CFG.NUM_CLASSES} classes, but found {len(found_indices)}. {error_msg}")
        
    if not samples:
        raise FileNotFoundError(f"No valid image samples found in {characters_dir}")

    print(f"Found {len(samples)} image samples across {len(found_indices)} classes.")
    return samples

def calculate_norm_stats(image_paths):
    print(f"Calculating normalization stats from {len(image_paths)} training images...")
    pixel_sum = torch.zeros(1, dtype=torch.float64)
    pixel_sum_sq = torch.zeros(1, dtype=torch.float64)
    total_pixels = 0
    batch_size = 512 

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Calculating Stats"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for img_path in batch_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load image {img_path} during stat calculation. Skipping.")
                continue
            
            if img.ndim == 2: 
                img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0) 
                images.append(img_tensor)
            else:
                print(f"Warning: Image {img_path} has unexpected shape {img.shape}. Skipping.")

        if not images: continue

        batch_tensor = torch.stack(images) 
        pixel_sum += torch.sum(batch_tensor)
        pixel_sum_sq += torch.sum(batch_tensor ** 2)
        total_pixels += batch_tensor.numel()

    if total_pixels == 0:
        raise ValueError("No valid pixels found to calculate normalization stats.")

    mean = (pixel_sum / total_pixels).item()
    variance = (pixel_sum_sq / total_pixels) - (mean ** 2)
    std = torch.sqrt(torch.clamp(torch.tensor(variance), min=1e-6)).item()

    print(f"Calculated Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std

class KannadaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples 
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"cv2.imread failed for path: {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Returning placeholder.")
            raise RuntimeError(f"Failed to load or process image: {image_path}") from e
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(train_mean, train_std):
    train_tensor_transforms = T.Compose([
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ])

    train_transform = T.Compose([
        T.ToImage(), 
        T.ToDtype(torch.float32, scale=True), 
        train_tensor_transforms,
        T.Normalize(mean=[train_mean], std=[train_std])
    ])

    val_test_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[train_mean], std=[train_std])
    ])

    return train_transform, val_test_transform

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
    print(f"Building model: {cfg.MODEL_ARCHITECTURE} with attention")
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
    return model.to(cfg.DEVICE)

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device, epoch, writer):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_batches = len(loader)
    valid_batches = 0

    pbar = tqdm(enumerate(loader), total=num_batches, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Train]")

    for batch_idx, (inputs, labels) in pbar:
        if inputs is None or labels is None:
            print(f"Skipping batch {batch_idx} due to invalid data")
            continue

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        if CFG.USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler and CFG.SCHEDULER == 'OneCycleLR':
            scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        valid_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
    accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Epoch {epoch+1} Train Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    writer.add_scalar('Accuracy/train_epoch', accuracy, epoch)
    writer.add_scalar('LearningRate/epoch', optimizer.param_groups[0]['lr'], epoch)

    return avg_loss, accuracy

def validate_one_epoch(model, loader, loss_fn, metrics_calculator, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)
    metrics_calculator.reset()

    pbar = tqdm(loader, total=num_batches, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Val]")

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            metrics_calculator.update(outputs, labels)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches
    epoch_metrics = metrics_calculator.compute()
    accuracy = epoch_metrics['val_Accuracy'].item() * 100.0 

    print(f"Epoch {epoch+1} Val Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
    for name, value in epoch_metrics.items():
        writer.add_scalar(f"{name.replace('val_', 'Accuracy/')}/val_epoch", value.item(), epoch) 

    return avg_loss, accuracy

@torch.no_grad()
def evaluate_model(model, loader, loss_fn, metrics_calculator, device, class_names):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)
    metrics_calculator.reset()
    all_preds = []
    all_labels = []
    all_features = []

    features = {}
    def get_features_hook(mod, input, output):
        features['feats'] = torch.flatten(output, 1)

    feature_layer = model.features.norm5
    hook_handle = feature_layer.register_forward_hook(get_features_hook)

    print("Running final evaluation on test set...")
    pbar = tqdm(loader, total=num_batches, desc="Evaluating Test Set")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast('cuda'):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        total_loss += loss.item()
        metrics_calculator.update(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_features.append(features['feats'].cpu().numpy())

    hook_handle.remove()

    avg_loss = total_loss / num_batches
    final_metrics = metrics_calculator.compute()

    print("\n--- Test Set Evaluation Results ---")
    print(f"Average Loss: {avg_loss:.4f}")
    for name, value in final_metrics.items():
        print(f"{name.replace('test_', '')}: {value.item():.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_features = np.concatenate(all_features, axis=0)

    print("\nClassification Report:")
    try:
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        print(report)
    except ValueError:
        print("Classification Report (numeric labels):")
        report = classification_report(all_labels, all_preds, digits=4)
        print(report)

    os.makedirs(CFG.VISUALIZATION_DIR, exist_ok=True)
    
    plot_cf_matrix(all_labels, all_preds, class_names, CFG.VISUALIZATION_DIR)
    
    plot_misclassified(model, loader.dataset, all_labels, all_preds, class_names, CFG.VISUALIZATION_DIR, CFG.DEVICE)
    
    plot_tsne(all_features, all_labels, class_names, CFG.VISUALIZATION_DIR, CFG.TSNE_SAMPLE_SIZE)

    return final_metrics

def plot_cf_matrix(labels, preds, class_names, output_dir):
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(15, 15))

    try:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, cbar=True)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix Heatmap (Density Plot)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        cf_path = os.path.join(output_dir, 'confusion_matrix_heatmap.png')
        plt.savefig(cf_path)
        print(f"Saved confusion matrix heatmap to {cf_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate confusion matrix heatmap: {e}")

    np.fill_diagonal(cm, 0)
    flat_cm = cm.flatten()
    top_indices = np.argsort(flat_cm)[-20:]
    top_values = flat_cm[top_indices]

    print("\nTop 20 Misclassifications (True Class -> Predicted Class: Count):")
    rows, cols = np.unravel_index(top_indices, cm.shape)
    for r, c, v in zip(reversed(rows), reversed(cols), reversed(top_values)):
        if v > 0:
            true_char = class_names[r] if r < len(class_names) else str(r)
            pred_char = class_names[c] if c < len(class_names) else str(c)
            print(f"  '{true_char}' -> '{pred_char}': {v}")

def plot_misclassified(model, test_dataset, labels, preds, class_names, output_dir, device):
    print("\nGenerating Misclassified Examples Plot...")
    misclassified_indices = np.where(preds != labels)[0]
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return

    num_to_plot = min(CFG.N_MISCLASSIFIED_EXAMPLES, len(misclassified_indices))
    plot_indices = random.sample(list(misclassified_indices), num_to_plot)

    grid_size = int(np.ceil(np.sqrt(num_to_plot)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    print(f"Plotting {num_to_plot} misclassified examples...")
    plotted_count = 0
    for i, idx in enumerate(plot_indices):
        if plotted_count >= len(axes): break

        img_path, true_label = test_dataset.samples[idx]
        pred_label = preds[idx]

        try:
            img_display = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_display is None: continue

            ax = axes[plotted_count]
            ax.imshow(img_display, cmap='gray')
            true_char = class_names[true_label] if true_label < len(class_names) else str(true_label)
            pred_char = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
            ax.set_title(f"True: '{true_char}'\nPred: '{pred_char}'", fontsize=9)
            ax.axis('off')
            plotted_count += 1
        except Exception as e:
            print(f"Error plotting misclassified image {img_path}: {e}")

    for j in range(plotted_count, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    mc_path = os.path.join(output_dir, 'misclassified_examples.png')
    plt.savefig(mc_path)
    print(f"Saved misclassified examples plot to {mc_path}")
    plt.close(fig)

def plot_tsne(features, labels, class_names, output_dir, sample_size):
    print("\nGenerating t-SNE plot...")
    if len(features) > sample_size:
        print(f"Sampling {sample_size} points for t-SNE...")
        indices = np.random.choice(len(features), sample_size, replace=False)
        features = features[indices]
        labels = labels[indices]

    try:
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, random_state=CFG.SEED)
        tsne_results = tsne.fit_transform(features)

        df = pd.DataFrame()
        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]
        df['y'] = labels
        
        map_to_char = len(np.unique(labels)) < 50
        if map_to_char:
            df['label'] = df['y'].apply(lambda i: class_names[i] if i < len(class_names) else str(i))
        else:
            df['label'] = df['y']

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label" if map_to_char else None,
            palette=sns.color_palette("hsv", len(np.unique(df['label']))),
            data=df,
            legend="full" if map_to_char else False,
            alpha=0.6
        )
        plt.title('t-SNE projection of features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        if map_to_char:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        tsne_path = os.path.join(output_dir, 'tsne_features.png')
        plt.savefig(tsne_path)
        print(f"Saved t-SNE plot to {tsne_path}")
        plt.close()

    except Exception as e:
        print(f"Could not generate t-SNE plot: {e}")

def visualize_augmentations(train_loader, class_names, output_dir, train_mean, train_std):
    print("\nVisualizing Augmentations...")
    try:
        inputs, labels = next(iter(train_loader))
        num_images = min(CFG.N_AUGMENTATION_EXAMPLES, len(inputs))
        grid_size = int(np.ceil(np.sqrt(num_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()

        inv_normalize = T.Normalize(
            mean=[-m/s for m, s in zip([train_mean], [train_std])],
            std=[1/s for s in [train_std]]
        )

        for i in range(num_images):
            ax = axes[i]
            img_tensor = inputs[i]
            img_denorm = inv_normalize(img_tensor)
            img_display = torch.clamp(img_denorm, 0, 1)
            img_display_np = img_display.squeeze().cpu().numpy()

            label_idx = labels[i].item()
            label_char = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)

            ax.imshow(img_display_np, cmap='gray')
            ax.set_title(f"Augmented\nClass: '{label_char}'", fontsize=9)
            ax.axis('off')

        for j in range(num_images, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        aug_path = os.path.join(output_dir, 'augmented_examples.png')
        plt.savefig(aug_path)
        print(f"Saved augmented examples plot to {aug_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Could not visualize augmentations: {e}")

if __name__ == '__main__':
    start_time = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    
    seed_everything(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CFG.LOG_DIR, exist_ok=True)
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.VISUALIZATION_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=CFG.LOG_DIR)
    print(f"Configuration:\n{ {k: v for k, v in CFG.__dict__.items() if not k.startswith('__')} }")
    print(f"Using device: {CFG.DEVICE}")
    
    print("\n--- Loading Data ---")
    all_samples = get_data_paths_and_labels(CFG.CHARACTERS_FOLDER)
    labels_only = [s[1] for s in all_samples]
    class_names = ["" for _ in range(CFG.NUM_CLASSES)]
    
    label_csv_path = os.path.join(CFG.DATASET_DIR, 'label.csv')
    if os.path.exists(label_csv_path):
        try:
            df_labels = pd.read_csv(label_csv_path, header=None, names=['Folder_Index', 'Kannada_Character'])
            df_labels = df_labels.sort_values('Folder_Index')
            class_names = df_labels['Kannada_Character'].tolist()
            print("Loaded class names from label.csv")
        except Exception as e:
            print(f"Warning: Could not load class names from {label_csv_path}: {e}. Using numeric labels.")
            class_names = [str(i) for i in range(CFG.NUM_CLASSES)]
    else:
        print("Warning: label.csv not found. Using numeric labels for classes.")
        class_names = [str(i) for i in range(CFG.NUM_CLASSES)]
    
    train_val_samples, test_samples, train_val_labels, _ = train_test_split(
        all_samples, labels_only,
        test_size=CFG.TEST_SPLIT,
        random_state=CFG.SEED,
        stratify=labels_only
    )
    
    val_split_adjusted = CFG.VALID_SPLIT / (1.0 - CFG.TEST_SPLIT)
    train_samples, val_samples, _, _ = train_test_split(
        train_val_samples, train_val_labels,
        test_size=val_split_adjusted,
        random_state=CFG.SEED,
        stratify=train_val_labels
    )
    print(f"Data split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    train_image_paths = [s[0] for s in train_samples]
    train_mean, train_std = calculate_norm_stats(train_image_paths)
    
    train_transform, val_test_transform = get_transforms(train_mean, train_std)
    
    train_dataset = KannadaDataset(train_samples, transform=train_transform)
    val_dataset = KannadaDataset(val_samples, transform=val_test_transform)
    test_dataset = KannadaDataset(test_samples, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                             num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY)
    
    visualize_augmentations(train_loader, class_names, CFG.VISUALIZATION_DIR, train_mean, train_std)
    
    print("\n--- Setting up Model & Training Components ---")
    model = build_model(CFG)
    
    if CFG.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE_MAX, weight_decay=CFG.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {CFG.OPTIMIZER}")

    if CFG.SCHEDULER == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CFG.LEARNING_RATE_MAX,
            epochs=CFG.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
    else:
        scheduler = None

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')
    
    metrics = torchmetrics.MetricCollection({
        'Accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=CFG.NUM_CLASSES, average=CFG.METRICS_AVERAGE),
        'Precision': torchmetrics.Precision(task="multiclass", num_classes=CFG.NUM_CLASSES, average=CFG.METRICS_AVERAGE),
        'Recall': torchmetrics.Recall(task="multiclass", num_classes=CFG.NUM_CLASSES, average=CFG.METRICS_AVERAGE),
        'F1Score': torchmetrics.F1Score(task="multiclass", num_classes=CFG.NUM_CLASSES, average=CFG.METRICS_AVERAGE)
    })
    val_metrics = metrics.clone(prefix='val_').to(CFG.DEVICE)
    test_metrics = metrics.clone(prefix='test_').to(CFG.DEVICE)
    
    print("\n--- Starting Training ---")
    best_val_accuracy = 0.0
    best_epoch = -1
    patience = 5
    epochs_no_improve = 0

    for epoch in range(CFG.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, CFG.DEVICE, epoch, writer)
        val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn, val_metrics, CFG.DEVICE, epoch, writer)
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            checkpoint_path = os.path.join(CFG.CHECKPOINT_DIR, f"{CFG.MODEL_NAME}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_accuracy': best_val_accuracy,
                'train_mean': train_mean,
                'train_std': train_std,
            }, checkpoint_path)
            print(f"Epoch {epoch+1}: Validation accuracy improved to {val_acc:.2f}%. Checkpoint saved to {checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    print(f"\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at Epoch {best_epoch+1}")
    
    print("\n--- Loading Best Model for Final Evaluation ---")
    best_checkpoint_path = os.path.join(CFG.CHECKPOINT_DIR, f"{CFG.MODEL_NAME}_best.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=CFG.DEVICE)
        
        model = build_model(CFG)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model checkpoint from {best_checkpoint_path}")
        
        loaded_mean = checkpoint.get('train_mean', train_mean)
        loaded_std = checkpoint.get('train_std', train_std)
        if loaded_mean != train_mean or loaded_std != train_std:
            print("Warning: Normalization stats from checkpoint differ from current run.")
            
        evaluate_model(model, test_loader, loss_fn, test_metrics, CFG.DEVICE, class_names)
    else:
        print("Warning: Best checkpoint not found. Evaluating the model from the last epoch.")
        evaluate_model(model, test_loader, loss_fn, test_metrics, CFG.DEVICE, class_names)

    writer.close()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time / 60:.2f} minutes")
    print("--- Script Finished ---")