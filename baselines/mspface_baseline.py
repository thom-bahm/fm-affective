import os
# Set the environment variable for cache
# (must be done before importing transformers)
os.environ["HF_HOME"] = "/media/data5/hf_cache"
import json
import torch
from torchvision import io
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import Dict, List, Optional
from utils import generate_confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
LABELS_FILE = '/media/data5/thomas/MSP_Face/Labels/label_consensus.json'
VIDEO_DIR = '/media/data5/MSP_Face_segments/Segments'
EMOTION_MAP = {
    "H": "Happy",
    "A": "Angry",
    "S": "Sad",
    "N": "Neutral",
    "D": "Disgust",
    "F": "Fear",
    "U": "Surprise",
    "C": "Contempt"
}

# Load model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="cuda")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def load_labels(labels_file: str) -> Dict[str, str]:
    """
    Loads the ground truth labels from the JSON file.
    """
    with open(labels_file, 'r') as file:
        data = json.load(file)
    
    labels = {}
    for video, details in data.items():
        if details:
            labels[video] = details[0]["EmoClass_Major"]
    return labels

def fetch_video(video_path: str, nframe_factor: int = 2, fps: float = 1.0) -> Optional[torch.Tensor]:
    """
    Reads a video from the given path and extracts a specified number of frames.
    """
    try:
        video, _, info = io.read_video(video_path, output_format="TCHW")
        nframes = round(video.size(0) / info["video_fps"] * fps)
        idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
        return video[idx]
    except Exception as e:
        logging.error(f"Error reading video {video_path}: {e}")
        return None

def run_inference_on_video(video_path: str) -> Optional[str]:
    """
    Runs the Qwen2VL model on the given video and returns the generated emotion label.
    """
    logging.info(f"Processing video: {video_path}")
    
    video_tensor = fetch_video(video_path)
    if video_tensor is None:
        return None

    # Prepare the conversation template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Classify the emotion in the video."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Preprocess inputs
    inputs = processor(
        text=[text_prompt],
        videos=[video_tensor],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    # Generate output
    try:
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if output_text:
            return output_text[0].strip()
        return None
    except Exception as e:
        logging.error(f"Error during inference on {video_path}: {e}")
        return None

def classify_mspface(labels_file: str, video_dir: str):
    """
    Performs classification on all videos and evaluates the results.
    """
    ground_truth = load_labels(labels_file)
    mp4_files = get_all_mp4_files(video_dir)
    logging.info(f"Found {len(mp4_files)} videos for processing.")
    
    predictions = {}

    # Run inference on all videos and collect predictions
    for video_path in mp4_files:
        video_name = os.path.basename(video_path)
        predicted_emotion = run_inference_on_video(video_path)
        if predicted_emotion:
            # Map the output to the defined emotion categories
            for key, emotion in EMOTION_MAP.items():
                if emotion.lower() in predicted_emotion.lower():
                    predictions[video_name] = key
                    break
    
    # Evaluate predictions against ground truth
    y_true = []
    y_pred = []

    i = 0
    for video, true_label in ground_truth.items():
        if i > 40: break
        predicted_label = predictions.get(video)
        if predicted_label is not None:
            y_true.append(true_label)
            y_pred.append(predicted_label)

    # Generate and display confusion matrix
    class_names = list(EMOTION_MAP.keys())
    confusion_matrix = generate_confusion_matrix(y_true, y_pred, class_names)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def get_all_mp4_files(root_dir: str) -> List[str]:
    """
    Recursively scans the given directory for .mp4 files and returns their paths.
    """
    mp4_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

if __name__ == "__main__":
    classify_mspface(LABELS_FILE, VIDEO_DIR)

