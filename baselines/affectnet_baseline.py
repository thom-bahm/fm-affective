import os
# Set the environment variable for cache
# (must be done before importing transformers)
os.environ["HF_HOME"] = "/media/data5/hf_cache"
import re
import argparse
import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image
from typing import Dict

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def load_model(model_name: str):
    """
    Load model and processor based on model name.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        model: Loaded model.
        processor: Loaded processor for the model.
    """
    if model_name == "Qwen2VL":
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype="auto",
            quantization_config=quantization_config,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    elif model_name == "VideoLlava":
        from transformers import VideoLlavaForConditionalGeneration
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            "VideoLlava/VideoLlava-7B",
            torch_dtype="auto",
            quantization_config=quantization_config,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained("VideoLlava/VideoLlava-7B")
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")
    return model, processor

def generate(model, processor, prompt: str, img):
    """
    Generate response from the model based on prompt and image.
    """
    inputs = processor(text=prompt, images=img, padding=True, return_tensors="pt")
    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=100)
    decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return decoded_out[0][-1]

def load_affectnet_annotations(annotations_dir: str) -> Dict[str, int]:
    """
    Load AffectNet annotations from .npy files into a dictionary.
    Args:
        annotations_dir (str): Path to the directory containing .npy annotation files.
    Returns:
        exps (dict): Dictionary mapping image IDs to their classifications.
    """
    exps = {}
    i = 0
    for filename in os.listdir(annotations_dir):
        if i == 1000:  # Limit to 1000 entries for faster processing
            break
        # Extract numeric part of the filename to use as the key
        match = re.match(r'^\d+', filename)
        if match:
            key = match.group()
            file_path = os.path.join(annotations_dir, filename)
            
            # Load the classification label from the .npy file
            if filename.endswith('exp.npy'):
                data = np.load(file_path, allow_pickle=True)
                print(data)
                exps[key] = str(data)  # Convert to string for comparison with model output
            i += 1
    return exps

def classify_affectnet(model_name: str, exps: Dict[str, str], img_folder: str, prompt: str) -> Dict[str, float]:
    """
    Evaluates the model on AffectNet dataset.
    Args:
        model_name (str): Name of the model to evaluate.
        exps (dict): Dictionary mapping image ids (str) to their classifications (str).
        img_folder (str): Path to the images folder relative to the affectnet directory.
        prompt (str): Text prompt to provide to the model.
    Returns:
        results (dict): Dictionary of accuracy scores for each class (0-7).
    """
    print("img_folder: ", img_folder)
    model, processor = load_model(model_name)
    results = {}
    total_corr = 0
    total_imgs = 0

    for img_path in os.listdir(img_folder):
        img_id = img_path.split('.')[0]
        label = exps.get(img_id)
        print(label)
        if label is not None:
            print(os.path.join(img_folder, img_path));
            img = Image.open(os.path.join(img_folder, img_path))
            response = generate(model, processor, prompt, img)

            print(f'IMG {img_id}:\nLabel: {label}, Response: {response}\n')
            if response in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                if response == label:
                    total_corr += 1
                total_imgs += 1

            if total_imgs == 1000:  # Limit processing to 1000 images
                break

    accuracy = total_corr / total_imgs if total_imgs > 0 else 0
    print(f"Accuracy on AffectNet dataset using {model_name}: {accuracy:.4f}")
    results["accuracy"] = accuracy
    return results

if __name__ == "__main__":
    # Parse command-line arguments
    affnet_dir = "/media/data5/affectnet"
    parser = argparse.ArgumentParser(description="Evaluate model on AffectNet dataset.")
    parser.add_argument("--model", type=str, default="Qwen2VL", help="Name of the model to use (default: Qwen2VL)")
    parser.add_argument("--img_folder", type=str, default=f"{affnet_dir}/val_set/images", help="Path to the AffectNet images folder")
    parser.add_argument("--annotations_dir", type=str, default=f"{affnet_dir}/val_set/annotations", help="Path to the AffectNet annotations folder")

    args = parser.parse_args()
    print(args)
    # Load the annotations from .npy files
    exps = load_affectnet_annotations(args.annotations_dir)
 
    prompt = """USER: <image>\n
    Classify the provided face as one of the 10 emotion categories below. State your answer as the number associated with the emotion of the face.
    0. Neutral
    1. Happiness
    2. Sad
    3. Surprise
    4. Fear
    5. Disgust
    6. Anger
    7. Contempt
    8. None
    9. Uncertain
    ASSISTANT: """

    classify_affectnet(args.model, exps, args.img_folder, prompt)
