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
from qwen_vl_utils import process_vision_info

# Quantization configuration for 4 bit
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
        model, processor: Loaded model and processor.
    """
    if model_name == "Qwen2VL":
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype="auto",
            # quantization_config=quantization_config,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    elif model_name == "VideoLlava":
        from transformers import VideoLlavaForConditionalGeneration
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            "VideoLlava/VideoLlava-7B",
            torch_dtype="auto",
            # quantization_config=quantization_config,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained("VideoLlava/VideoLlava-7B")
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")
    return model, processor

def generate_qwen2vl(model, processor, img, prompt=""):
    """
    Generate response using the Qwen2VL model.
    """

    prompt = """Classify the provided face as one of the emotion categories below. State your answer as the number associated with the emotion of the face.
    0. Neutral
    1. Happiness
    2. Sad
    3. Surprise
    4. Fear
    5. Disgust
    6. Anger
    7. Contempt
    """
    # Prepare the messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs using the Qwen-specific processing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def generate_videollava(model, processor, img, prompt=""):
    """
    Generate response using the Video-Llava model.
    """

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


    inputs = processor(text=prompt, images=img, padding=True, return_tensors="pt")
    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=1)
    decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return decoded_out[0][-1]

def load_affectnet_annotations(annotations_dir: str) -> Dict[str, int]:
    """
    Load AffectNet annotations from .npy files into a dictionary.
    """
    exps = {}
    i = 0
    for filename in os.listdir(annotations_dir): 
        match = re.match(r'^\d+', filename)
        if match:
            key = match.group()
            file_path = os.path.join(annotations_dir, filename)
            
            if filename.endswith('exp.npy'):
                data = np.load(file_path, allow_pickle=True)
                exps[key] = str(data)
            i += 1
    return exps

def classify_affectnet(model_name: str, exps: Dict[str, str], img_folder: str) -> Dict[str, float]:
    """
    Evaluates the model on the AffectNet dataset and computes total accuracy
    as well as accuracy for each individual class.
    """
    model, processor = load_model(model_name)
    
    # Initialize the results dictionary to store counts
    results = {
        'total_correct': 0,
        'total_images': 0,
    }
    
    # Add entries for class-specific counts
    class_names = {
        '0': 'Neutral',
        '1': 'Happiness',
        '2': 'Sad',
        '3': 'Surprise',
        '4': 'Fear',
        '5': 'Disgust',
        '6': 'Anger',
        '7': 'Contempt'
    }
    for label, name in class_names.items():
        results[f'{name}_correct'] = 0
        results[f'{name}_total'] = 0

    for img_path in os.listdir(img_folder):
        img_id = img_path.split('.')[0]
        label = exps.get(img_id)

        if label is not None:
            img = Image.open(os.path.join(img_folder, img_path))
            
            generate_map = {
                "Qwen2VL": generate_qwen2vl,
                "VideoLLaVA": generate_videollava,
                # more models here...
            }
            response = generate_map[model_name](model, processor, img=img)

            print(f'IMG {img_id}:\nLabel: {label}, Response: {response}\n')

            # Check if response is a valid label
            if response in class_names:
                # Update total counts
                results['total_images'] += 1
                if response == label:
                    results['total_correct'] += 1
                    results[f'{class_names[response]}_correct'] += 1
                results[f'{class_names[label]}_total'] += 1


    # Calculate total accuracy
    results['total_accuracy'] = results['total_correct'] / results['total_images'] if results['total_images'] > 0 else 0

    # Calculate accuracy for each class
    for label, name in class_names.items():
        correct = results[f'{name}_correct']
        total = results[f'{name}_total']
        results[f'{name}_accuracy'] = correct / total if total > 0 else 0.0

    return results

if __name__ == "__main__":
    # Parse command-line arguments
    affnet_dir = "/media/data5/affectnet"
    parser = argparse.ArgumentParser(description="Evaluate model on AffectNet dataset.")
    parser.add_argument("--model", type=str, default="Qwen2VL", help="Name of the model to use (default: Qwen2VL)")
    parser.add_argument("--img_folder", type=str, default=f"{affnet_dir}/val_set/images", help="Path to the AffectNet images folder")
    parser.add_argument("--annotations_dir", type=str, default=f"{affnet_dir}/val_set/annotations", help="Path to the AffectNet annotations folder")

    args = parser.parse_args()

    # Load the annotations from .npy files
    exps = load_affectnet_annotations(args.annotations_dir)

    results = classify_affectnet(args.model, exps, args.img_folder)
    print(results)
    np.save('./affectnet_baseline_qwen2vl', results)
