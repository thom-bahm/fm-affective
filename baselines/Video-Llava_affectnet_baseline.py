import os
import numpy as np
import re
import torch
from transformers import VideoLlavaForConditionalGeneration, BitsAndBytesConfig, VideoLlavaProcessor
from PIL import Image

afnet_dir = "../../../affectnet/val_set"

def generate(prompt : str, img):
    inputs = processor(text=prompt, images=img, padding=True, return_tensors="pt")
    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=100)
    decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    response = decoded_out[0][-1]
    return response

    
def classify_affectnet(exps: dict, img_folder: str, prompt : str) -> dict:
    """
    Returns a dictionary containing the accuracy scores on each class (0,1,...,7)
    Args:
    exps (dict): A dictionary mapping image ids (int) to their classifications (int)
    img_folder (str): Path to the images folder relative to the affectnet directory
    """
    results = {}
    total_corr = 0; total_imgs = 0

    # Process each image in the images directory
    for img_path in os.listdir(img_folder):
        img_id = img_path.split('.')[0]
        label = exps.get(img_id)

        if label is not None:
            img = Image.open(os.path.join(img_folder, img_path))
            # Generate response for image using given prompt
            response = generate(prompt, img)
            
            print(f'IMG {img_id}:\nLabel: {label}, Response: {response}\n')
            if response in {'0','1','2','3','4','5','6','7','8','9'}:
                if response == label:
                    total_corr += 1
                total_imgs += 1

        if total_imgs == 1000:  # Limit processing to 1000 images
            break

    # Calculate overall accuracy
    results['average'] = total_corr / total_imgs if total_imgs > 0 else None
    return results

# Create a dictionary 'exps' that stores an image id as a key, and the expression
# classification of that image as an int.
exps = {}

i = 0
for filename in os.listdir(f'{afnet_dir}/annotations'):
    if i == 1000: break
    # Get numeric part of npy files (the key)
    match = re.match(r'^\d+', filename)

    if match:
        key = match.group()
        file_path = os.path.join(f'{afnet_dir}/annotations', filename)

        if filename.endswith('exp.npy'):
            # Load data from .npy files and assign key to value
            data = np.load(file_path, allow_pickle=True)
            exps[key] = data
        i += 1

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",
                                                           quantization_config=quantization_config,
                                                           torch_dtype=torch.float16,
                                                           attn_implementation="flash_attention_2",
                                                           device_map="cuda:0")
model.tie_weights()
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
processor.tokenizer.padding_side = "left"
prompt = """USER: <image>\n
Classify the face shown in the image as one of the 10 categories below. State your answer as the number associated with the emotion of the face.
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

img_folder = f'{afnet_dir}/images'

results = classify_affectnet(exps=exps, img_folder=img_folder, prompt=prompt)
print(results)
print(f"Accuracy percentage: {results['average']}")
