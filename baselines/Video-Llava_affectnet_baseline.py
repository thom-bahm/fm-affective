import os
import numpy as np
import re
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, BitsAndBytesConfig, VideoLlavaProcessor
from PIL import Image

afnet_dir = "../../../affectnet/eval_set"

def class_accuracy(exps : dict, img_folders : list) -> dict:
    """
    Returns a dictionary containing the accuracy scores on each class (0,1,...,7)
    Args: 
    exps (dict): a dictonary mapping image ids (int) to their classifications (int)
    img_folders (list): A list of file paths for image folders (strings). In this case there will be 2 filepaths, {afnet_dir/images} and {afnet_dir/other_images}
    """
    results = {}
    total_corr = 0; total_imgs = 0
    
    for img_folder in img_folders:
        for class_folder in os.listdir(img_folder):
            class_path = os.path.join(img_folder, class_folder)
            
            corr = 0
            class_total = 0
            
            # Process each image in the current class folder
            for img_path in os.listdir(class_path):
                img_id = img_path.split('.')[0]
                label = exps.get(img_id)
                
                if label is not None:
                    img = Image.open(os.path.join(class_path, img_path))
                    inputs = processor(text=[prompt], images=[img], padding=True, return_tensors="pt")
                    out = model.generate(**inputs, max_new_tokens=100)
                    decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    response = decoded_out[0][-1]
                    
                    print(f'IMG {img_id}:\nLabel: {label}, Response: {response}\nFull response: {decoded_out}')
                    if response in {'0','1','2','3','4','5','6','7','8','9'}:
                        if response == label:
                            corr += 1
                        class_total += 1
            
            # Calculate and store accuracy for the current class
            if class_total > 0:
                results[class_folder] = corr / class_total
                total_corr += corr
                total_imgs += class_total
            else:
                results[class_folder] = None  # No images in this class folder

        if total_imgs == 10: break
        
    results['average'] = total_corr / total_imgs if total_imgs > 0 else None
    return results

exps = {}
lnds = {}

# Create a dictionary 'exps' that stores an image id as a key, and the expression 
# classification of that image as an int.
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
            # print(f'{key}, {exps[key]}')
                        
        i+=1
            

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
                                                           device_map="auto")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

prompt = """USER: <image>\n
Classify the provided face as one of the 10 emotion categories below. State your answer just as the number associated with the emotion.
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


img_folders = [f'{afnet_dir}/images', f'{afnet_dir}/other_images']

results = class_accuracy(exps=exps, img_folders=img_folders)
print(results)
print(f"Average: {results['average']}")

# inputs = processor(text=[prompt] * len(images), images=images, padding=True, return_tensors="pt")
# out = model.generate(**inputs, max_new_tokens=100)
# decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# response = decoded_out[0][-1]

