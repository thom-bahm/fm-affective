import os
import numpy as np
import re
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, BitsAndBytesConfig, VideoLlavaProcessor
from PIL import Image

afnet_dir = "../../../affectnet/eval_set"

exps = {}
lnds = []
slnds = []

i = 0
for filename in os.listdir(f'{afnet_dir}/annotations'):
    if i == 500: break
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


images = []

corr = 0
for img_path in os.listdir(f'{afnet_dir}/images/0'):
    key = img_path.split('.')[0]
    label = exps.get(key)
    if not label == None:
        images.append(Image.open(f'{afnet_dir}/images/0/{img_path}'))
        img = Image.open(f'{afnet_dir}/images/0/{img_path}')
        
        inputs = processor(text=[prompt], images=[img], padding=True, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=100)
        decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response = decoded_out[0][-1]
        
        if response == label: corr += 1

print(f'Classified {corr}/{len(images)} images correct')
        

        
    


# inputs = processor(text=[prompt] * len(images), images=images, padding=True, return_tensors="pt")

# out = model.generate(**inputs, max_new_tokens=100)
# decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# response = decoded_out[0][-1]

