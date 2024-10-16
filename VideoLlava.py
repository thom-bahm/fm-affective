import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, BitsAndBytesConfig, VideoLlavaProcessor
from huggingface_hub import hf_hub_download
from PIL import Image

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model in half-precision
model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",
                                                           quantization_config=quantization_config,
                                                           device_map="auto")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
# Load the video as an np.array, sampling uniformly 8 frames
# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
# container = av.open(video_path)
# total_frames = container.streams.video[0].frames
# indices = np.arange(0, total_frames, total_frames / 8).astype(int)
# video = read_video_pyav(container, indices)

# For better results, we recommend to prompt the model in the following format
# prompt = "USER: <video>\nWhy is this funny? ASSISTANT:"
# inputs = processor(text=prompt, videos=video, return_tensors="pt")
img_1 = Image.open('./eval_set/0/2490.jpg')
img_2 = Image.open('./eval_set/0/2498.jpg')
print("image opened")
prompt = """USER: <image>\n
Describe the following face as one of the 10 emotion categories below, and explain your reasoning:
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
ASSISTANT:"""
inputs = processor(text=prompt, images=img_2, padding=True, return_tensors="pt")


out = model.generate(**inputs, max_new_tokens=100)
decoded_out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(out)
print(out.shape)
print(decoded_out)
