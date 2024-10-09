from PIL import Image

def load_eval_subset_affectnet(start_index : int, n : int):
    i = start_index
    while i < start_index + n:
        try:
            img = Image.open(f'../../affectnet/eval_set/images/0/{str(i)}.jpg')
            img.save(f'./eval_set/0/{i}.jpg')
        except:
            print(f"Error loading or saving image with index '{i}'")
        i = i + 1

load_eval_subset_affectnet(280, 100)