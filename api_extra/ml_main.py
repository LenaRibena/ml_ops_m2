from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


def predict_step(image_paths: list[str]):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Setting up!")
    global model, feature_extractor, tokenizer, device, gen_kwargs
    
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
        
    
    yield
    print("Cleaning up!")


app = FastAPI(lifespan=lifespan)

@app.post("/caption/")
async def caption(image: UploadFile = File(...)):
    with open("image.jpg", "wb") as buffer:
        buffer.write(await image.read())
    return predict_step(["image.jpg"])