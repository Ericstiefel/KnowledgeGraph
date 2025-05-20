from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

def getCaptionFromImage(image: Image.Image):
    checkpoint = "ydshieh/vit-gpt2-coco-en"

    # Load components
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    feature_extractor = ViTImageProcessor.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Ensure RGB mode
    image = image.convert("RGB")

    # Preprocess image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=50, num_beams=5)

    # Decode token IDs to text
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption
