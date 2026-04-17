from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import requests

def main():
    model_name = "facebook/deit-tiny-patch16-224"
    local_weights_path = "pytorch_model.bin"
    
    print(f"Loading image processor for {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    print(f"Loading model {model_name} with local weights from {local_weights_path}...")
    # Load the local weights
    state_dict = torch.load(local_weights_path, map_location="cpu")
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    
    # Initialize the model with the configuration from the Hub
    model = AutoModelForImageClassification.from_config(config)
    
    # Load the local weights, ignore mismatched sizes strictly if needed
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("Fetching an example image for testing...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    print("Running inference...")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    print("\n--- Prediction Result ---")
    print(f"Predicted class ID: {predicted_class_idx}")
    print(f"Predicted class label: {model.config.id2label[predicted_class_idx]}")

if __name__ == "__main__":
    main()
