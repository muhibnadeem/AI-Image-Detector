# from PIL import Image
# import torch
# from torchvision import transforms
# from model import get_model

# model = get_model()
# model.load_state_dict(torch.load("model.pt"))
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])

# img = Image.open("test.jpg")
# img = transform(img).unsqueeze(0)

# with torch.no_grad():
#     output = model(img)
#     prediction = torch.argmax(output, 1)

# print("AI" if prediction.item() == 1 else "Real")
from PIL import Image
import torch
from torchvision import transforms
from model import get_model

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.load_state_dict(torch.load("model.pt", map_location=device)) # Handle CPU/GPU loading
model.to(device)
model.eval() # Set to evaluation mode (important!)

# 2. TRANSFORM (Must match training!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. PREDICT
def predict(image_path):
    try:
        img = Image.open(image_path).convert('RGB') # Ensure it's RGB
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            # Apply Softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get the likely class
            predicted_class_index = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_index].item()

        # NOTE: Check your 'data' folder to see which class is 0 and which is 1.
        # Usually, ImageFolder sorts alphabetically. 
        # If folders are 'ai' and 'real': 0=ai, 1=real
        classes = ['AI', 'Real'] # Update this based on your folder names!
        result = classes[predicted_class_index]
        
        print(f"Prediction: {result} ({confidence*100:.2f}%)")
        
    except Exception as e:
        print(f"Error: {e}")

# Run
predict("test.jpg")