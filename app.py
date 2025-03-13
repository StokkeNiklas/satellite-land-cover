import streamlit as st
import torch
import torchvision.transforms as transforms
import requests
from io import BytesIO
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import folium
from streamlit_folium import st_folium
import os
from dotenv import load_dotenv  # ✅ Import dotenv

# ✅ Load API Key from .env file
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Set page config
st.set_page_config(page_title="🛰️ Land Cover Prediction", layout="wide")

# Define class labels
CLASS_LABELS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load RGB Model ===
class RGB_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(RGB_Classifier, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = RGB_Classifier(num_classes=10)
    model.load_state_dict(torch.load("rgb_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

rgb_model = load_model()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0).to(device)
    return img

# Function to get satellite image from Google Maps
def fetch_satellite_image(lat, lon):
    if not GOOGLE_MAPS_API_KEY:
        st.error("❌ API Key is missing! Make sure your .env file is loaded correctly.")
        return None

    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=18&size=640x640&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    
    response = requests.get(url)

    # ✅ Check for Google Maps API errors
    if response.status_code != 200:
        st.error(f"Google Maps API Error: {response.status_code}")
        st.error(f"Response Content: {response.content[:500].decode('utf-8', errors='ignore')}")
        return None

    # ✅ Check if response is a valid image
    try:
        image = Image.open(BytesIO(response.content))
        return image.convert("RGB")  # Convert to RGB mode
    except Exception as e:
        st.error("❌ Error: Could not retrieve a valid satellite image.")
        st.error(f"API Response (First 500 chars): {response.content[:500].decode('utf-8', errors='ignore')}")
        return None

# === Streamlit UI ===
st.title("🛰️ Land Cover Prediction from Satellite Imagery")
st.markdown("Click on the map to select a location and predict its land cover.")

# Create the map with Google Satellite Tiles
m = folium.Map(location=[51.505, -0.09], zoom_start=15, tiles=None)

# Add Google Satellite Layer
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google Maps",
    name="Google Satellite",
    overlay=False,
    control=True
).add_to(m)

# Show the map in Streamlit
map_data = st_folium(m, height=500, width=700)

# Detect click event
if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

    # Fetch satellite image
    sat_image = fetch_satellite_image(lat, lon)

    if sat_image:
        # Resize the image to 250x250 pixels
        sat_image = sat_image.resize((250, 250))

        # Save and display the image
        img_path = "satellite_image.jpg"
        sat_image.save(img_path, "JPEG")

        # Preprocess & predict
        img_tensor = preprocess_image(sat_image)
        with torch.no_grad():
            output = rgb_model(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Display Prediction (Text above the image)
        st.markdown(f"### ✅ Predicted Land Cover: **{CLASS_LABELS[predicted_class]}**")
        st.image(img_path, caption=f"🌍 Satellite Image at ({lat}, {lon})", use_container_width=False)

else:
    st.warning("🖱️ Click on the map to select a location for prediction.")
