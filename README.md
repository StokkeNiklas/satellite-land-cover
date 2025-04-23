# Satellite Land Cover Classification with EuroSAT

This project trains convolutional neural networks (CNNs) to classify land cover types using the EuroSAT dataset. The trained model is deployed as an interactive web application using Streamlit.

Try the deployed app here: 👉 [Open in Streamlit](https://satellite-land-cover-mpskprq6od3uxkbdrnd8jp.streamlit.app)

# Reproducing the Results

To train the models and replicate the results, you can either train locally on your computer or use Google Colab. Follow the steps below based on your preferred method.

## Option 1: Train Locally

1. **Download the EuroSAT Dataset**\
   Download the EuroSAT RGB and MS datasets from:\
   👉 https://zenodo.org/record/7711810

2. **Organize the Dataset**\
   Unzip the datasets and place the folders in the following structure:

   ```
   Datasets/
   ├── EuroSAT_RGB/
   └── EuroSAT_MS/
   ```

3. **Set Up the Environment**

   - Ensure Python 3.8+ is installed.

   - Clone this repository:

     ```
     git clone 
     cd 
     ```

   - Install the required dependencies:

     ```
     pip install -r requirements.txt
     ```

4. **Train the Model**

   - Run the training script to train the CNN on the EuroSAT RGB dataset:

     ```
     python train.py
     ```

   - The trained model will be saved as `rgb_model.pth` in the project directory.

5. **Run the Streamlit App**

   - Launch the Streamlit app to interact with the trained model:

     ```
     streamlit run app.py
     ```

   - Open the provided URL (usually `http://localhost:8501`) in your browser to use the app.

## Option 2: Train Using Google Colab

1. **Access the Colab Notebook**\
   Open the provided Google Colab notebook:\
   👉 https://colab.research.google.com/drive/1FVT347pcCLGu8AdSejU-A2zkRf35S2e-?usp=sharing

2. **Set Up the Dataset**

   - Download the EuroSAT RGB and MS datasets from:\
     👉 https://zenodo.org/record/7711810
   - Upload and unzip the datasets to your Google Drive in the following structure:

     <pre>
     MyDrive/Datasets/
     ├── EuroSAT_RGB/
     └── EuroSAT_MS/
     </pre>

   - Ensure the `Datasets` folder is located in the `MyDrive` root directory of your Google Drive.

3. **Run the Notebook**

   - Follow the instructions in the notebook to mount your Google Drive, access the dataset at `/content/drive/MyDrive/Datasets`, train the model, and save the trained model as `rgb_model.pth`.
   - The notebook includes all necessary code to set up the environment and train the model.

4. **Download the Trained Model**

   - After training, download the `rgb_model.pth` file from Colab to your local machine.

5. **Run the Streamlit App Locally**

   - Clone this repository and install dependencies as described in **Option 1, Step 3**.

   - Place the downloaded `rgb_model.pth` file in the project directory.

   - Run the Streamlit app:

     <pre>
     streamlit run app.py
     </pre>
# Notes

- The Streamlit app (`app.py`) requires the `rgb_model.pth` file to function. Ensure it is in the project directory before running the app.
- For local training, ensure you have a compatible GPU and the necessary libraries (e.g., PyTorch) installed for faster training.
- The Google Colab option is recommended for users without access to a powerful local machine.

# 🔐 Google Maps API Key (Optional)

To enable satellite imagery in the map interface of the Streamlit app, create a `.env` file in the project directory with your Google Maps API key:

```
GOOGLE_MAPS_API_KEY=your_api_key_here
```

- Obtain a Google Maps API key from the Google Cloud Console.
- Without the API key, the map interface in `app.py` will not display satellite imagery, but other functionalities will still work.
