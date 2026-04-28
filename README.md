# Potato Disease Classifier

Early and accurate detection of potato leaf diseases can prevent massive crop losses.
This project uses deep learning to classify potato plant diseases from leaf images —
helping farmers identify problems before they spread.

---

### About the Project

Potato crops are highly vulnerable to diseases like Early Blight and Late Blight, which
can destroy entire harvests if not detected early. Traditional detection relies on manual
inspection by experts — slow, expensive, and not scalable for small farmers.

This project builds an automated image classification system using Transfer Learning with
EfficientNetB7, pretrained on ImageNet and fine-tuned on the PlantVillage potato dataset.
The model takes a leaf image as input and predicts whether the plant is healthy or infected,
along with the disease type. A simple Flask web app allows users to upload images and get
predictions instantly.

---

### Dataset

**PlantVillage Potato Disease Dataset** — [Download from Kaggle](https://www.kaggle.com/datasets/assassinldrago/potato-plant-disease)

The dataset contains ~2,152 images of potato leaves across 3 classes:

| Class | Description |
|---|---|
| **Early Blight** | Caused by *Alternaria solani* fungus — brown spots with yellow rings |
| **Late Blight** | Caused by *Phytophthora infestans* — dark lesions, spreads rapidly |
| **Healthy** | No visible disease symptoms |

Place the downloaded folder as it is


---

### How It Works
Input Image  →  EfficientNetB7 (pretrained on ImageNet)
→  Data Augmentation (flip, rotate, zoom)
→  Dense layers (512 → 256 → 3)
→  Softmax → Predicted Class

- **Transfer Learning** — EfficientNetB7 base frozen, only top layers trained
- **Data Augmentation** — random flip, rotation, zoom to prevent overfitting
- **Early Stopping** — training stops automatically when validation loss plateaus
- **Model Checkpointing** — best weights saved automatically during training

---

### Stack

| | |
|---|---|
| **Model** | EfficientNetB7 · Transfer Learning |
| **Framework** | TensorFlow · Keras |
| **Data** | Pandas · NumPy · OpenCV · ImageDataGenerator |
| **Visualization** | Matplotlib · Seaborn |
| **Web App** | Flask · HTML |

---
### Project Structure

```
potato-disease-classifier/
│
├── potato_lab.ipynb
├── app.py
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
# Open potato_lab.ipynb in Jupyter or Google Colab and run all cells

# Run the web app
python app.py
```

Then open `http://localhost:5000` in your browser and upload a potato leaf image.

---

### Results

| Metric | Value |
|---|---|
| **Base Model** | EfficientNetB7 |
| **Training Images** | ~1,936 |
| **Test Images** | ~216 |
| **Test Accuracy** | 94% |
