# Multimodal AI Visual Search Engine

## 📌 Project Overview
This project is an advanced **Content-Based Visual Recommendation System**. Instead of relying on traditional text tags or user click-history, this system leverages **Computer Vision** to "read" the visual features of a product (shape, pattern, color) and instantly recommends similar items. 

Built with scalability in mind, it utilizes Transfer Learning and Vector Similarity Search to process and serve recommendations from thousands of fashion products through an interactive web interface.

## ⚙️ Core Architecture & Tech Stack

**Data Science & Engineering Pipeline:**
* **Exploratory Data Analysis (EDA) & Data Wrangling:** Handled missing/corrupted image paths, parsed messy CSV metadata, and analyzed class distributions to curate a strategic subset for efficient local prototyping.
* **Feature Normalization & Scaling:** Applied L2 Normalization (`sklearn.preprocessing.normalize`) to both visual and textual feature spaces before fusion, ensuring mathematical equilibrium and preventing one modality from dominating the Cosine Similarity calculation.
* **Data Pipeline:** Optimized image ingestion and preprocessing using `tf.data.Dataset` (batching, prefetching) to handle thousands of high-resolution images efficiently without memory overflow.

**Machine Learning & Multimodal AI:**
* **True Multimodal Fusion:** Combined visual features and semantic text features to create a robust, context-aware recommendation engine.
* **Computer Vision (Image Embeddings):** Utilized **ResNet50** (Pre-trained CNN) with Global Average Pooling to extract 2048-dimensional latent space vectors from raw product images.
* **Natural Language Processing (Text Embeddings):** Applied **TF-IDF** vectorization on product metadata (`productDisplayName` and `articleType`) to capture semantic context and product nuances.
* **Vector Similarity Search:** Concatenated the normalized visual and textual vectors with a customized 70/30 weight distribution, using **Cosine Similarity** to retrieve identical items in milliseconds.

**Backend & Deployment:**
* **Backend Framework:** Built a robust RESTful architecture using **Flask (Python)** to bridge the machine learning model with the frontend.
* **Frontend UI:** Developed a responsive and interactive user interface using **HTML, CSS, and Bootstrap**, featuring dynamic routing and an AI processing dashboard for seamless visual exploration.

## 📂 Project Structure
* `01_visual_feature_extraction.ipynb`: Data engineering, strategic filtering, and ResNet50 embedding pipeline.
* `app.py`: The Flask backend logic handling routing and real-time similarity computations.
* `templates/index.html`: The interactive frontend catalog and recommendation display.
* `image_embeddings_5k.npy`: The stored 2048-D vector matrix of the product catalog.

## 🚀 How to Run Locally
1. Clone this repository.
2. Download the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) and extract the `images/` folder into the root directory.
3. Install requirements: `pip install tensorflow flask scikit-learn pandas numpy opencv-python`.
4. Run the backend: `python app.py`.
5. Open `http://127.0.0.1:5000` in your browser to start exploring!
