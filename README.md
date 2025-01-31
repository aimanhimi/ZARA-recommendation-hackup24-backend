# Product Recommendation Backend
This repository contains the backend of our product recommendation system, built with FastAPI and powered by CNN for image and text-based similarity retrieval.

![image](https://github.com/user-attachments/assets/c40b4595-4b47-4308-8c41-b55eb52d220a)
![image](https://github.com/user-attachments/assets/89613515-7508-4dfb-a3cf-969a87682aaa)

## Overview
Our system allows users to select a product and receive recommendations for similar products using similarity search.

## Tech Stack
- FastAPI (Backend framework)
- TensorFlow & Keras (CNN-based embeddings feature extraction with ResNet)
- Pinecone (Vector database for efficient similarity search)
- Scikit-learn (TF-IDF vectorization for text similarity)
- Pandas & NumPy (Data processing)

## Image-Based Recommendations
- A ResNet50 model extracts visual features from product images.
These features are stored and used to find similar products based on cosine similarity.
### Text-Based Recommendations
- A TF-IDF model converts product descriptions into numerical vectors.
These vectors are indexed in Pinecone, allowing fast similarity search based on captions.

## Next Steps
- Improve the recommendation model
- Optimize query speed
- Build a real-time live camera recommendation system
