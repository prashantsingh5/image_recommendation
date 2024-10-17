# House Room Image Recommendation System

This project implements an image recommendation system for house rooms such as bedrooms, kitchens, bathrooms, living rooms, and dining rooms. It extracts visual features like embeddings, dominant colors, and contrast from images and uses them to recommend similar images. The system is built into a single script for ease of use.

## Project Structure

The project contains a single Python script that handles everything from feature extraction to image recommendation and hosting a Flask web API.

        House_Room_Dataset/
        
                ├── Bedroom/
                        
                ├── Bathroom/
                        
                ├── Dining/
                        
                ├── Kitchen/
                        
                └── Livingroom/


## Features

- **ResNet50 Embeddings**: 
  Extracts image embeddings using the pre-trained ResNet50 model.
  
- **Dominant Color Extraction**: 
  Identifies the dominant color in each image using KMeans clustering.
  
- **Contrast Calculation**: 
  Measures image contrast using pixel intensity standard deviation.
  
- **Image Recommendation**: 
  Recommends images based on cosine similarity of embeddings, color similarity, and contrast.

- **Flask API Integration**: 
  Provides a web API to accept user-uploaded images and return recommendations.

## Installation and Setup

To install the required libraries, use the following:

1.  Clone the repository:
   
```bash
git clone https://github.com/your-repository/image-recommendation.git

2.  Install the required dependencies:

```bash
pip install -r requirements.txt

3.  Train and extract image features by running the dataframe_training.py script:

```bash
python dataframe_training.py

after running the above command change the path of dataframe in in flask_integration.py

4.  To run the Flask app:

```bash
python flask_integration.py

