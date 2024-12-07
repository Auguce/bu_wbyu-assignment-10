# Assignment 10: Image Search
  
This project implements a simplified version of Google Image Search, fulfilling the requirements for Assignment 10. Users can perform image searches via text queries, image uploads, or hybrid queries combining text and image input. Additionally, users can utilize PCA embeddings with configurable k values to customize their search results.
  
## **Features**  
- **Text Query**: Input a text query and view the top 5 most relevant images with similarity scores.  
- **Image Query**: Upload an image, optionally use PCA embeddings with a user-defined k value, and view the top 5 most relevant images with similarity scores.  
- **Hybrid Query**: Combine a text query and an image upload with a configurable weight between 0.0 and 1.0 to control the importance of the text query relative to the image query.  

## **Required Files for Application to Run**  
To ensure the application runs as expected, the following files and directories are required but not included in this repository due to their size or nature:   

- **coco_images_resized/**: A directory containing resized COCO dataset images. You can generate this directory by resizing images as described in the assignment or lab instructions.  
- **image_embeddings.pickle**: A precomputed pickle file containing the image embeddings for the COCO dataset. This file is used for efficient image similarity searches.  
- **uploaded/**: A directory where user-uploaded images will be temporarily stored during runtime. This directory will be created automatically if it does not exist.  
- **clip_model_cache/**: A cache directory for storing the CLIP model downloaded by Hugging Face Transformers. This directory will be created automatically during runtime.  
  
## **Project Setup**  
### **Dependencies**  
This project requires Python 3.9 or later. Install dependencies using:  
```bash  
make setup  
