from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from PIL import Image
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

COCO_IMAGES_DIR = 'coco_images_resized'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

clip_model = None
clip_processor = None
embeddings = None
image_names = None


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def encode_text_query(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features.numpy().flatten()


def encode_image_query(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.numpy().flatten()


def combined_embedding(text_embedding, image_embedding, weight):
    return (
        weight * text_embedding +
        (1 - weight) * image_embedding
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/coco_images_resized/<filename>')
def serve_coco_image(filename):
    return send_from_directory(COCO_IMAGES_DIR, filename)


@app.route('/search_images', methods=['POST'])
def search_images():
    raw_use_pca_value = request.form.get('use_pca', 'false')
    query_type = request.form.get('query_type', 'image').strip()
    text_query = request.form.get('text_query', '').strip()
    weight = float(request.form.get('weight', 0.5))
    use_pca = (raw_use_pca_value == 'true')
    pca_k = int(request.form.get('pca_k', '50'))  # default is 50

    print("===== Debugging Parameters =====")
    print(f"query_type: {query_type}")
    print(f"text_query: '{text_query}'")
    print(f"weight: {weight}")
    print(f"raw_use_pca_value: '{raw_use_pca_value}'")
    print(f"use_pca (interpreted): {use_pca}")
    print(f"pca_k: {pca_k}")
    print("===============================")

    file = request.files.get('image_query', None)

    text_embedding = None
    image_embedding = None
    query_embedding = None

    if query_type in ['text', 'hybrid'] and text_query:
        print("Fetching text embedding...")
        text_embedding = encode_text_query(text_query)

    if (
        query_type in ['image', 'hybrid'] and
        file and
        allowed_file(file.filename)
    ):
        print("Fetching image embedding...")
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploaded', filename)
        if not os.path.exists('uploaded'):
            os.makedirs('uploaded')
        file.save(filepath)
        image_embedding = encode_image_query(filepath)

    if query_type == 'text':
        use_pca = False
        print("Text-only query. Forced use_pca = False.")
        if text_embedding is not None:
            query_embedding = text_embedding
        else:
            print("No valid text query provided.")
            return jsonify({"error": "No valid text query provided."}), 400

    elif query_type == 'image':
        if image_embedding is not None:
            if use_pca:
                print("Image-only query with PCA.")
                local_pca = PCA(n_components=pca_k)
                local_pca.fit(embeddings)
                image_pca = local_pca.transform(
                    image_embedding.reshape(1, -1)
                )
                reduced_embeddings = local_pca.transform(embeddings)
                image_pca = normalize(image_pca, axis=1)
                reduced_embeddings = normalize(reduced_embeddings, axis=1)
                sim_scores = cosine_similarity(
                    image_pca,
                    reduced_embeddings
                ).flatten()
                sim_scores = (sim_scores + 1) / 2
                sim_scores = np.clip(sim_scores, 0, 1)

                top_k = 5
                nearest_indices = np.argsort(sim_scores)[::-1][:top_k]
                results = []
                for i in nearest_indices:
                    img_name = image_names[i]
                    results.append({
                        "image_name": img_name,
                        "similarity": float(sim_scores[i])
                    })
                print("Returning PCA-based image search results.")
                return jsonify({"results": results})
            else:
                print("Image-only query without PCA.")
                query_embedding = image_embedding
        else:
            print("No valid image query provided.")
            return jsonify({"error": "No valid image query provided."}), 400

    elif query_type == 'hybrid':
        if text_embedding is None or image_embedding is None:
            print("No valid hybrid query provided.")
            return jsonify({"error": "No valid hybrid query provided."}), 400

        if use_pca:
            print("Hybrid query with PCA.")
            local_pca = PCA(n_components=pca_k)
            local_pca.fit(embeddings)
            image_pca = local_pca.transform(
                image_embedding.reshape(1, -1)
            )
            image_back = local_pca.inverse_transform(image_pca)
            query_embedding = combined_embedding(
                text_embedding,
                image_back.flatten(),
                weight
            )
        else:
            print("Hybrid query without PCA.")
            query_embedding = combined_embedding(
                text_embedding,
                image_embedding,
                weight
            )

    else:
        print("Invalid query type.")
        return jsonify({"error": "Invalid query type"}), 400

    query_embedding = normalize(
        query_embedding.reshape(1, -1),
        axis=1
    )
    normalized_embeddings = normalize(embeddings, axis=1)
    sim_scores = cosine_similarity(
        query_embedding,
        normalized_embeddings
    ).flatten()
    sim_scores = (sim_scores + 1) / 2
    sim_scores = np.clip(sim_scores, 0, 1)

    top_k = 5
    nearest_indices = np.argsort(sim_scores)[::-1][:top_k]
    results = []
    for i in nearest_indices:
        img_name = image_names[i]
        results.append({
            "image_name": img_name,
            "similarity": float(sim_scores[i])
        })

    print("Returning final results.")
    return jsonify({"results": results})


def initialize_global_resources():
    global clip_model, clip_processor, embeddings, image_names

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with open('image_embeddings.pickle', 'rb') as f:
        image_embeddings = pickle.load(f)

    print("=== Debugging image_embeddings ===")
    if isinstance(image_embeddings, pd.DataFrame):
        print(image_embeddings.head())
        image_names = image_embeddings['file_name'].tolist()
        embeddings_list = image_embeddings['embedding'].to_list()
        embeddings = np.array(embeddings_list, dtype=float)
    else:
        raise ValueError("Check your pickle file.")

    print("Initialization done.")


if __name__ == '__main__':
    initialize_global_resources()
    app.run(debug=True)
