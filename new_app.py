import streamlit as st
import os 
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import tensorflow
import pandas as pd
import tensorflow 
from plotly.offline import init_notebook_mode
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201, Xception

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames .pkl','rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.set_page_config(page_title="Fashion Recommender System", page_icon="👗")
st.title('Fashion Recommender System')

# Load product links from CSV
@st.cache_data()
def load_product_links():
    return pd.read_csv(r'images.csv')

product_links = load_product_links()

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload -> save
uploaded_file = st.file_uploader("Upload an image of a Product item", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(r'uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Feature extraction
    features = feature_extraction(file_path, model)
    
    # Recommendation
    indices = recommend(features, feature_list)
    
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Display recommended images and product links
    st.header('Recommended Products')
    
    # Image Carousel
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for i, col in enumerate(cols):
        recommended_filename = filenames[indices[0][i]]
        recommended_image_path = os.path.join(r'images', os.path.basename(recommended_filename))
        recommended_image = Image.open(recommended_image_path)

        # Apply hover effect
        col.image(recommended_image, caption=f'Recommended {i+1}', use_column_width=True, 
                  output_format='JPEG')
        
        # Extract product ID from recommended filename
        product_id = os.path.splitext(os.path.basename(recommended_filename))[0]

        # Find corresponding filename in CSV file
        matching_filename = None
        for filename in product_links['filename']:
            if product_id in filename:
                matching_filename = filename
                break

        # If matching filename is found, retrieve the product link
        if matching_filename is not None:
            product_link_row = product_links.loc[product_links['filename'] == matching_filename]
            product_link = product_link_row['link'].iloc[0]
            col.write(f"Product Link: {product_link}")
        else:
            col.write("Product Link Not Available")

# Custom CSS for enhanced styling
st.markdown('''
    <style>
        .stImage:hover {
            opacity: 0.8;
            transition: opacity '0.5s' ease-in-out;
        }
        .stMarkdown:hover {
            opacity: 0.8;
            transition: opacity '0.5s' ease-in-out;
        }
        .stMarkdown a {
            text-decoration: none;
            color: #4c87cd;
            font-weight: bold;
        }
        
    </style>
''',unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("Built with Streamlit by Kathir & Siva")

