import streamlit as st
import json
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GENERAL SETTINGS
PAGE_TITLE = "Nego Savy Assistant"
PAGE_ICON = ":wave:"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# Initialize Sentence-Transformer model
@st.cache_resource
def load_model():
    logging.info("Loading Sentence-Transformer model...")
    return SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient embedding model

# Initialize Summarization model
@st.cache_resource
def load_summarizer():
    logging.info("Loading summarization model...")
    return pipeline("summarization", model="facebook/bart-large-cnn")

class EcommerceChatAssistant:
    def __init__(self, data_path):
        logging.info("Initializing EcommerceChatAssistant...")
        self.model = load_model()
        self.summarizer = load_summarizer()
        self.data = self.load_data(data_path)
        self.embeddings_matrix = self.generate_embeddings(self.data)
        self.faiss_index = self.build_faiss_index(self.embeddings_matrix)

    # Load the JSON dataset
    def load_data(self, data_path):
        @st.cache_data
        def _load_data(path):
            logging.info(f"Loading data from {path}...")
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data

        return _load_data(data_path)

    # Generate embeddings for all products using Sentence-Transformer
    def generate_embeddings(self, data, batch_size=32):
        logging.info("Generating embeddings for products...")
        texts = [product['product_name'] for product in data]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_embeddings = self.model.encode(texts[i:i+batch_size], batch_size=batch_size, convert_to_tensor=False)
            embeddings.extend(batch_embeddings)
        for product, embedding in zip(data, embeddings):
            product['embedding'] = embedding
        return np.array(embeddings)

    # Build the FAISS index for approximate nearest neighbors search
    def build_faiss_index(self, embeddings):
        logging.info("Building FAISS index...")
        dim = embeddings.shape[1]  # Embedding dimension
        index = faiss.IndexFlatL2(dim)  # Using L2 (Euclidean) distance, which can be adapted for cosine similarity
        index.add(embeddings)
        return index

    # Get recommendations using FAISS for fast search
    def get_recommendations(self, product_id, top_n=5):
        logging.info(f"Getting recommendations for product ID: {product_id}...")
        product = next(item for item in self.data if item['product_id'] == product_id)
        product_embedding = np.array(product['embedding']).reshape(1, -1)

        # Perform FAISS search
        distances, indices = self.faiss_index.search(product_embedding, top_n+1)  # Exclude self
        recommendations = [self.data[i] for i in indices.flatten()[1:]]  # Skip the first result (self)
        
        # Save search and recommendations to recommend.json
        self.save_recommendations(product['product_name'], recommendations)
        
        return recommendations

    # Save the searched product name and recommendations to a JSON file
    def save_recommendations(self, searched_product_name, recommendations):
        logging.info(f"Saving recommendations for {searched_product_name}...")
        st.session_state.username 
        logging.info(f"username is {st.session_state.username }")
        logging.info(f"password is {st.session_state.password }")
        recommend_data = {
            'searched_product_name': searched_product_name,
            'recommendations': [{
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'actual_price': product['actual_price'],
                'category': product['category'],
                'rating': product['rating'],
                'rating_count': product['rating_count'],
                'img_link': product.get('img_link', ''),  # Default to empty string if 'img_link' is not present
                'about_product': product.get('about_product', '')  # Include 'about_product' directly
            } for product in recommendations]
        }
        with open('recommend.json', 'w', encoding='utf-8') as file:
            json.dump(recommend_data, file, ensure_ascii=False, indent=4)

    # Save the selected product for negotiation
    def save_negotiation(self, product):
        logging.info(f"Saving negotiation data for product ID: {product['product_id']}...")
        negotiation_data = {
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'discounted_price':product['discounted_price'],
            'actual_price': product['actual_price'],
            'product_link':product['product_link'],
            'discount_percentage': product['discount_percentage'],
            'weighted_discount': product['weighted_discount'],
            'category': product['category'],
            'username': st.session_state.username,
            'password': st.session_state.password,
            'rating': product['rating'],
            'rating_count': product['rating_count'],
            'img_link': product.get('img_link', ''),  # Default to empty string if 'img_link' is not present
            'about_product': product.get('about_product', '')  # Include 'about_product' directly
        }
        with open('negotiate.json', 'w', encoding='utf-8') as file:
            json.dump(negotiation_data, file, ensure_ascii=False, indent=4)

    # Summarize the 'about_product' field for display
    def summarize_about_product(self, text, max_length=50):
        logging.info("Summarizing product description...")
        summary = self.summarizer(text, max_length=max_length, min_length=50, do_sample=False)
        return summary[0]['summary_text']

    # Display the recommendations on the Streamlit page
    def display_recommendations(self, recommendations):
        st.header("Recommendations")
        
        for product in recommendations:
            st.subheader(f"Product ID: {product['product_id']}")
            st.write(f"Product Name: {product['product_name']}")
            st.write(f"**Actual Price: {str(product['actual_price']).upper()}**")
            st.write(f"Category: {product['category']}")
            st.write(f"Rating: {product['rating']} ({product['rating_count']})")
            
            # Display the summarized 'about_product'
            summarized_about_product = self.summarize_about_product(product.get('about_product', 'No description available.'))
            st.write(f"**About this product:** {summarized_about_product}")

            # Display the image
            st.image(product['img_link'], use_column_width=True)

            # Adding "Negotiate" button
            if st.button(f"Proceed with Order/Negotiation for Product ID: {product['product_id']}", key=f"negotiate_{product['product_id']}"):
                self.save_negotiation(product)
                st.success(f"Negotiation/Order in Progress for Product ID: {product['product_id']}. Please proceed to the Negotiation Page for further details.")

    # Main run loop of the app
    def run(self):
        logging.info("Running the EcommerceChatAssistant application...")
        st.title('Ecommerce Recommendation Assistant')

        # Display category selection
        categories = sorted(set(product['category'] for product in self.data))
        selected_category = st.selectbox('Select a category:', categories)

        # Filter data based on the selected category
        filtered_data = [product for product in self.data if product['category'] == selected_category]

        # Display product name selection with "Select..." as the first option
        product_names = ["Select..."] + sorted(product['product_name'] for product in filtered_data)
        selected_product_name = st.selectbox('Select a product:', product_names)

        # Check if a valid product name is selected (not "Select...")
        if selected_product_name != "Select...":
            # Attempt to find the selected product in the filtered_data
            selected_product = next((product for product in filtered_data if product['product_name'] == selected_product_name), None)

            # Only proceed if the product was found
            if selected_product:
                selected_product_id = selected_product['product_id']

                # Get recommendations based on the selected product
                recommendations = self.get_recommendations(selected_product_id)

                # Display recommendations
                self.display_recommendations(recommendations)
            else:
                st.warning("Selected product not found. Please select a valid product.")

# Instantiate and run the app
if __name__ == '__main__':
    assistant = EcommerceChatAssistant(data_path='products.json')
    assistant.run()
