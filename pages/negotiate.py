# import openai
from dotenv import load_dotenv
import streamlit as st
import os
import logging
import json
import re
import pickle
# import openai
import pandas as pd
from langchain_openai.chat_models.base import ChatOpenAI
from styles import css, bot_template, user_template
from langchain_core.prompts.prompt import PromptTemplate 
import sys

# Load environment variables
load_dotenv()
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,  # Adjust the level as needed (e.g., DEBUG, WARNING)
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Direct logs to standard output (Terminal)
)

class AIChatAssistant:
    def __init__(self):
        self.load_state()
        
        self.chat = ChatOpenAI(
            model="gpt-4-turbo",  # You can use "gpt-4" if available
            temperature=0.8,
            frequency_penalty=0.2
        )
        # Initialize session state
        if 'expected_price' in st.session_state:
            self.expected_price = st.session_state.get('expected_price', None)
        if 'order_placed' not in st.session_state:
            st.session_state['order_placed'] = False
        if 'show_negotiation' not in st.session_state:
            st.session_state['show_negotiation'] = False
        if 'order_accepted' not in st.session_state:
            st.session_state['order_accepted'] = False
        if 'verify_started' not in st.session_state:
            st.session_state['verify_started'] = False
            
        
        
            
            
        logging.info("Loading models and scalers for predictions.")
        # Load models and scalers for predictions
        try:
            with open('artifacts/model.pkl', 'rb') as model_file:
                self.loaded_model = pickle.load(model_file)
            with open('artifacts/scaler.pkl', 'rb') as scaler_file:
                self.loaded_scaler = pickle.load(scaler_file)
            with open('artifacts/pca.pkl', 'rb') as pca_file:
                self.loaded_pca = pickle.load(pca_file)
            with open('artifacts/selector.pkl', 'rb') as selector_file:
                self.loaded_selector = pickle.load(selector_file)
            with open('artifacts/category_mean.pkl', 'rb') as file:
                self.loaded_category_mean = pickle.load(file)
            with open('artifacts/imputer.pkl', 'rb') as imputer_file:
                self.loaded_imputer = pickle.load(imputer_file)
            logging.info("Models and scalers loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            st.error("Error loading models. Please check the model files and try again.")
            st.stop()
        # Load data
        self.product_data = self.load_negotiate_data()
        predicted_discount_percentage=self.predict_discount()
        customer_discount_perscentage=20
        product_discount = self.product_data['actual_price'] - (self.product_data['actual_price'] * predicted_discount_percentage) / 100
        self.product_discount = float(product_discount)
        customer_discount = self.product_data['actual_price'] - (self.product_data['actual_price'] * customer_discount_perscentage) / 100
        self.customer_discount=((self.product_discount*1)+(customer_discount*2))/3
        self.customer_discount=float(self.customer_discount)
        self.customer_data = pd.read_csv("customer.csv")
        
        
    def preprocess_percentage(self, value):
        """Convert percentage string to numeric value."""
        if isinstance(value, str) and '%' in value:
            return float(value.replace('%', '').strip()) / 100.0
        return float(value)

        
    def predict_discount(self):
        try:
            logging.info("Predicting discount for the product.")
            # Ensure the category is treated as a single string and not a list
            category_value = self.product_data['category']
            if isinstance(category_value, list):
                category_value = category_value[0]  # Take the first item if it's a list

            # Convert percentage values
            self.product_data['discount_percentage'] = self.preprocess_percentage(self.product_data['discount_percentage'])

            # Get the encoded category or default to the mean encoding
            new_category_encoded = self.loaded_category_mean.get(category_value, self.loaded_category_mean.mean())
            self.product_data['category_encoded'] = new_category_encoded

            # Create DataFrame with correct columns
            new_data_combined = pd.DataFrame({
                'actual_price': [self.product_data['actual_price']],
                'discount_percentage': [self.product_data['discount_percentage']],
                'rating': [self.product_data['rating']],
                'rating_count': [self.product_data['rating_count']],
                'category_encoded': [new_category_encoded],
            })

            # Ensure DataFrame columns are in the same order as during training
            expected_columns = ['actual_price', 'discount_percentage', 'rating', 'rating_count', 'category_encoded']
            new_data_combined = new_data_combined[expected_columns]

            # Apply preprocessing steps
            new_data_imputed = self.loaded_imputer.transform(new_data_combined)
            new_data_scaled = self.loaded_scaler.transform(new_data_imputed)
            new_data_pca = self.loaded_pca.transform(new_data_scaled)
            new_data_selected = self.loaded_selector.transform(new_data_pca)

            # Predict discount
            new_prediction = self.loaded_model.predict(new_data_selected)
            logging.info(f"Predicted discount: {new_prediction[0]}")
            return new_prediction[0]
        except Exception as e:
            logging.error(f"Error predicting discount: {str(e)}")
            st.error(f"Error predicting discount: {str(e)}")
            st.stop()    
    def load_negotiate_data(self):
        try:
            with open('negotiate.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                required_keys = ['product_id', 'product_name', 'category', 'actual_price', 'discounted_price', 
                                    'discount_percentage', 'rating', 'rating_count', 'about_product', 'img_link', 'username','password']
                if not data or not all(k in data for k in required_keys):
                    logging.error("Incomplete product data. Please log in and choose a product.")
                    st.error("Incomplete product data. Please log in and choose a product for order or negotiation.")
                    st.stop()
                logging.info("Negotiation data loaded successfully.")
                return data
        except FileNotFoundError:
            logging.error("Product data file not found.")
            st.error("No product data file found. Please log in and choose a product for order or negotiation.")
            st.stop()
        except json.JSONDecodeError:
            logging.error("Error decoding product data file.")
            st.error("Error decoding product data file. Please check the file and try again.")
            st.stop()
    def load_state(self):
        if 'history' not in st.session_state:
            st.session_state.history = []
            
    def classify_intent_with_llm(self, user_reply):
        prompt = f"""
        You are an assistant for a negotiation chatbot. Determine the intent of the user's reply based on the following categories:
        
        1. Negotiating Price: User is asking about a discount or trying to negotiate the price.
        2. Verifying Registration: User is asking about account verification or registration.
        3. Registered Customer: User is telling that registered person.
        4. Not Registered: User is telling that not registered person.
        5. Placing Order: User wants to place an order or proceed with a purchase.
        6. General Inquiry: User is asking about product details or has a general question.
        
        User Reply: "{user_reply}"
        
        Please provide the intent as one of the following:
        - Negotiating Price
        - Verifying Registration
        - Registered Customer
        - Not Registered
        - Placing Order
        - General Inquiry
        """

        # Call the LLM to classify the intent
        try:
            
            response = self.chat.invoke(prompt)
            intent = response.content.strip()
            print(intent)
         
            logging.info(f"Classified intent: {intent}")
            
            # Ensure the intent is one of the expected values
            valid_intents = [
                "Negotiating Price",
                "Verifying Registration",
                "Registered Customer",
                "Not Registered",
                "Placing Order",
                "General Inquiry"
            ]
            
            if intent not in valid_intents:
                logging.warning(f"Unrecognized intent: {intent}")
                intent = "General Inquiry"  # Default to a safe value

            return intent
        except Exception as e:
            logging.error(f"Error classifying intent: {e}")
            st.error(f"Error classifying intent: {str(e)}")
            st.stop()
            
    

    def select_template_llm(self, user_reply):
        # Use LLM to classify the intent from the user reply
        self.intent = self.classify_intent_with_llm(user_reply)
        
        # Use expected_price from session state
        self.expected_price = st.session_state.get('expected_price', None)
        
        logging.info(f"User intent: {self.intent}")
        logging.info(f"expected_price: {self.expected_price}")
        logging.info(f"product_discount: {self.product_discount}")
        logging.info(f"expected_price type: {type(self.expected_price)}")
        logging.info(f"product_discount type: {type(self.product_discount)}")
        logging.info(f"customer_discount: {self.customer_discount}")

        
        if self.intent == "Negotiating Price" and st.session_state['order_accepted'] == False and st.session_state['verify_started'] == False:
            if self.expected_price is not None and self.expected_price < self.product_discount:
                template = "give a alternative conversation reply only  for  Ask if he is a registered customer."
            else:
                template = "give a alternative conversation reply only in single sentence  for     We ve reviewed your offer and we're glad to work with you! Could you please confirm that you're comfortable with the price of {expected_price} Once you're happy with the details, kindly confirm your order so we can proceed"
                st.session_state['order_accepted'] = True
        elif self.intent == "Verifying Registration":
            template = "give a alternative conversation reply only  for  Ask them to verify their account using a password to access the customer discount."
            st.session_state['verify_started'] = True
            
        elif self.intent == "Registered Customer":
            result=self.verify_account()
         
            if result == True:
                
                if self.expected_price is not None and self.customer_discount<self.expected_price < self.product_discount:
                    template = "give a alternative conversation reply only in single sentence  for     We ve reviewed your offer and we're glad to work with you! Could you please confirm that you're comfortable with the price of {expected_price} Once you're happy with the details, kindly confirm your order so we can proceed"
                    st.session_state['order_accepted'] = True
                else:
                    template = "give a alternative conversation reply only  for     based on your transaction history, you will get the product in {customer_discount}"
        
                
               
            else:
                template = "give a alternative conversation reply only  for   you are not a registered customer. Your maximum discount is {product_discount}"
               
                
            
        elif self.intent =="Not Registered":
            
            template="give a alternative conversation reply only  for   your maximum discount is {product_discount}"
        elif self.intent == "Placing Order":
            template = "Order placed successfully."
            st.session_state['order_accepted'] = True
            st.success("Great! Your order has been placed.")
            logging.info("Order placed.")
            
        logging.info(f"template: {template}")
        return template  # Return only the template string
    
    def verify_account(self):
        user_row = self.customer_data[
            (self.customer_data['username'] == self.product_data['username']) & 
            (self.customer_data['password'] == self.product_data['password']) &
            (self.customer_data['Subscription Status'] == 'Yes')
        ]
        # Return True if the user is found and has a subscription, otherwise False
        if not user_row.empty:
            return True
        return True

    def generate_response(self, prompt):
        
       
        response = self.chat.invoke(prompt).content
        
        return response

    def display_conversation(self):
        for message in st.session_state.history:
            if message["role"] == "user":
                st.write(f"You: {message['content']}")
            else:
                st.write(f"Assistant: {message['content']}")
                
            
    def display_product_details(self):
        st.subheader("Product Details")
        st.write(f"**Product ID:** {self.product_data['product_id']}")
        st.write(f"**Product Name:** {self.product_data['product_name']}")
        st.write(f"**Category:** {self.product_data['category']}")
        st.write(f"**Price:** {self.product_data['actual_price']}")
        st.write(f"**Rating:** {self.product_data['rating']} ({self.product_data['rating_count']} ratings)")
        st.image(self.product_data['img_link'], width=200)
        st.write("**About the Product:**")
        st.write(self.product_data['about_product'].replace("|", "\n- "))
                
    def display_ui(self):
        st.set_page_config(page_title="Nego Savy Assistant", layout="wide")
        st.write(css, unsafe_allow_html=True)
        st.header("Hey, Let's Make a Deal")

        self.display_product_details()

        st.markdown("---")  # Adds a horizontal line separator

        # Manage button visibility based on state
        if not st.session_state['order_placed']:
            st.subheader("Ready to Order?")
            st.write("If you're happy with the price and ready to place your order, click the button below:")

            order_col1, order_col2 = st.columns([1, 1])

            with order_col1:
                if not st.session_state['show_negotiation']:
                    if st.button("Yes, I'm ready to order!"):
                        st.session_state['order_placed'] = True
                        st.success("Great! Your order has been placed.")
                        logging.info("Order placed.")

            with order_col2:
                if not st.session_state['show_negotiation']:
                    if st.button("Negotiate Price"):
                        if st.session_state['order_placed']:
                            st.session_state['order_placed'] = False
                        st.session_state['show_negotiation'] = True

        if st.session_state['show_negotiation']:
            chat_assistant.run()
            

    def run(self):
        st.title("Negotiation Assistant")
        # st.write("Tell me about your expected Price!")
                
        # Input field for user message
        user_input = st.text_input("You: ", "Tell me about your expected Price")

        if st.button("Send"):
            if user_input:
                
                match = re.search(r'\d+(\.\d+)?', user_input)
                if match:
                    self.expected_price = float(match.group())
                st.session_state['expected_price'] = self.expected_price
                logging.info(f"Expected price extracted: {self.expected_price}")
                
                # Detect the user's intent
                template = self.select_template_llm(user_input)
                follow_up_prompt = PromptTemplate(
                input_variables=["expected_price", "product_discount","customer_discount"],
                template=template  # Use the template string directly
            )
                formatted_follow_up_prompt = follow_up_prompt.format(expected_price=self.expected_price, product_discount=self.product_discount,customer_discount=self.customer_discount)
                # Generate a response based on the detected intent
                response = self.generate_response(formatted_follow_up_prompt)

                # Append user input and assistant response to chat history
                st.session_state.history.append({"role": "user", "content": user_input})
                st.session_state.history.append({"role": "assistant", "content": response})

                # Display the conversation history
                self.display_conversation()

if __name__ == "__main__":
    chat_assistant = AIChatAssistant()
    chat_assistant.display_ui()
    
