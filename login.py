import streamlit as st
import pandas as pd
import recommendation  

st.title("Welcome to the AI Ecommerce chat Assistant")
st.write("After selecting a product, please proceed to the 'Negotiate' page via the sidebar to continue.")

# Class to handle the login process 
class LoginApp:
    def __init__(self):
        self.customer_data = self.load_customer_data('customer.csv')
        self.initialize_session_state()

    @staticmethod
    @st.cache_data
    def load_customer_data(file_path):
        return pd.read_csv(file_path, encoding='ISO-8859-1')

    def initialize_session_state(self):
        if 'login_status' not in st.session_state:
            st.session_state.login_status = False

        if 'username' not in st.session_state:
            st.session_state.username = ""

        if 'password' not in st.session_state:
            st.session_state.password = ""

    def authenticate(self, username, password):
        user_record = self.customer_data[
            (self.customer_data['username'] == username) & 
            (self.customer_data['password'] == password)
        ]
        return not user_record.empty

    def show_login_page(self):
        st.title('Login Page')

        with st.form(key='login_form'):
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            login_button = st.form_submit_button('Login')

            if login_button:
                if self.authenticate(username, password):
                    # Save the username and password in session state
                    st.session_state.username = username
                    st.session_state.password = password

                    # Set login status to True
                    st.session_state.login_status = True

                    # Display a success message
                    st.success(f'Welcome, {username}!')

                else:
                    st.error('Invalid username or password')

    def show_chatbot(self):
        # If user is already logged in, show the chatbot
        st.success(f'Welcome back, {st.session_state.username}!')
        chatbot = recommendation.EcommerceChatAssistant(data_path='products.json')  # Create an instance of the Chatbot class from rec2
        chatbot.run()  # Run the chatbot

    def run(self):
        if st.session_state.login_status:
            self.show_chatbot()
        else:
            self.show_login_page()

# Create an instance of the LoginApp class and run it
if __name__ == "__main__":
    app = LoginApp()
    app.run()
