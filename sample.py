# sample.py
from flask import *
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import re
from apps import Chatbot # Import your QASystem class here
import os
from firebase_connection import read_user_by_credentials,create_user
import logging
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
app.secret_key = '43b7129bb683eed1038c16f068c551c'
qa_system = Chatbot()  # Initialize your QA system

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'py', 'js', 'java', 'cpp', 'cs', 'rb', 'go', 'png', 'jpg', 'jpeg'}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def is_password_valid(password):
    # Password must be at least 8 characters long and contain letters, numbers, and special characters
    if len(password) < 8:
        return False
    if not re.search(r"[A-Za-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        user = read_user_by_credentials(email, password)
        if user:
            return render_template('chatbot.html', user=user)
        else:
            return "<script>alert('Invalid Credentials!'); window.location = '/';</script>"

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        firstname = request.form['first_name']
        lastname = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            return "<script>alert('Password and Confirm Password do not match!'); window.location = '/register';</script>"

        create_user(firstname, lastname, email, password)
        return "<script>alert('Registeration Sucess'); window.location = '/';</script>"

    return render_template('register.html')


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    return render_template('chatbot.html')

# app.py
@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    files = request.files.getlist('files')
    if not files:
        return jsonify({"message": "No files uploaded."}), 400

    try:
        documents = []
        for file in files:
            # Determine file type and read accordingly (using DocumentProcessor methods)
            if file.filename.lower().endswith('.pdf'):
                content = qa_system.document_processor._read_pdf(file) # Pass file object directly
            elif file.filename.lower().endswith('.docx'):
                content = qa_system.document_processor._read_docx(file)  # Pass file object directly
            elif file.filename.lower().endswith(('.txt', '.text')):  # Handle .txt and .text
                content = qa_system.document_processor._read_txt(file)  # Pass file object directly
            else:
                return jsonify({"message": f"Unsupported file type: {file.filename.split('.')[-1]}"}), 400

            if content:  # Check if content was successfully read
                documents.append(content)
            else:
                return jsonify({"message": f"Failed to process file: {file.filename}"}), 400

        if documents: #check if any document is there or not
            qa_system.document_processor.upload_documents(documents) # Pass list of strings
        else:
            return jsonify({"message": "No valid documents found in the upload"}), 400

        return jsonify({"message": "Documents uploaded and processed successfully."})

    except Exception as e:
        return jsonify({"message": f"Error during upload and processing: {str(e)}"}), 500


@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('question', '')

    if not query:
        return jsonify({"success": False, "message": "No question provided."})

    try:
        answer = qa_system.generate_response(query)  # Use generate_response
        relevant_chunks = qa_system.document_processor.retrieve(query, top_k=3)  # Retrieve used segments
        segments = [chunk[0] for chunk in relevant_chunks] if relevant_chunks else [] # Extract segments

        return jsonify({"success": True, "response": answer, "segments": segments})

    except Exception as e:
        return jsonify({"success": False, "message": f"Error processing question: {str(e)}"}), 500








# =================================================================================================================================================================
#                                                           Android--User Module
#
# =================================================================================================================================================================


# Simple in-memory storage (for testing purposes only)
users = {}


# Replace your android_login and android_register functions with these fixed versions:

@app.route('/android_login', methods=['POST'])
def android_login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data received'
            }), 400

        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({
                'success': False,
                'message': 'Email and password are required'
            }), 400

        # OPTION 1: Use your existing Firebase authentication
        # Uncomment this if you want to use Firebase
        # user = read_user_by_credentials(email, password)
        # if user:
        #     return jsonify({
        #         'success': True,
        #         'message': 'Login successful',
        #         'user': user
        #     }), 200
        # else:
        #     return jsonify({
        #         'success': False,
        #         'message': 'Invalid email or password'
        #     }), 401

        # OPTION 2: Use in-memory storage (current approach)
        # Check if user exists and password matches
        if email in users and users[email]['password'] == password:
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': {
                    'email': email,
                    'first_name': users[email]['first_name'],
                    'last_name': users[email]['last_name']
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid email or password. Please register first or check your credentials.'
            }), 401

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/android_register', methods=['POST'])
def android_register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data received'
            }), 400

        first_name = data.get('first_name')
        last_name = data.get('last_name')
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        # Validation
        if not all([first_name, last_name, email, password, confirm_password]):
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400

        if password != confirm_password:
            return jsonify({
                'success': False,
                'message': 'Passwords do not match'
            }), 400

        if len(password) < 6:
            return jsonify({
                'success': False,
                'message': 'Password must be at least 6 characters long'
            }), 400

        # Check if user already exists
        if email in users:
            return jsonify({
                'success': False,
                'message': 'Email already exists'
            }), 409

        # Create new user in memory
        users[email] = {
            'first_name': first_name,
            'last_name': last_name,
            'password': password
        }

        # ALSO save to Firebase (optional, but recommended for persistence)
        try:
            create_user(first_name, last_name, email, password)
        except Exception as firebase_error:
            logger.warning(f"Firebase save failed: {firebase_error}")
            # Continue even if Firebase fails

        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'email': email,
                'first_name': first_name,
                'last_name': last_name
            }
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


# Add this test endpoint to check registered users
@app.route('/debug/users', methods=['GET'])
def debug_users():
    return jsonify({
        'registered_users': list(users.keys()),
        'user_count': len(users)
    })


@app.route('/android_ask_question', methods=['POST'])
def android_ask_question():
    try:
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({
                'success': False,
                'message': 'Question is required'
            }), 400

        # Simple response (replace with your AI logic)
        response = f"I received your question: '{question}'. This is a test response from the server."

        return jsonify({
            'success': True,
            'response': response,
            'segments': ['Sample segment 1', 'Sample segment 2']
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Server error'
        }), 500


@app.route('/android_upload_documents', methods=['POST'])
def android_upload_documents():
    try:
        files = request.files.getlist('files')

        if not files:
            return jsonify({"message": "No files uploaded."}), 400

        documents = []
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                content = qa_system.document_processor._read_pdf(file)
            elif file.filename.lower().endswith('.docx'):
                content = qa_system.document_processor._read_docx(file)
            elif file.filename.lower().endswith(('.txt', '.text')):
                content = qa_system.document_processor._read_txt(file)
            else:
                return jsonify({"message": f"Unsupported file type: {file.filename.split('.')[-1]}"}), 400

            if content:
                documents.append(content)
            else:
                return jsonify({"message": f"Failed to process file: {file.filename}"}), 400

        if documents:
            qa_system.document_processor.upload_documents(documents)
        else:
            return jsonify({"message": "No valid documents found in the upload"}), 400

        return jsonify({"message": "Documents uploaded and processed successfully."})

    except Exception as e:
        return jsonify({"message": f"Error during upload and processing: {str(e)}"}), 500



@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'Server is running',
        'registered_users': len(users)
    }), 200


if __name__ == '__main__':
    print("Starting EasyDocs Test Server...")
    print("Registered endpoints:")
    print("- POST /android_login")
    print("- POST /android_register")
    print("- POST /android_ask_question")
    print("- POST /android_upload_documents")
    print("- GET /health")
    # Ensure the uploads directory exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(host='0.0.0.0', port=5000, debug=True)
