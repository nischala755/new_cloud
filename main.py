import streamlit as st
import os
import base64
import hashlib
import random
import string
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import torch
import torch.nn as nn
from datetime import datetime
from PIL import Image
import io

import re
import uuid
import pickle
from collections import defaultdict

#\t=True)

# ... existing imports ...

# Add new session state variables
if 'encryption_keys' not in st.session_state:
    st.session_state.encryption_keys = {}
if 'key_history' not in st.session_state:
    st.session_state.key_history = []

# Enhance encrypt_file function
def encrypt_file(file_data, method, file_id):
    """Enhanced encrypt file with key tracking"""
    encryption_time = datetime.now()
    
    if method == 'AES':
        key = generate_secure_key(32)
        iv = os.urandom(16)
        # Store key info
        st.session_state.encryption_keys[file_id] = {
            'method': 'AES',
            'key': base64.b64encode(key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'timestamp': encryption_time,
            'key_size': len(key) * 8
        }
        # ... existing AES encryption code ...
    
    elif method == 'RSA':
        # ... existing RSA code ...
        st.session_state.encryption_keys[file_id] = {
            'method': 'RSA',
            'public_key': public_key.public_bytes(),
            'timestamp': encryption_time,
            'key_size': 2048
        }
        # ... rest of RSA code ...
    
    # Similar updates for Kyber and NTRU
    # ... existing code ...

# Add new function to view encryption details
def view_encryption_details(file_id, file_info):
    """View detailed encryption information"""
    st.markdown("### Encryption Details")
    
    if file_id in st.session_state.encryption_keys:
        key_info = st.session_state.encryption_keys[file_id]
        
        st.markdown(f"""
        <div class='dashboard-card'>
            <h4>Encryption Method: {key_info['method']}</h4>
            <p>Key Size: {key_info['key_size']} bits</p>
            <p>Generated: {key_info['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show key preview (first few characters)
        if 'key' in key_info:
            key_preview = key_info['key'][:16] + "..."
            st.code(f"Key Preview: {key_preview}")
        
        # Show encrypted content preview
        if 'data' in file_info:
            encrypted_preview = base64.b64encode(file_info['data'][:100]).decode()
            st.markdown("#### Encrypted Content Preview")
            st.code(encrypted_preview)

# Enhance render_my_files function
def render_my_files():
    """Enhanced file management with encryption details"""
    st.markdown("## My Files")
    
    # ... existing code ...
    
    for i, (file_id, file_info) in enumerate(filtered_files.items()):
        col = cols[i % 3]
        
        with col:
            # ... existing file card code ...
            
            # Add encryption details button
            if st.button(f"View Encryption Details {file_id}", key=f"encrypt_details_{file_id}"):
                view_encryption_details(file_id, file_info)
            
            # Enhanced sharing dialog
            if st.button(f"Share {file_id}", key=f"share_{file_id}"):
                show_enhanced_sharing_dialog(file_id, file_info)

def show_enhanced_sharing_dialog(file_id, file_info):
    """Enhanced file sharing with key management"""
    st.markdown("### Share File")
    
    # Basic file info
    st.markdown(f"""
    <div class='file-card'>
        <h4>{file_info['filename']}</h4>
        <p>Encryption: {file_info['encryption_method']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sharing options
    other_users = [user for user in st.session_state.users.keys() 
                  if user != st.session_state.username]
    target_user = st.selectbox("Share with", other_users)
    
    # Advanced sharing options
    sharing_options = st.expander("Advanced Sharing Options")
    with sharing_options:
        share_key = st.checkbox("Share encryption key", value=True)
        expiry_days = st.number_input("Access expiry (days)", min_value=1, value=7)
        permission = st.selectbox("Permission", ["read", "write", "full"])
    
    if st.button("Share"):
        share_file_with_key(file_id, target_user, share_key, expiry_days, permission)
        st.success(f"File shared with {target_user}")

def share_file_with_key(file_id, target_user, share_key, expiry_days, permission):
    """Share file with optional key sharing"""
    # Get file info
    file_info = st.session_state.users[st.session_state.username]['files'][file_id]
    
    # Create shared file record
    shared_file = file_info.copy()
    shared_file['owner'] = st.session_state.username
    shared_file['permission'] = permission
    shared_file['expiry'] = datetime.now() + timedelta(days=expiry_days)
    
    if share_key and file_id in st.session_state.encryption_keys:
        shared_file['encryption_key'] = st.session_state.encryption_keys[file_id]
    
    # Add to target user's shared files
    st.session_state.users[target_user]['shared_files'][file_id] = shared_file
    
    # Log sharing
    log_file_access(file_id, st.session_state.username, 'share')

# Add to render_dashboard
def render_dashboard():
    # ... existing dashboard code ...
    
    # Add encryption key statistics
    st.markdown("### Encryption Key Statistics")
    
    # Count keys by method
    key_stats = defaultdict(int)
    for key_info in st.session_state.encryption_keys.values():
        key_stats[key_info['method']] += 1
    
    # Display stats
    col1, col2 = st.columns(2)
    
    with col1:
        # Key distribution chart
        if key_stats:
            fig, ax = plt.subplots()
            methods = list(key_stats.keys())
            counts = list(key_stats.values())
            ax.bar(methods, counts)
            ax.set_title("Encryption Methods Distribution")
            st.pyplot(fig)
    
    with col2:
        # Recent keys
        st.markdown("#### Recent Encryption Keys")
        recent_keys = sorted(
            st.session_state.encryption_keys.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )[:5]
        
        for file_id, key_info in recent_keys:
            st.markdown(f"""
            <div style='padding: 10px; background-color: white; border-radius: 5px; margin-bottom: 10px;'>
                <p><strong>File ID:</strong> {file_id}</p>
                <p><strong>Method:</strong> {key_info['method']}</p>
                <p><strong>Generated:</strong> {key_info['timestamp']}</p>
            </div>
            """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="CipherCloud",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg {
        padding: 1rem 1rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .css-1cpxqw2 {
        border-radius: 10px;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .css-1v3fvcr {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4e8df5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
    }
    .file-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .file-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .encryption-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-right: 5px;
    }
    .aes-badge {
        background-color: #d4f1f9;
        color: #05a2dc;
    }
    .rsa-badge {
        background-color: #ffeaa7;
        color: #fdcb6e;
    }
    .kyber-badge {
        background-color: #e3f9e5;
        color: #27ae60;
    }
    .ntru-badge {
        background-color: #f9e3e3;
        color: #e74c3c;
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        height: 100%;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4e8df5;
    }
    .metric-label {
        font-size: 1rem;
        color: #718096;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'users' not in st.session_state:
    st.session_state.users = {
        "admin": {
            "password": "admin123",
            "files": {},
            "score": 85,
            "shared_files": {},
            "access_history": []
        },
        "user1": {
            "password": "user123",
            "files": {},
            "score": 75,
            "shared_files": {},
            "access_history": []
        }
    }
if 'entropy_pool' not in st.session_state:
    st.session_state.entropy_pool = []
if 'file_counter' not in st.session_state:
    st.session_state.file_counter = 0
if 'merkle_trees' not in st.session_state:
    st.session_state.merkle_trees = {}
if 'nlp_permissions' not in st.session_state:
    st.session_state.nlp_permissions = {}
if 'encryption_stats' not in st.session_state:
    st.session_state.encryption_stats = {
        'AES': 0,
        'RSA': 0,
        'Kyber': 0,
        'NTRU': 0
    }
if 'access_logs' not in st.session_state:
    st.session_state.access_logs = []
if 'qubit_keys' not in st.session_state:
    st.session_state.qubit_keys = {}

# File storage directory
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================ UTILITY FUNCTIONS ================

def collect_entropy(event_data):
    """Collect entropy from user events for stronger key generation"""
    st.session_state.entropy_pool.append(event_data)
    if len(st.session_state.entropy_pool) > 1000:
        st.session_state.entropy_pool = st.session_state.entropy_pool[-1000:]

def generate_secure_key(length=32):
    """Generate a secure key using collected entropy"""
    if not st.session_state.entropy_pool:
        # Fallback if no entropy collected
        return os.urandom(length)
    
    # Mix entropy with system randomness
    entropy_str = ''.join(str(e) for e in st.session_state.entropy_pool)
    seed = hashlib.sha256(entropy_str.encode() + os.urandom(16)).digest()
    return seed[:length]

def simulate_qubit_key_generation(file_id, length=32):
    """Simulate quantum computing-based key generation"""
    # Simulate quantum superposition and entanglement
    qubits = []
    for _ in range(length * 8):  # 8 bits per byte
        # Simulate a qubit in superposition (0 and 1 simultaneously)
        qubit = random.choice([0, 1])
        qubits.append(qubit)
    
    # Simulate measurement collapsing superposition
    measured_bits = ''.join(str(q) for q in qubits)
    key = hashlib.sha256(measured_bits.encode()).digest()
    
    # Store the key
    st.session_state.qubit_keys[file_id] = key
    return key

def analyze_file_for_encryption(file_data, filename):
    """AI analysis to determine the best encryption method for a file"""
    file_size = len(file_data)
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Simple neural network to decide encryption method
    class EncryptionSelector(nn.Module):
        def __init__(self):
            super(EncryptionSelector, self).__init__()
            self.fc1 = nn.Linear(3, 10)
            self.fc2 = nn.Linear(10, 4)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=0)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.softmax(self.fc2(x))
            return x
    
    # Feature extraction
    # 1. File size (normalized)
    size_feature = min(file_size / 10000000, 1.0)  # Normalize to 0-1 range
    
    # 2. File type sensitivity (based on extension)
    sensitive_extensions = ['.pdf', '.docx', '.xlsx', '.txt', '.key']
    medium_extensions = ['.jpg', '.png', '.mp3', '.mp4']
    sensitivity = 0.9 if file_extension in sensitive_extensions else 0.5 if file_extension in medium_extensions else 0.3
    
    # 3. Content complexity (simple heuristic)
    complexity = min(len(set(file_data[:1000])) / 256, 1.0)  # Unique byte ratio
    
    # Create input tensor
    features = torch.tensor([size_feature, sensitivity, complexity], dtype=torch.float32)
    
    # Initialize model
    model = EncryptionSelector()
    
    # Forward pass
    with torch.no_grad():
        output = model(features)
    
    # Decision logic
    encryption_methods = ['AES', 'RSA', 'Kyber', 'NTRU']
    probabilities = output.numpy()
    
    # Deterministic logic override for demo purposes
    if file_size < 1000000:  # < 1MB
        selected_method = 'AES'
        reason = "Small file size, symmetric encryption is efficient"
    elif sensitivity > 0.7:
        selected_method = 'Kyber' if random.random() > 0.5 else 'NTRU'
        reason = "Sensitive file content, quantum-resistant encryption recommended"
    elif complexity > 0.7:
        selected_method = 'RSA'
        reason = "Complex file content, asymmetric encryption provides better security"
    else:
        # Use the model's prediction
        selected_index = np.argmax(probabilities)
        selected_method = encryption_methods[selected_index]
        reason = "AI model recommendation based on file characteristics"
    
    # Update stats
    st.session_state.encryption_stats[selected_method] += 1
    
    return selected_method, reason, dict(zip(encryption_methods, probabilities))

def encrypt_file(file_data, method, file_id):
    """Encrypt file using the selected method"""
    if method == 'AES':
        key = generate_secure_key(32)
        iv = os.urandom(16)
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(file_data) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return base64.b64encode(encrypted_data), {'key': base64.b64encode(key).decode(), 'iv': base64.b64encode(iv).decode()}
    
    elif method == 'RSA':
        # For demonstration, using smaller key size for speed
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # RSA can only encrypt small chunks, so we'll use hybrid encryption
        aes_key = generate_secure_key(32)
        iv = os.urandom(16)
        
        # Encrypt the AES key with RSA
        encrypted_key = public_key.encrypt(
            aes_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Encrypt the file with AES
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(file_data) + padder.finalize()
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine encrypted key and data
        result = base64.b64encode(encrypted_key) + b'.' + base64.b64encode(encrypted_data)
        
        # For demo purposes, we're storing the private key - in a real system, this would be handled differently
        return result, {
            'private_key': pickle.dumps(private_key),
            'iv': base64.b64encode(iv).decode()
        }
    
    elif method == 'Kyber':
        # Simulate Kyber (a post-quantum KEM)
        # In a real implementation, you would use a library like liboqs
        key = simulate_qubit_key_generation(file_id, 32)
        iv = os.urandom(16)
        
        # Use AES for the actual encryption (simulating hybrid encryption)
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(file_data) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return base64.b64encode(encrypted_data), {
            'key': base64.b64encode(key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'quantum_resistant': True
        }
    
    elif method == 'NTRU':
        # Simulate NTRU (another post-quantum algorithm)
        key = simulate_qubit_key_generation(file_id, 32)
        iv = os.urandom(16)
        
        # Use AES for the actual encryption (simulating hybrid encryption)
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(file_data) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return base64.b64encode(encrypted_data), {
            'key': base64.b64encode(key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'quantum_resistant': True
        }
    
    else:
        raise ValueError(f"Unsupported encryption method: {method}")

def decrypt_file(encrypted_data, encryption_info, method):
    """Decrypt file using the selected method"""
    if method == 'AES':
        key = base64.b64decode(encryption_info['key'])
        iv = base64.b64decode(encryption_info['iv'])
        encrypted_bytes = base64.b64decode(encrypted_data)
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()
        
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    elif method == 'RSA':
        parts = encrypted_data.split(b'.')
        encrypted_key = base64.b64decode(parts[0])
        encrypted_bytes = base64.b64decode(parts[1])
        
        private_key = pickle.loads(encryption_info['private_key'])
        iv = base64.b64decode(encryption_info['iv'])
        
        # Decrypt the AES key with RSA
        aes_key = private_key.decrypt(
            encrypted_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt the file with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()
        
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    elif method in ['Kyber', 'NTRU']:
        key = base64.b64decode(encryption_info['key'])
        iv = base64.b64decode(encryption_info['iv'])
        encrypted_bytes = base64.b64decode(encrypted_data)
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()
        
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    else:
        raise ValueError(f"Unsupported encryption method: {method}")

def create_merkle_tree(file_data, file_id):
    """Create a Merkle tree for file integrity verification"""
    # Split file into chunks
    chunk_size = 1024  # 1KB chunks
    chunks = [file_data[i:i+chunk_size] for i in range(0, len(file_data), chunk_size)]
    
    # Calculate leaf hashes
    leaf_hashes = [hashlib.sha256(chunk).hexdigest() for chunk in chunks]
    
    # Build the tree
    tree = leaf_hashes.copy()
    while len(tree) > 1:
        # If odd number of nodes, duplicate the last one
        if len(tree) % 2 == 1:
            tree.append(tree[-1])
        
        # Create parent nodes
        parents = []
        for i in range(0, len(tree), 2):
            combined = tree[i] + tree[i+1]
            parent_hash = hashlib.sha256(combined.encode()).hexdigest()
            parents.append(parent_hash)
        
        tree = parents
    
    # Store the tree
    st.session_state.merkle_trees[file_id] = {
        'root': tree[0],
        'leaves': leaf_hashes,
        'access_history': []
    }
    
    return tree[0]  # Return the root hash

def verify_file_integrity(file_data, file_id):
    """Verify file integrity using the Merkle tree"""
    if file_id not in st.session_state.merkle_trees:
        return False, "No Merkle tree found for this file"
    
    # Split file into chunks
    chunk_size = 1024  # 1KB chunks
    chunks = [file_data[i:i+chunk_size] for i in range(0, len(file_data), chunk_size)]
    
    # Calculate leaf hashes
    leaf_hashes = [hashlib.sha256(chunk).hexdigest() for chunk in chunks]
    
    # Compare with stored leaves
    stored_leaves = st.session_state.merkle_trees[file_id]['leaves']
    
    if len(leaf_hashes) != len(stored_leaves):
        return False, "File size has changed"
    
    # Find any modified chunks
    modified_chunks = []
    for i, (current, stored) in enumerate(zip(leaf_hashes, stored_leaves)):
        if current != stored:
            modified_chunks.append(i)
    
    if modified_chunks:
        return False, f"File has been modified in chunks: {modified_chunks}"
    
    return True, "File integrity verified"

def log_file_access(file_id, username, action):
    """Log file access to the Merkle tree and global logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'username': username,
        'action': action,
        'file_id': file_id
    }
    
    # Add to global logs
    st.session_state.access_logs.append(log_entry)
    
    # Add to user's access history
    st.session_state.users[username]['access_history'].append(log_entry)
    
    # Add to file's Merkle tree access history
    if file_id in st.session_state.merkle_trees:
        st.session_state.merkle_trees[file_id]['access_history'].append(log_entry)
    
    # Update user score based on activity
    update_user_score(username, action)

def update_user_score(username, action):
    """Update user score based on their file access patterns and actions"""
    score_changes = {
        'upload': 5,
        'download': 1,
        'share': 3,
        'delete': -1,
        'view': 0.5
    }
    
    if action in score_changes:
        st.session_state.users[username]['score'] += score_changes[action]
        # Cap score between 0 and 100
        st.session_state.users[username]['score'] = max(0, min(100, st.session_state.users[username]['score']))

# Replace the parse_nlp_permission function with a simpler version that doesn't use NLTK
def parse_nlp_permission(command, username):
    """Parse natural language permission commands without using NLTK"""
    command = command.lower()
    
    # Extract action (grant, revoke, etc.)
    action_words = {'grant', 'give', 'allow', 'share', 'revoke', 'remove', 'deny', 'restrict'}
    action = None
    for word in action_words:
        if word in command:
            action = 'grant' if word in {'grant', 'give', 'allow', 'share'} else 'revoke'
            break
    
    # Extract permission type (read, write, etc.)
    permission_types = {'read', 'write', 'edit', 'delete', 'view', 'access', 'full'}
    permission = None
    for word in permission_types:
        if word in command:
            permission = word
            break
    
    # Extract target user
    users = list(st.session_state.users.keys())
    target_user = None
    for user in users:
        if user in command and user != username:
            target_user = user
            break
    
    # Extract file name/id
    file_pattern = r'file\s+(\w+)'
    file_matches = re.findall(file_pattern, command)
    file_id = file_matches[0] if file_matches else None
    
    # If file_id not found by pattern, try to find any file ID in the command
    if not file_id:
        user_files = st.session_state.users[username]['files']
        for fid in user_files:
            if fid in command:
                file_id = fid
                break
    
    return {
        'action': action,
        'permission': permission if permission else 'read',  # Default to read
        'target_user': target_user,
        'file_id': file_id,
        'success': all([action, target_user, file_id])
    }

def apply_permission(parsed_command, username):
    """Apply the parsed permission command"""
    if not parsed_command['success']:
        missing = []
        if not parsed_command['action']:
            missing.append('action (grant or revoke)')
        if not parsed_command['target_user']:
            missing.append('target user')
        if not parsed_command['file_id']:
            missing.append('file identifier')
        
        return False, f"Could not understand the command. Missing: {', '.join(missing)}"
    
    action = parsed_command['action']
    permission = parsed_command['permission']
    target_user = parsed_command['target_user']
    file_id = parsed_command['file_id']
    
    # Check if file exists and belongs to the user
    if file_id not in st.session_state.users[username]['files']:
        return False, f"You don't own a file with ID {file_id}"
    
    # Check if target user exists
    if target_user not in st.session_state.users:
        return False, f"User {target_user} does not exist"
    
    # Initialize permissions structure if needed
    if file_id not in st.session_state.nlp_permissions:
        st.session_state.nlp_permissions[file_id] = {}
    
    if action == 'grant':
        # Grant permission
        st.session_state.nlp_permissions[file_id][target_user] = permission
        
        # Add to shared files for the target user
        if file_id not in st.session_state.users[target_user]['shared_files']:
            file_info = st.session_state.users[username]['files'][file_id].copy()
            file_info['owner'] = username
            file_info['permission'] = permission
            st.session_state.users[target_user]['shared_files'][file_id] = file_info
        else:
            st.session_state.users[target_user]['shared_files'][file_id]['permission'] = permission
        
        return True, f"Granted {permission} permission to {target_user} for file {file_id}"
    
    elif action == 'revoke':
        # Revoke permission
        if target_user in st.session_state.nlp_permissions.get(file_id, {}):
            del st.session_state.nlp_permissions[file_id][target_user]
            
            # Remove from shared files for the target user
            if file_id in st.session_state.users[target_user]['shared_files']:
                del st.session_state.users[target_user]['shared_files'][file_id]
            
            return True, f"Revoked permissions from {target_user} for file {file_id}"
        else:
            return False, f"{target_user} doesn't have permissions for file {file_id}"
    
    return False, "Unknown action"

# ================ UI COMPONENTS ================

def login_page():
    """Render the login page"""
    st.markdown("<h1 style='text-align: center; color: #4e8df5;'>CipherCloud</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Secure File Sharing with Advanced Encryption</p>", unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Login</h2>", unsafe_allow_html=True)
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if username in st.session_state.users and st.session_state.users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        st.markdown("<p style='text-align: center; margin-top: 20px;'>Demo accounts: admin/admin123 or user1/user123</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def main_app():
    """Render the main application after login"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"<h3>Welcome, {st.session_state.username}</h3>", unsafe_allow_html=True)
        
        # User score display
        user_score = st.session_state.users[st.session_state.username]['score']
        st.markdown(f"<div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 20px;'>"
                   f"<p style='margin-bottom: 5px;'>User Score</p>"
                   f"<div style='background-color: #e0e0e0; border-radius: 10px; height: 10px;'>"
                   f"<div style='background-color: {'#4CAF50' if user_score > 70 else '#FFC107' if user_score > 40 else '#F44336'}; width: {user_score}%; height: 100%; border-radius: 10px;'></div>"
                   f"</div>"
                   f"<p style='text-align: right; margin-top: 5px;'>{user_score}/100</p>"
                   f"</div>", unsafe_allow_html=True)
        
        # Navigation
               # Navigation
        st.markdown("### Navigation")
        app_mode = st.radio("", ["Dashboard", "My Files", "Shared Files", "Upload File", "File Permissions", "Settings"])
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    
    # Main content
    st.markdown("<h1 style='color: #4e8df5;'>CipherCloud</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2em;'>Secure File Sharing with Advanced Encryption</p>", unsafe_allow_html=True)
    
    # Display different sections based on navigation
    if app_mode == "Dashboard":
        render_dashboard()
    elif app_mode == "My Files":
        render_my_files()
    elif app_mode == "Shared Files":
        render_shared_files()
    elif app_mode == "Upload File":
        render_upload_file()
    elif app_mode == "File Permissions":
        render_file_permissions()
    elif app_mode == "Settings":
        render_settings()

def render_dashboard():
    """Render the dashboard with analytics and visualizations"""
    st.markdown("## Dashboard")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        num_files = len(st.session_state.users[st.session_state.username]['files'])
        st.markdown(f"<p class='metric-label'>My Files</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{num_files}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        num_shared = len(st.session_state.users[st.session_state.username]['shared_files'])
        st.markdown(f"<p class='metric-label'>Shared With Me</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{num_shared}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        user_score = st.session_state.users[st.session_state.username]['score']
        st.markdown(f"<p class='metric-label'>User Score</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{user_score}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        access_count = len(st.session_state.users[st.session_state.username]['access_history'])
        st.markdown(f"<p class='metric-label'>Activities</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{access_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create visualization row
    st.markdown("### Encryption Methods Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Encryption methods pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(st.session_state.encryption_stats.keys())
        sizes = list(st.session_state.encryption_stats.values())
        
        if sum(sizes) > 0:
            colors = ['#4e8df5', '#f5924e', '#4ef58d', '#f54e8d']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("No files have been encrypted yet.")
    
    with col2:
        # Recent activity
        st.markdown("### Recent Activity")
        
        user_history = st.session_state.users[st.session_state.username]['access_history']
        if user_history:
            for i, activity in enumerate(reversed(user_history[-5:])):
                st.markdown(f"""
                <div style='padding: 10px; background-color: white; border-radius: 5px; margin-bottom: 10px;'>
                    <p style='margin: 0; color: #718096;'>{activity['timestamp']}</p>
                    <p style='margin: 0; font-weight: bold;'>{activity['action'].capitalize()} - File ID: {activity['file_id']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent activity.")
    
    # File access patterns
    st.markdown("### File Access Patterns")
    
    # Get access history for visualization
    if st.session_state.access_logs:
        # Create a DataFrame for visualization
        df = pd.DataFrame(st.session_state.access_logs)
        
        # Filter for current user if needed
        # df = df[df['username'] == st.session_state.username]
        
        if not df.empty:
            # Group by action and count
            action_counts = df['action'].value_counts().reset_index()
            action_counts.columns = ['Action', 'Count']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Action', y='Count', data=action_counts, ax=ax)
            ax.set_title('Actions Performed')
            ax.set_xlabel('Action Type')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.info("Not enough data for visualization.")
    else:
        st.info("No access logs available yet.")

def render_my_files():
    """Render the user's files"""
    st.markdown("## My Files")
    
    user_files = st.session_state.users[st.session_state.username]['files']
    
    if not user_files:
        st.info("You haven't uploaded any files yet. Go to the Upload File section to get started.")
        return
    
    # Create a search box
    search_term = st.text_input("Search files", "")
    
    # Filter files based on search term
    filtered_files = {
        file_id: file_info for file_id, file_info in user_files.items()
        if search_term.lower() in file_info['filename'].lower()
    }
    
    # Display files in a grid
    cols = st.columns(3)
    
    for i, (file_id, file_info) in enumerate(filtered_files.items()):
        col = cols[i % 3]
        
        with col:
            st.markdown(f"""
            <div class='file-card'>
                <h3>{file_info['filename']}</h3>
                <p>Uploaded: {file_info['upload_date']}</p>
                <p>Size: {file_info['size']} bytes</p>
                <p>Encryption: <span class='{file_info['encryption_method'].lower()}-badge encryption-badge'>{file_info['encryption_method']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # File actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"Download {file_id}", key=f"download_{file_id}"):
                    # Decrypt and download file
                    try:
                        encrypted_data = file_info['data']
                        decrypted_data = decrypt_file(
                            encrypted_data,
                            file_info['encryption_info'],
                            file_info['encryption_method']
                        )
                        
                        # Log the access
                        log_file_access(file_id, st.session_state.username, 'download')
                        
                        # Verify integrity
                        integrity_ok, message = verify_file_integrity(decrypted_data, file_id)
                        if not integrity_ok:
                            st.warning(message)
                        
                        # Create download link
                        b64 = base64.b64encode(decrypted_data).decode()
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_info["filename"]}">Click to download</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error downloading file: {str(e)}")
            
            with col2:
                if st.button(f"Share {file_id}", key=f"share_{file_id}"):
                    # Set session state to show sharing dialog
                    st.session_state.sharing_file_id = file_id
                    st.rerun()
            
            with col3:
                if st.button(f"Delete {file_id}", key=f"delete_{file_id}"):
                    # Delete file
                    del st.session_state.users[st.session_state.username]['files'][file_id]
                    
                    # Log the access
                    log_file_access(file_id, st.session_state.username, 'delete')
                    
                    st.success(f"File {file_id} deleted successfully.")
                    st.rerun()
    
    # Handle file sharing dialog
    if hasattr(st.session_state, 'sharing_file_id'):
        file_id = st.session_state.sharing_file_id
        
        st.markdown("### Share File")
        st.markdown(f"Sharing file: {st.session_state.users[st.session_state.username]['files'][file_id]['filename']}")
        
        # List of users to share with
        other_users = [user for user in st.session_state.users.keys() if user != st.session_state.username]
        target_user = st.selectbox("Select user to share with", other_users)
        
        permission = st.selectbox("Permission", ["read", "write", "full"])
        
        if st.button("Share"):
            # Create NLP command
            command = f"grant {permission} access to {target_user} for file {file_id}"
            parsed = parse_nlp_permission(command, st.session_state.username)
            success, message = apply_permission(parsed, st.session_state.username)
            
            if success:
                st.success(message)
                # Log the access
                log_file_access(file_id, st.session_state.username, 'share')
            else:
                st.error(message)
            
            # Clear sharing state
            del st.session_state.sharing_file_id
            st.rerun()
        
        if st.button("Cancel"):
            # Clear sharing state
            del st.session_state.sharing_file_id
            st.rerun()

def render_shared_files():
    """Render files shared with the user"""
    st.markdown("## Files Shared With Me")
    
    shared_files = st.session_state.users[st.session_state.username]['shared_files']
    
    if not shared_files:
        st.info("No files have been shared with you yet.")
        return
    
    # Display shared files
    cols = st.columns(3)
    
    for i, (file_id, file_info) in enumerate(shared_files.items()):
        col = cols[i % 3]
        
        with col:
            st.markdown(f"""
            <div class='file-card'>
                <h3>{file_info['filename']}</h3>
                <p>Owner: {file_info['owner']}</p>
                <p>Permission: {file_info['permission']}</p>
                <p>Encryption: <span class='{file_info['encryption_method'].lower()}-badge encryption-badge'>{file_info['encryption_method']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # File actions
            if st.button(f"View {file_id}", key=f"view_shared_{file_id}"):
                # Get file from owner's storage
                owner = file_info['owner']
                owner_file_info = st.session_state.users[owner]['files'][file_id]
                
                # Decrypt and view file
                try:
                    encrypted_data = owner_file_info['data']
                    decrypted_data = decrypt_file(
                        encrypted_data,
                        owner_file_info['encryption_info'],
                        owner_file_info['encryption_method']
                    )
                    
                    # Log the access
                    log_file_access(file_id, st.session_state.username, 'view')
                    
                    # Create download link
                    b64 = base64.b64encode(decrypted_data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_info["filename"]}">Click to download</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error viewing file: {str(e)}")

def render_upload_file():
    """Render the file upload section"""
    st.markdown("## Upload File")
    
    uploaded_file = st.file_uploader("Choose a file", type=None)
    
    if uploaded_file is not None:
        # Read file data
        file_data = uploaded_file.read()
        
        # Analyze file for encryption
        encryption_method, reason, probabilities = analyze_file_for_encryption(file_data, uploaded_file.name)
        
        # Display encryption recommendation
        st.markdown("### AI Encryption Analysis")
        
        st.markdown(f"""
        <div style='background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4>Recommended Encryption: <span class='{encryption_method.lower()}-badge encryption-badge'>{encryption_method}</span></h4>
            <p><strong>Reason:</strong> {reason}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization of encryption probabilities
        st.markdown("#### Encryption Method Probabilities")
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'Method': list(probabilities.keys()),
            'Probability': list(probabilities.values())
        })
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = sns.barplot(x='Method', y='Probability', data=df, ax=ax)
        
        # Add percentage labels on top of bars
        for i, p in enumerate(bars.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                    f'{height:.1%}',
                    ha="center", fontsize=10)
        
        ax.set_ylim(0, 1.1)
        ax.set_title('Encryption Method Probabilities')
        ax.set_ylabel('Probability')
        st.pyplot(fig)
        
        # Allow user to override encryption method
        selected_method = st.selectbox(
            "Select encryption method",
            ['AES', 'RSA', 'Kyber', 'NTRU'],
            index=['AES', 'RSA', 'Kyber', 'NTRU'].index(encryption_method)
        )
        
        # Upload button
        if st.button("Upload and Encrypt"):
            # Generate file ID
            file_id = f"file_{st.session_state.file_counter}"
            st.session_state.file_counter += 1
            
            # Encrypt file
            encrypted_data, encryption_info = encrypt_file(file_data, selected_method, file_id)
            
            # Create Merkle tree for integrity verification
            merkle_root = create_merkle_tree(file_data, file_id)
            
            # Store file information
            st.session_state.users[st.session_state.username]['files'][file_id] = {
                'filename': uploaded_file.name,
                'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'size': len(file_data),
                'encryption_method': selected_method,
                'encryption_info': encryption_info,
                'data': encrypted_data,
                'merkle_root': merkle_root
            }
            
            # Log the access
            log_file_access(file_id, st.session_state.username, 'upload')
            
            st.success(f"File uploaded and encrypted successfully with {selected_method}!")
            st.markdown(f"File ID: {file_id}")

def render_file_permissions():
    """Render the file permissions management section"""
    st.markdown("## File Permissions")
    
    # Simplified permission management
    st.markdown("### Manage File Permissions")
    
    # Get user files
    user_files = st.session_state.users[st.session_state.username]['files']
    
    if not user_files:
        st.info("You haven't uploaded any files yet.")
        return
    
    # File selection
    file_options = {f"{file_id}: {info['filename']}" for file_id, info in user_files.items()}
    selected_file = st.selectbox("Select a file", list(file_options), label_visibility="visible")
    
    if selected_file:
        file_id = selected_file.split(":")[0].strip()
        
        # User selection
        other_users = [user for user in st.session_state.users.keys() if user != st.session_state.username]
        if not other_users:
            st.warning("No other users to share with.")
            return
            
        target_user = st.selectbox("Select user to share with", other_users)
        
        # Permission selection
        permission = st.selectbox("Select permission level", ["read", "write", "full"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Grant Permission"):
                # Create command
                command = f"grant {permission} access to {target_user} for file {file_id}"
                parsed = parse_nlp_permission(command, st.session_state.username)
                success, message = apply_permission(parsed, st.session_state.username)
                
                if success:
                    st.success(message)
                    # Log the access
                    log_file_access(file_id, st.session_state.username, 'share')
                else:
                    st.error(message)
        
        with col2:
            if st.button("Revoke Permission"):
                # Create command
                command = f"revoke access from {target_user} for file {file_id}"
                parsed = parse_nlp_permission(command, st.session_state.username)
                success, message = apply_permission(parsed, st.session_state.username)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Display current permissions
    st.markdown("### Current Permissions")
    
    for file_id, file_info in user_files.items():
        st.markdown(f"""
        <div style='background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
            <h4>{file_info['filename']} (ID: {file_id})</h4>
        """, unsafe_allow_html=True)
        
        # Get permissions for this file
        file_permissions = st.session_state.nlp_permissions.get(file_id, {})
        
        if file_permissions:
            st.markdown("<table style='width: 100%;'>", unsafe_allow_html=True)
            st.markdown("<tr><th>User</th><th>Permission</th></tr>", unsafe_allow_html=True)
            
            for user, permission in file_permissions.items():
                st.markdown(f"""
                <tr>
                    <td>{user}</td>
                    <td>{permission}</td>
                </tr>
                """, unsafe_allow_html=True)
            
            st.markdown("</table>", unsafe_allow_html=True)
        else:
            st.markdown("<p>No permissions set for this file.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
def render_settings():
    """Render the settings section"""
    st.markdown("## Settings")
    
    # Entropy collection
    st.markdown("### Entropy Collection")
    st.markdown("""
    Move your mouse randomly in the box below to generate entropy for stronger encryption keys.
    This simulates collecting randomness from browser events.
    """)
    
    # Create a canvas for mouse movement
    st.markdown("""
    <div id="entropy-canvas" style="width: 100%; height: 200px; background-color: #f0f2f6; border-radius: 10px; position: relative;">
        <div id="entropy-pointer" style="width: 10px; height: 10px; background-color: #4e8df5; border-radius: 50%; position: absolute; top: 50%; left: 50%;"></div>
    </div>
    <script>
        const canvas = document.getElementById('entropy-canvas');
        const pointer = document.getElementById('entropy-pointer');
        
        canvas.addEventListener('mousemove', function(e) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            pointer.style.left = x + 'px';
            pointer.style.top = y + 'px';
            
            // In a real app, we would send this data to the server
            console.log('Entropy collected:', x, y, Date.now());
        });
    </script>
    """, unsafe_allow_html=True)
    
    # Simulate entropy collection
    if st.button("Simulate Entropy Collection"):
        # Generate random mouse movements
        for _ in range(100):
            x = random.randint(0, 100)
            y = random.randint(0, 100)
            timestamp = time.time()
            collect_entropy(f"{x},{y},{timestamp}")
        
        st.success(f"Collected 100 entropy samples. Total pool size: {len(st.session_state.entropy_pool)}")
    
    # Quantum key simulation
    st.markdown("### Quantum Key Simulation")
    st.markdown("""
    This section simulates quantum computing-based key generation using qubits.
    In a real quantum computer, qubits can exist in multiple states simultaneously due to superposition.
    """)
    
    if st.button("Simulate Qubit Key Generation"):
        # Generate a sample key
        key = simulate_qubit_key_generation("demo", 16)
        key_hex = key.hex()
        
        # Visualize the key as qubits
        st.markdown("#### Simulated Qubit States")
        
        # Convert key to binary representation
        key_bin = ''.join(format(byte, '08b') for byte in key[:8])  # Show first 8 bytes
        
        # Display as a grid of qubits
        cols = st.columns(8)
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"**Qubit {i+1}**")
                for j in range(8):
                    idx = i*8 + j
                    if idx < len(key_bin):
                        bit = key_bin[idx]
                        color = "#4e8df5" if bit == "1" else "#f5f7f9"
                        st.markdown(f"""
                        <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {color}; 
                                    margin: 5px auto; display: flex; justify-content: center; align-items: center;
                                    color: white; font-weight: bold;">
                            {bit}
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown(f"**Generated Key (Hex):** `{key_hex}`")

# ================ MAIN APPLICATION LOGIC ================

def main():
    """Main application entry point"""
    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()