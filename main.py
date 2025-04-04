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
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta
from PIL import Image
import io
import re
import uuid
import pickle
from collections import defaultdict
import zlib
import threading
import queue
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum

# Try to import torch, but make it optional
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes/functions if needed
    class DummyNN:
        class Module:
            pass
    class DummyTorch:
        nn = DummyNN
    torch = DummyTorch

# ... rest of your code ...

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ciphercloud')

# Constants
# Constants
MAX_LOGIN_ATTEMPTS = 3
LOGIN_COOLDOWN = 300  # seconds
FILE_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for large file handling
ENTROPY_POOL_SIZE = 10000
ENCRYPTION_METHODS = ['AES', 'RSA', 'Kyber', 'NTRU', 'ChaCha20']

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
            "shared_files": {},
            "access_history": [],
            "score": 75
        },
        "user1": {
            "password": "user123",
            "files": {},
            "shared_files": {},
            "access_history": [],
            "score": 50
        }
    }
if 'file_counter' not in st.session_state:
    st.session_state.file_counter = 1
if 'access_logs' not in st.session_state:
    st.session_state.access_logs = []
if 'merkle_trees' not in st.session_state:
    st.session_state.merkle_trees = {}
if 'nlp_permissions' not in st.session_state:
    st.session_state.nlp_permissions = {}
if 'entropy_pool' not in st.session_state:
    st.session_state.entropy_pool = []
if 'encryption_stats' not in st.session_state:
    st.session_state.encryption_stats = {
        'AES': 0,
        'RSA': 0,
        'Kyber': 0,
        'NTRU': 0,
        'ChaCha20': 0
    }
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = {}
if 'file_versions' not in st.session_state:
    st.session_state.file_versions = {}
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'activity_metrics' not in st.session_state:
    st.session_state.activity_metrics = defaultdict(lambda: defaultdict(int))
if 'encryption_keys' not in st.session_state:
    st.session_state.encryption_keys = {}
if 'key_history' not in st.session_state:
    st.session_state.key_history = []
if 'security_alerts' not in st.session_state:
    st.session_state.security_alerts = []
if 'file_comments' not in st.session_state:
    st.session_state.file_comments = {}
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

# Custom CSS for enhanced UI
def load_custom_css():
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #4e8df5;
            --secondary-color: #f5924e;
            --background-color: #f9fafc;
            --text-color: #333333;
            --card-bg-color: white;
            --success-color: #4CAF50;
            --warning-color: #FFC107;
            --danger-color: #F44336;
        }
        
        /* Dark mode colors */
        .dark-mode {
            --primary-color: #6e9df5;
            --secondary-color: #f5a24e;
            --background-color: #1e1e1e;
            --text-color: #f0f0f0;
            --card-bg-color: #2d2d2d;
        }
        
        /* Animation keyframes */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        /* General styles */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
        }
        
        /* Card styles */
        .dashboard-card, .file-card {
            background-color: var(--card-bg-color);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.5s ease-out;
        }
        
        .file-card {
            height: 100%;
            transition: transform 0.3s ease;
        }
        
        .file-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Metrics */
        .metric-label {
            font-size: 1em;
            color: #718096;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 0;
        }
        
        /* Encryption badges */
        .encryption-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            color: white;
        }
        
        .aes-badge {
            background-color: #4e8df5;
        }
        
        .rsa-badge {
            background-color: #f5924e;
        }
        
        .kyber-badge {
            background-color: #4ef58d;
        }
        
        .ntru-badge {
            background-color: #f54e8d;
        }
        
        .chacha20-badge {
            background-color: #9d4ef5;
        }
        
        /* Login container */
        .login-container {
            background-color: var(--card-bg-color);
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            animation: fadeIn 1s ease-in;
        }
        
        /* Notification styles */
        .notification {
            padding: 10px 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            animation: slideIn 0.3s ease-out;
        }
        
        .notification-info {
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
        }
        
        .notification-success {
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
        }
        
        .notification-warning {
            background-color: #fff8e1;
            border-left: 4px solid #FFC107;
        }
        
        .notification-error {
            background-color: #ffebee;
            border-left: 4px solid #F44336;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: var(--primary-color);
        }
        
        /* Button styles */
        .stButton>button {
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        th {
            background-color: #f8fafc;
            font-weight: bold;
        }
        
        tr:hover {
            background-color: #f8fafc;
        }
    </style>
    """, unsafe_allow_html=True)

# ================ SECURITY FUNCTIONS ================
# ... existing code ...
def render_shared_files():
    """Render the shared files section"""
    st.markdown("## Shared Files")
    
    # Check if user has any shared files
    if not st.session_state.users[st.session_state.username]["shared_files"]:
        st.info("No files have been shared with you yet.")
        return
    
    # Display shared files in a grid
    shared_files = st.session_state.users[st.session_state.username]["shared_files"]
    
    # Create columns for the grid
    cols = st.columns(3)
    
    for i, (file_id, file_info) in enumerate(shared_files.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="file-card">
                <h3>{file_info['filename']}</h3>
                <p>Shared by: {file_info['shared_by']}</p>
                <p>Encryption: <span class="{file_info['encryption'].lower()}-badge encryption-badge">{file_info['encryption']}</span></p>
                <p>Shared on: {file_info['shared_date']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            if st.button(f"Download {file_info['filename']}", key=f"download_shared_{file_id}"):
                try:
                    # Decrypt the file
                    decrypted_data = decrypt_file(
                        file_info['encrypted_data'],
                        file_info['encryption_info'],
                        file_info['encryption']
                    )
                    
                    # Create download link
                    b64 = base64.b64encode(decrypted_data).decode()
                    mime_type = "application/octet-stream"
                    href = f'<a href="data:{mime_type};base64,{b64}" download="{file_info["filename"]}">Download {file_info["filename"]}</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Log access
                    log_security_event(
                        st.session_state.username,
                        "file_access",
                        f"Downloaded shared file: {file_info['filename']}"
                    )
                    
                    # Update access history
                    st.session_state.users[st.session_state.username]["access_history"].append({
                        "file_id": file_id,
                        "action": "download",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"Error downloading file: {str(e)}")
            
            # File details button
            if st.button(f"Details", key=f"details_shared_{file_id}"):
                st.session_state.selected_file = file_id
                st.session_state.selected_file_type = "shared"
                st.rerun()
def login_page():
    """Render the login page"""
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>CipherCloud Login</h2>", unsafe_allow_html=True)
    
    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login"):
            if username and password:
                # Collect entropy from login attempt
                collect_entropy(f"{username}_{time.time()}")
                
                # Verify login
                success, message = verify_login(username, password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter both username and password")
    
    with col2:
        if st.button("Register"):
            if username and password:
                if username in st.session_state.users:
                    st.error("Username already exists")
                else:
                    # Check password strength
                    strength = check_password_strength(password)
                    
                    if strength['score'] < 40:
                        st.warning(f"Password strength: {strength['level']} ({strength['score']}/100)")
                        st.markdown("<ul>", unsafe_allow_html=True)
                        for feedback in strength['feedback']:
                            st.markdown(f"<li>{feedback}</li>", unsafe_allow_html=True)
                        st.markdown("</ul>", unsafe_allow_html=True)
                        
                        # Allow override with confirmation
                        if st.button("Use Weak Password Anyway"):
                            create_new_user(username, password)
                            st.success("Account created successfully! You can now login.")
                    else:
                        create_new_user(username, password)
                        st.success("Account created successfully! You can now login.")
            else:
                st.error("Please enter both username and password")
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_new_user(username, password):
    """Create a new user account"""
    st.session_state.users[username] = {
        "password": password,
        "files": {},
        "shared_files": {},
        "access_history": [],
        "score": 50  # Default score for new users
    }
    
    # Log user creation
    log_security_event(username, "user_created", "New user account created")

# ... existing code ...
def generate_secure_key(length=32):
    """Generate a secure key using entropy pool"""
    if not st.session_state.entropy_pool:
        # If entropy pool is empty, use os.urandom as fallback
        return os.urandom(length)
    
    # Mix entropy from pool with system randomness
    entropy_str = ''.join(random.sample(st.session_state.entropy_pool, min(100, len(st.session_state.entropy_pool))))
    entropy_hash = hashlib.sha512((entropy_str + str(time.time())).encode()).digest()
    
    # Use entropy as seed for random generator
    random.seed(entropy_hash)
    
    # Generate key
    key = bytearray(length)
    for i in range(length):
        key[i] = random.randint(0, 255)
    
    # Record key generation in history
    st.session_state.key_history.append({
        'timestamp': datetime.now(),
        'length': length,
        'entropy_used': len(entropy_str)
    })
    
    return bytes(key)

def collect_entropy(data):
    """Collect entropy for secure key generation"""
    # Hash the input data to extract entropy
    entropy_hash = hashlib.sha256(str(data).encode()).hexdigest()
    
    # Add to entropy pool, keeping pool size limited
    st.session_state.entropy_pool.append(entropy_hash)
    if len(st.session_state.entropy_pool) > ENTROPY_POOL_SIZE:
        st.session_state.entropy_pool.pop(0)  # Remove oldest entry
def render_dashboard():
    """Render the dashboard section with user statistics and metrics"""
    st.markdown("## Dashboard")
    
    # Create a multi-column layout for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # User security score
        user_score = st.session_state.users[st.session_state.username]["score"]
        score_color = "#4CAF50" if user_score >= 70 else "#FFC107" if user_score >= 40 else "#F44336"
        
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="metric-label">Security Score</p>
            <h2 class="metric-value" style="color: {score_color};">{user_score}/100</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Files count
        files_count = len(st.session_state.users[st.session_state.username]["files"])
        
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="metric-label">Your Files</p>
            <h2 class="metric-value">{files_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Shared files count
        shared_count = len(st.session_state.users[st.session_state.username]["shared_files"])
        
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="metric-label">Shared With You</p>
            <h2 class="metric-value">{shared_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("### Recent Activity")
    
    # Get user's access history
    access_history = st.session_state.users[st.session_state.username]["access_history"]
    
    if access_history:
        # Display the 5 most recent activities
        recent_activities = access_history[-5:]
        
        for activity in reversed(recent_activities):
            timestamp = activity.get("timestamp", "Unknown time")
            action = activity.get("action", "Unknown action")
            file_id = activity.get("file_id", "Unknown file")
            
            # Get file name if available
            file_name = "Unknown file"
            if file_id in st.session_state.users[st.session_state.username]["files"]:
                file_name = st.session_state.users[st.session_state.username]["files"][file_id]["filename"]
            elif file_id in st.session_state.users[st.session_state.username]["shared_files"]:
                file_name = st.session_state.users[st.session_state.username]["shared_files"][file_id]["filename"]
            
            # Format the activity message
            if action == "upload":
                icon = "📤"
                message = f"Uploaded file: {file_name}"
            elif action == "download":
                icon = "📥"
                message = f"Downloaded file: {file_name}"
            elif action == "share":
                icon = "🔗"
                message = f"Shared file: {file_name}"
            elif action == "delete":
                icon = "🗑️"
                message = f"Deleted file: {file_name}"
            else:
                icon = "🔄"
                message = f"{action.capitalize()}: {file_name}"
            
            st.markdown(f"""
            <div class="notification notification-info">
                <div style="display: flex; justify-content: space-between;">
                    <div>{icon} {message}</div>
                    <div style="color: #718096; font-size: 0.9em;">{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent activity to display")
    
    # Encryption statistics
    st.markdown("### Encryption Usage")
    
    # Prepare data for the chart
    methods = list(st.session_state.encryption_stats.keys())
    counts = list(st.session_state.encryption_stats.values())
    
    # Create a bar chart using Plotly
    fig = px.bar(
        x=methods,
        y=counts,
        labels={'x': 'Encryption Method', 'y': 'Usage Count'},
        color=methods,
        color_discrete_map={
            'AES': '#4e8df5',
            'RSA': '#f5924e',
            'Kyber': '#4ef58d',
            'NTRU': '#f54e8d',
            'ChaCha20': '#9d4ef5'
        }
    )
    
    fig.update_layout(
        title='Encryption Methods Usage',
        xaxis_title='Encryption Method',
        yaxis_title='Usage Count',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Security tips
    st.markdown("### Security Tips")
    
    security_tips = [
        "Regularly update your password to maintain account security.",
        "Use different encryption methods for different types of files.",
        "Enable two-factor authentication for additional security.",
        "Regularly check your access logs for any suspicious activity.",
        "Don't share sensitive files with users you don't trust."
    ]
    
    tip_index = random.randint(0, len(security_tips) - 1)
    
    st.markdown(f"""
    <div class="notification notification-warning">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.5em; margin-right: 10px;">💡</div>
            <div>{security_tips[tip_index]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
def render_my_files():
    """Render the user's files section"""
    st.markdown("## My Files")
    
    # Upload new file
    uploaded_file = st.file_uploader("Upload a new file", type=None, key="file_uploader")
    
    if uploaded_file is not None:
        # Read file data
        file_data = uploaded_file.read()
        
        # Analyze file for best encryption method
        recommended_method, reason, probabilities = analyze_file_for_encryption(file_data, uploaded_file.name)
        
        # Create columns for encryption options
        st.markdown("### Encryption Options")
        st.markdown(f"**Recommended method:** {recommended_method} - {reason}")
        
        # Display encryption method selection
        selected_method = st.selectbox(
            "Select encryption method",
            options=ENCRYPTION_METHODS,
            index=ENCRYPTION_METHODS.index(recommended_method)
        )
        
        # Encrypt and save button
        if st.button("Encrypt and Save"):
            with st.spinner("Encrypting file..."):
                # Generate a unique file ID
                file_id = f"file_{st.session_state.file_counter}"
                st.session_state.file_counter += 1
                
                # Encrypt the file
                encrypted_data, encryption_info = encrypt_file(file_data, selected_method, file_id)
                
                # Save file info
                st.session_state.users[st.session_state.username]["files"][file_id] = {
                    "filename": uploaded_file.name,
                    "encryption": selected_method,
                    "encrypted_data": encrypted_data,
                    "encryption_info": encryption_info,
                    "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_size": len(file_data)
                }
                
                # Update access history
                st.session_state.users[st.session_state.username]["access_history"].append({
                    "file_id": file_id,
                    "action": "upload",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Log file upload
                log_security_event(
                    st.session_state.username,
                    "file_upload",
                    f"Uploaded and encrypted file: {uploaded_file.name} using {selected_method}"
                )
                
                st.success(f"File encrypted and saved successfully using {selected_method}")
                st.rerun()
    
    # Display existing files
    if not st.session_state.users[st.session_state.username]["files"]:
        st.info("You haven't uploaded any files yet.")
        return
    
    # Create a search box
    search_query = st.text_input("Search files", "")
    
    # Filter files based on search query
    files = st.session_state.users[st.session_state.username]["files"]
    if search_query:
        files = {k: v for k, v in files.items() if search_query.lower() in v["filename"].lower()}
    
    # Create columns for the grid
    cols = st.columns(3)
    
    for i, (file_id, file_info) in enumerate(files.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="file-card">
                <h3>{file_info['filename']}</h3>
                <p>Encryption: <span class="{file_info['encryption'].lower()}-badge encryption-badge">{file_info['encryption']}</span></p>
                <p>Uploaded: {file_info['upload_date']}</p>
                <p>Size: {file_info['file_size'] / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # File actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download button
                if st.button(f"Download", key=f"download_{file_id}"):
                    try:
                        # Decrypt the file
                        decrypted_data = decrypt_file(
                            file_info['encrypted_data'],
                            file_info['encryption_info'],
                            file_info['encryption']
                        )
                        
                        # Create download link
                        b64 = base64.b64encode(decrypted_data).decode()
                        mime_type = "application/octet-stream"
                        href = f'<a href="data:{mime_type};base64,{b64}" download="{file_info["filename"]}">Download {file_info["filename"]}</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Log access
                        log_security_event(
                            st.session_state.username,
                            "file_access",
                            f"Downloaded file: {file_info['filename']}"
                        )
                        
                        # Update access history
                        st.session_state.users[st.session_state.username]["access_history"].append({
                            "file_id": file_id,
                            "action": "download",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                    except Exception as e:
                        st.error(f"Error downloading file: {str(e)}")
            
            with col2:
                # Share button
                if st.button(f"Share", key=f"share_{file_id}"):
                    st.session_state.selected_file = file_id
                    st.session_state.selected_file_type = "own"
                    st.session_state.sharing_mode = True
                    st.rerun()
            
            with col3:
                # Details button
                if st.button(f"Details", key=f"details_{file_id}"):
                    st.session_state.selected_file = file_id
                    st.session_state.selected_file_type = "own"
                    st.session_state.sharing_mode = False
                    st.rerun()
def simulate_qubit_key_generation(seed, length=32):
    """Simulate quantum key generation using qubits"""
    # In a real quantum system, this would use actual quantum hardware
    # For simulation, we'll use a deterministic but complex algorithm
    
    # Create a seed hash
    seed_hash = hashlib.sha256(str(seed).encode()).digest()
    
    # Convert to numpy array for "quantum" operations
    seed_array = np.frombuffer(seed_hash, dtype=np.uint8)
    
    # Simulate quantum superposition by creating a probability distribution
    probabilities = np.sin(seed_array / 255.0 * np.pi) ** 2
    
    # Simulate measurement collapse
    measurements = np.random.binomial(1, probabilities, size=length * 8)
    
    # Convert bit array to bytes
    result = bytearray(length)
    for i in range(length):
        for j in range(8):
            if measurements[i * 8 + j]:
                result[i] |= (1 << j)
    
    return bytes(result)

def check_password_strength(password):
    """Check password strength and return a score and feedback"""
    score = 0
    feedback = []
    
    # Length check
    if len(password) >= 12:
        score += 25
    elif len(password) >= 8:
        score += 15
        feedback.append("Consider using a longer password (12+ chars)")
    else:
        feedback.append("Password is too short")
    
    # Character variety checks
    if re.search(r'[A-Z]', password):
        score += 10
    else:
        feedback.append("Add uppercase letters")
        
    if re.search(r'[a-z]', password):
        score += 10
    else:
        feedback.append("Add lowercase letters")
        
    if re.search(r'[0-9]', password):
        score += 10
    else:
        feedback.append("Add numbers")
        
    if re.search(r'[^A-Za-z0-9]', password):
        score += 15
    else:
        feedback.append("Add special characters")
    
    # Common patterns check
    common_patterns = ['123', 'abc', 'qwerty', 'password', 'admin']
    for pattern in common_patterns:
        if pattern in password.lower():
            score -= 10
            feedback.append(f"Avoid common patterns like '{pattern}'")
            break
    
    # Determine strength level and color
    if score >= 70:
        level = "Strong"
        color = "#4CAF50"  # Green
    elif score >= 40:
        level = "Moderate"
        color = "#FFC107"  # Yellow
    else:
        level = "Weak"
        color = "#F44336"  # Red
    
    return {
        'score': score,
        'level': level,
        'color': color,
        'feedback': feedback
    }

def verify_login(username, password):
    """Verify login credentials with rate limiting"""
    # Check if user is locked out
    current_time = time.time()
    if username in st.session_state.login_attempts:
        attempts, lockout_time = st.session_state.login_attempts[username]
        if attempts >= MAX_LOGIN_ATTEMPTS and current_time - lockout_time < LOGIN_COOLDOWN:
            remaining = int(LOGIN_COOLDOWN - (current_time - lockout_time))
            logger.warning(f"Account {username} is locked. Try again in {remaining} seconds")
            return False, f"Too many failed attempts. Try again in {remaining} seconds"
    
    # Verify credentials
    if username in st.session_state.users and st.session_state.users[username]["password"] == password:
        # Reset login attempts on successful login
        st.session_state.login_attempts[username] = (0, current_time)
        
        # Log successful login
        log_security_event(username, "login_success", "Successful login")
        return True, "Login successful"
    else:
        # Increment failed login attempts
        if username in st.session_state.login_attempts:
            attempts, _ = st.session_state.login_attempts[username]
            st.session_state.login_attempts[username] = (attempts + 1, current_time)
        else:
            st.session_state.login_attempts[username] = (1, current_time)
        
        # Log failed login attempt
        log_security_event(username, "login_failure", "Failed login attempt")
        return False, "Invalid username or password"

def log_security_event(username, event_type, description):
    """Log security events for auditing"""
    event = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'username': username,
        'event_type': event_type,
        'description': description,
        'ip_address': '127.0.0.1'  # In a real app, get the actual IP
    }
    
    st.session_state.security_alerts.append(event)
    logger.info(f"Security event: {event_type} - {description} - User: {username}")

# ================ ENCRYPTION FUNCTIONS ================

def analyze_file_for_encryption(file_data, filename):
    """Analyze file to recommend best encryption method"""
    # Extract file extension
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    # Get file size
    file_size = len(file_data)
    
    # Initialize probabilities
    probabilities = {
        'AES': 0.25,
        'RSA': 0.25,
        'Kyber': 0.25,
        'NTRU': 0.15,
        'ChaCha20': 0.10
    }
    
    # Adjust based on file type
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        # Images - prefer symmetric encryption for speed
        probabilities['AES'] += 0.2
        probabilities['ChaCha20'] += 0.1
        probabilities['RSA'] -= 0.1
        reason = "Image files benefit from fast symmetric encryption"
    elif ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
        # Documents - balanced approach
        probabilities['AES'] += 0.1
        probabilities['RSA'] += 0.1
        probabilities['Kyber'] += 0.05
        reason = "Document files need balanced security and performance"
    elif ext in ['.txt', '.csv', '.json', '.xml']:
        # Text files - may contain sensitive data
        probabilities['RSA'] += 0.15
        probabilities['Kyber'] += 0.1
        probabilities['AES'] -= 0.05
        reason = "Text files may contain sensitive data requiring stronger encryption"
    elif ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
        # Archives - already compressed, need efficient encryption
        probabilities['AES'] += 0.15
        probabilities['ChaCha20'] += 0.15
        probabilities['RSA'] -= 0.1
        reason = "Archive files need efficient encryption due to their size and structure"
    elif ext in ['.exe', '.dll', '.so', '.bin']:
        # Executables - need integrity protection
        probabilities['AES'] += 0.1
        probabilities['NTRU'] += 0.1
        probabilities['Kyber'] += 0.05
        reason = "Executable files need strong integrity protection"
    else:
        # Unknown file type - use quantum-resistant as precaution
        probabilities['Kyber'] += 0.1
        probabilities['NTRU'] += 0.1
        reason = "Unknown file type, using quantum-resistant encryption as precaution"
    
    # Adjust based on file size
    if file_size > 10 * 1024 * 1024:  # > 10MB
        # Large files - prefer faster encryption
        probabilities['AES'] += 0.15
        probabilities['ChaCha20'] += 0.1
        probabilities['RSA'] -= 0.15
        probabilities['NTRU'] -= 0.05
        reason += " and fast encryption is preferred for large files"
    elif file_size < 1024:  # < 1KB
        # Small files - can use more intensive encryption
        probabilities['RSA'] += 0.1
        probabilities['Kyber'] += 0.05
        reason += " and small files can use more intensive encryption methods"
    
    # Normalize probabilities
    total = sum(probabilities.values())
    probabilities = {k: v/total for k, v in probabilities.items()}
    
    # Select method with highest probability
    recommended_method = max(probabilities, key=probabilities.get)
    
    return recommended_method, reason, probabilities

def encrypt_file(file_data, method, file_id):
    """Encrypt file using the selected method with enhanced security"""
    encryption_time = datetime.now()
    
    # Update encryption stats
    if method in st.session_state.encryption_stats:
        st.session_state.encryption_stats[method] += 1
    
    if method == 'AES':
        # Generate a secure key and initialization vector
        key = generate_secure_key(32)  # 256-bit key
        iv = os.urandom(16)
        
        # Store key info
        st.session_state.encryption_keys[file_id] = {
            'method': 'AES',
            'key': base64.b64encode(key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'timestamp': encryption_time,
            'key_size': len(key) * 8
        }
        
        # Pad the data to ensure it's a multiple of the block size
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(file_data) + padder.finalize()
        
        # Create and use the cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return base64 encoded encrypted data and encryption info
        return base64.b64encode(encrypted_data), {
            'key': base64.b64encode(key).decode(),
            'iv': base64.b64encode(iv).decode()
        }
    
    elif method == 'RSA':
        # For demonstration, using smaller key size for speed
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # Store key info
        st.session_state.encryption_keys[file_id] = {
            'method': 'RSA',
            'public_key': base64.b64encode(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )).decode(),
            'timestamp': encryption_time,
            'key_size': 2048
        }
        
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
        
        # Store key info
        st.session_state.encryption_keys[file_id] = {
            'method': 'Kyber',
            'key': base64.b64encode(key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'timestamp': encryption_time,
            'key_size': len(key) * 8,
            'quantum_resistant': True
        }
        
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
        
        # Store key info
        st.session_state.encryption_keys[file_id] = {
            'method': 'NTRU',
            'key': base64.b64encode(key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'timestamp': encryption_time,
            'key_size': len(key) * 8,
            'quantum_resistant': True
        }
        
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
    elif method == 'ChaCha20':
        # ChaCha20 is a modern stream cipher
        key = generate_secure_key(32)
        nonce = os.urandom(16)
        
        # Store key info
        st.session_state.encryption_keys[file_id] = {
            'method': 'ChaCha20',
            'key': base64.b64encode(key).decode(),
            'nonce': base64.b64encode(nonce).decode(),
            'timestamp': encryption_time,
            'key_size': len(key) * 8
        }
        
        # Create and use the cipher
        algorithm = algorithms.ChaCha20(key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(file_data)
        
        return base64.b64encode(encrypted_data), {
            'key': base64.b64encode(key).decode(),
            'nonce': base64.b64encode(nonce).decode()
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
    
    elif method == 'ChaCha20':
        key = base64.b64decode(encryption_info['key'])
        nonce = base64.b64decode(encryption_info['nonce'])
        encrypted_bytes = base64.b64decode(encrypted_data)
        
        algorithm = algorithms.ChaCha20(key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_bytes) + decryptor.finalize()
    
    else:
        raise ValueError(f"Unsupported encryption method: {method}")
def render_settings():
    """Render the settings section"""
    st.markdown("## Settings")
    
    # Create tabs for different settings categories
    settings_tab1, settings_tab2, settings_tab3 = st.tabs(["Security", "Appearance", "Advanced"])
    
    with settings_tab1:
        st.markdown("### Security Settings")
        
        # Password change
        st.markdown("#### Change Password")
        current_password = st.text_input("Current Password", type="password", key="current_password")
        new_password = st.text_input("New Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_password")
        
        if st.button("Change Password"):
            # Verify current password
            if st.session_state.users[st.session_state.username]["password"] != current_password:
                st.error("Current password is incorrect")
            elif new_password != confirm_password:
                st.error("New passwords do not match")
            elif not new_password:
                st.error("New password cannot be empty")
            else:
                # Check password strength
                strength = check_password_strength(new_password)
                
                if strength['score'] < 40:
                    st.warning(f"Password strength: {strength['level']} ({strength['score']}/100)")
                    st.markdown("<ul>", unsafe_allow_html=True)
                    for feedback in strength['feedback']:
                        st.markdown(f"<li>{feedback}</li>", unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)
                    
                    # Allow override with confirmation
                    if st.button("Use Weak Password Anyway"):
                        st.session_state.users[st.session_state.username]["password"] = new_password
                        st.success("Password changed successfully")
                        log_security_event(st.session_state.username, "password_change", "Password changed (weak)")
                else:
                    st.session_state.users[st.session_state.username]["password"] = new_password
                    st.success("Password changed successfully")
                    log_security_event(st.session_state.username, "password_change", "Password changed")
        
        # Entropy collection
        st.markdown("#### Entropy Collection")
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
        
        # Security logs
        st.markdown("#### Security Logs")
        if st.session_state.security_alerts:
            logs_df = pd.DataFrame(st.session_state.security_alerts[-10:])
            st.dataframe(logs_df)
        else:
            st.info("No security logs available")
    
    with settings_tab2:
        st.markdown("### Appearance Settings")
        
        # Dark mode toggle
        dark_mode = st.toggle("Dark Mode", st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            if dark_mode:
                st.markdown("""
                <script>
                    document.body.classList.add('dark-mode');
                </script>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <script>
                    document.body.classList.remove('dark-mode');
                </script>
                """, unsafe_allow_html=True)
            st.rerun()
        
        # Color theme selection
        st.markdown("#### Color Theme")
        theme_options = ["Blue (Default)", "Green", "Purple", "Orange"]
        selected_theme = st.selectbox("Select Theme", theme_options)
        
        # Preview the selected theme
        theme_colors = {
            "Blue (Default)": {"primary": "#4e8df5", "secondary": "#f5924e"},
            "Green": {"primary": "#4CAF50", "secondary": "#FFC107"},
            "Purple": {"primary": "#9C27B0", "secondary": "#FF9800"},
            "Orange": {"primary": "#FF5722", "secondary": "#2196F3"}
        }
        
        selected_colors = theme_colors[selected_theme]
        st.markdown(f"""
        <div style="display: flex; margin-top: 20px;">
            <div style="background-color: {selected_colors['primary']}; width: 100px; height: 100px; border-radius: 10px; margin-right: 20px; display: flex; justify-content: center; align-items: center; color: white;">
                Primary
            </div>
            <div style="background-color: {selected_colors['secondary']}; width: 100px; height: 100px; border-radius: 10px; display: flex; justify-content: center; align-items: center; color: white;">
                Secondary
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Apply Theme"):
            # In a real app, this would update CSS variables
            st.success(f"Theme changed to {selected_theme}")
            
            # Store in user preferences
            if 'appearance' not in st.session_state.user_preferences:
                st.session_state.user_preferences['appearance'] = {}
            
            st.session_state.user_preferences['appearance']['theme'] = selected_theme
            st.rerun()
    
    with settings_tab3:
        st.markdown("### Advanced Settings")
        
        # Encryption defaults
        st.markdown("#### Default Encryption Method")
        default_method = st.selectbox(
            "Select default encryption method",
            ['AES', 'RSA', 'Kyber', 'NTRU', 'ChaCha20']
        )
        
        if st.button("Set Default"):
            # Store in user preferences
            if 'encryption' not in st.session_state.user_preferences:
                st.session_state.user_preferences['encryption'] = {}
            
            st.session_state.user_preferences['encryption']['default_method'] = default_method
            st.success(f"Default encryption method set to {default_method}")
        
        # Quantum key simulation
        st.markdown("#### Quantum Key Simulation")
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
        
        # Export/Import settings
        st.markdown("#### Export/Import Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Settings"):
                # Create a JSON representation of user settings
                user_data = {
                    "username": st.session_state.username,
                    "preferences": st.session_state.user_preferences,
                    "encryption_stats": st.session_state.encryption_stats,
                    "score": st.session_state.users[st.session_state.username]['score']
                }
                
                # Convert to JSON string
                json_data = json.dumps(user_data, indent=2)
                
                # Create download link
                b64 = base64.b64encode(json_data.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="ciphercloud_settings.json">Download Settings</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            uploaded_settings = st.file_uploader("Import Settings", type=["json"])
            if uploaded_settings is not None:
                try:
                    settings_data = json.loads(uploaded_settings.read().decode())
                    
                    # Validate settings data
                    if "preferences" in settings_data:
                        st.session_state.user_preferences = settings_data["preferences"]
                        st.success("Settings imported successfully")
                    else:
                        st.error("Invalid settings file")
                except Exception as e:
                    st.error(f"Error importing settings: {str(e)}")

def render_analytics():
    """Render advanced analytics dashboard"""
    st.markdown("## Analytics Dashboard")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        total_files = sum(len(user_data['files']) for user_data in st.session_state.users.values())
        st.markdown(f"<p class='metric-label'>Total Files</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{total_files}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        total_users = len(st.session_state.users)
        st.markdown(f"<p class='metric-label'>Total Users</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{total_users}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        total_shares = sum(len(user_data['shared_files']) for user_data in st.session_state.users.values())
        st.markdown(f"<p class='metric-label'>Total Shares</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{total_shares}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        total_logs = len(st.session_state.access_logs)
        st.markdown(f"<p class='metric-label'>Total Activities</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{total_logs}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Encryption methods distribution
    st.markdown("### Encryption Methods Distribution")
    
    # Create DataFrame for visualization
    encryption_data = pd.DataFrame({
        'Method': list(st.session_state.encryption_stats.keys()),
        'Count': list(st.session_state.encryption_stats.values())
    })
    
    # Create interactive chart with Plotly
    fig = px.pie(
        encryption_data, 
        values='Count', 
        names='Method',
        title='Encryption Methods Distribution',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # User activity over time
    st.markdown("### User Activity Over Time")
    
    if st.session_state.access_logs:
        # Create DataFrame from logs
        logs_df = pd.DataFrame(st.session_state.access_logs)
        
        # Convert timestamp to datetime
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        
        # Group by day and action
        logs_df['date'] = logs_df['timestamp'].dt.date
        activity_by_date = logs_df.groupby(['date', 'action']).size().reset_index(name='count')
        
        # Create interactive time series chart
        fig = px.line(
            activity_by_date,
            x='date',
            y='count',
            color='action',
            title='User Activity Over Time',
            labels={'count': 'Number of Actions', 'date': 'Date'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No activity logs available for visualization")
    
    # User scores comparison
    st.markdown("### User Scores Comparison")
    
    # Create DataFrame for user scores
    user_scores = pd.DataFrame({
        'User': list(st.session_state.users.keys()),
        'Score': [user_data['score'] for user_data in st.session_state.users.values()]
    })
    
    # Create bar chart
    fig = px.bar(
        user_scores,
        x='User',
        y='Score',
        title='User Security Scores',
        labels={'Score': 'Security Score (0-100)'},
        color='Score',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # File size distribution
    st.markdown("### File Size Distribution")
    
    # Collect file sizes
    file_sizes = []
    file_types = []
    
    for username, user_data in st.session_state.users.items():
        for file_id, file_info in user_data['files'].items():
            file_sizes.append(file_info['size'])
            file_extension = os.path.splitext(file_info['filename'])[1].lower()
            file_types.append(file_extension if file_extension else 'unknown')
    
    if file_sizes:
        # Create DataFrame
        files_df = pd.DataFrame({
            'Size (KB)': [size / 1024 for size in file_sizes],
            'Type': file_types
        })
        
        # Create histogram
        fig = px.histogram(
            files_df,
            x='Size (KB)',
            color='Type',
            title='File Size Distribution',
            labels={'Size (KB)': 'File Size (KB)'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No files available for analysis")

def render_file_details(file_id, file_info, is_owner=True):
    """Render detailed view of a file"""
    st.markdown(f"## File Details: {file_info['filename']}")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### File Information")
        
        # Basic info
        st.markdown(f"**File ID:** {file_id}")
        st.markdown(f"**Filename:** {file_info['filename']}")
        st.markdown(f"**Upload Date:** {file_info['upload_date']}")
        st.markdown(f"**Size:** {file_info['size']} bytes ({file_info['size'] / 1024:.2f} KB)")
        
        # Encryption info
        st.markdown(f"**Encryption Method:** <span class='{file_info['encryption_method'].lower()}-badge encryption-badge'>{file_info['encryption_method']}</span>", unsafe_allow_html=True)
        
        # File type and icon
        file_extension = os.path.splitext(file_info['filename'])[1].lower()
        file_type = file_extension[1:] if file_extension else "unknown"
        
        # File preview (if possible)
        st.markdown("### Preview")
        
        # For demo purposes, just show a placeholder
        st.markdown(f"""
        <div style="background-color: #f5f7f9; padding: 20px; border-radius: 10px; text-align: center;">
            <p style="font-size: 3em; margin: 0;">📄</p>
            <p>{file_type.upper()} File</p>
            <p>Preview not available for encrypted files</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File actions
        st.markdown("### Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download", key=f"detail_download_{file_id}"):
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
            if is_owner and st.button("Share", key=f"detail_share_{file_id}"):
                st.session_state.sharing_file_id = file_id
                st.rerun()
        
        with col3:
            if is_owner and st.button("Delete", key=f"detail_delete_{file_id}"):
                del st.session_state.users[st.session_state.username]['files'][file_id]
                log_file_access(file_id, st.session_state.username, 'delete')
                st.success(f"File {file_id} deleted successfully.")
                st.session_state.viewing_file_id = None
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # File history and activity
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### Activity History")
        
        # Get file access history
        file_history = []
        for log in st.session_state.access_logs:
            if log['file_id'] == file_id:
                file_history.append(log)
        
        if file_history:
            for activity in reversed(file_history[-5:]):
                st.markdown(f"""
                <div style="padding: 10px; background-color: #f5f7f9; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #718096;">{activity['timestamp']}</p>
                    <p style="margin: 0; font-weight: bold;">{activity['username']} - {activity['action'].capitalize()}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No activity recorded for this file")
        
        # File integrity
        st.markdown("### File Integrity")
        
        if file_id in st.session_state.merkle_trees:
            merkle_root = st.session_state.merkle_trees[file_id]['root']
            st.markdown(f"**Merkle Root:** `{merkle_root[:10]}...`")
            
            if st.button("Verify Integrity"):
                try:
                    encrypted_data = file_info['data']
                    decrypted_data = decrypt_file(
                        encrypted_data,
                        file_info['encryption_info'],
                        file_info['encryption_method']
                    )
                    
                    integrity_ok, message = verify_file_integrity(decrypted_data, file_id)
                    
                    if integrity_ok:
                        st.success("File integrity verified successfully")
                    else:
                        st.error(message)
                        
                except Exception as e:
                    st.error(f"Error verifying integrity: {str(e)}")
        else:
            st.warning("No integrity data available for this file")
        
        # File comments
        st.markdown("### Comments")
        
        # Initialize comments for this file if needed
        if file_id not in st.session_state.file_comments:
            st.session_state.file_comments[file_id] = []
        
        # Display existing comments
        for comment in st.session_state.file_comments[file_id]:
            st.markdown(f"""
            <div style="padding: 10px; background-color: #f5f7f9; border-radius: 5px; margin-bottom: 10px;">
                <p style="margin: 0; color: #718096;">{comment['timestamp']} - {comment['username']}</p>
                <p style="margin: 0;">{comment['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add new comment
        new_comment = st.text_area("Add a comment", key=f"comment_{file_id}")
        if st.button("Post Comment"):
            if new_comment:
                comment = {
                    'username': st.session_state.username,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': new_comment
                }
                st.session_state.file_comments[file_id].append(comment)
                st.success("Comment added")
                st.rerun()
            else:
                st.warning("Comment cannot be empty")
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_notifications():
    """Render user notifications"""
    if st.session_state.notifications:
        with st.sidebar:
            st.markdown("### Notifications")
            
            for i, notification in enumerate(st.session_state.notifications):
                notification_type = notification.get('type', 'info')
                
                if notification_type == 'info':
                    class_name = 'notification-info'
                elif notification_type == 'success':
                    class_name = 'notification-success'
                elif notification_type == 'warning':
                    class_name = 'notification-warning'
                elif notification_type == 'error':
                    class_name = 'notification-error'
                
                st.markdown(f"""
                <div class="notification {class_name}">
                    <p style="margin: 0; font-weight: bold;">{notification['title']}</p>
                    <p style="margin: 0;">{notification['message']}</p>
                    <p style="margin: 0; font-size: 0.8em; color: #718096;">{notification['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Dismiss", key=f"dismiss_{i}"):
                    st.session_state.notifications.pop(i)
                    st.rerun()

def add_notification(title, message, notification_type='info'):
    """Add a notification to the user's notification list"""
    notification = {
        'title': title,
        'message': message,
        'type': notification_type,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.notifications.append(notification)

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
        st.markdown("### Navigation")
        app_mode = st.radio("", ["Dashboard", "My Files", "Shared Files", "Upload File", "File Permissions", "Analytics", "Settings"])
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    
    # Main content
    st.markdown("<h1 style='color: #4e8df5;'>CipherCloud</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2em;'>Secure File Sharing with Advanced Encryption</p>", unsafe_allow_html=True)
    
    # Render notifications
    render_notifications()
    
    # Check if viewing a specific file
    if hasattr(st.session_state, 'viewing_file_id') and st.session_state.viewing_file_id:
        file_id = st.session_state.viewing_file_id
        
        # Check if file is in user's files
        if file_id in st.session_state.users[st.session_state.username]['files']:
            file_info = st.session_state.users[st.session_state.username]['files'][file_id]
            render_file_details(file_id, file_info, is_owner=True)
        # Check if file is in shared files
        elif file_id in st.session_state.users[st.session_state.username]['shared_files']:
            file_info = st.session_state.users[st.session_state.username]['shared_files'][file_id]
            owner = file_info['owner']
            owner_file_info = st.session_state.users[owner]['files'][file_id]
            render_file_details(file_id, owner_file_info, is_owner=False)
        else:
            st.error("File not found")
            st.session_state.viewing_file_id = None
            st.rerun()
        
        if st.button("Back to List"):
            st.session_state.viewing_file_id = None
            st.rerun()
    else:
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
        elif app_mode == "Analytics":
            render_analytics()
        elif app_mode == "Settings":
            render_settings()

# ================ MAIN APPLICATION LOGIC ================

def main():
    """Main application entry point"""
    # Load custom CSS
    load_custom_css()
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()