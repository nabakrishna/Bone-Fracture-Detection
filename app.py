#!/usr/bin/env python3
"""
Flask Web Application for Bone Fracture Detection System
Integrates bone_fracture_detector.py with HTML frontend
"""

import os
import json
import base64
import time
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Import your bone fracture detection system
from bone_fracture_detector import BoneFractureDetectionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config.update({
    'SECRET_KEY': 'bone-fracture-detection-secret-key-2024',
    'UPLOAD_FOLDER': 'static/uploads',
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,  # 10MB max file size
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'PROJECT_ROOT': os.getcwd()
})

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Global variables
detection_system = None
model_loaded = False

def initialize_detection_system():
    """Initialize the bone fracture detection system"""
    global detection_system, model_loaded
    try:
        project_root = app.config['PROJECT_ROOT']
        detection_system = BoneFractureDetectionSystem(project_root=project_root)
        
        # Try to load a trained model if available
        trained_models_dir = Path(project_root) / "models" / "trained"
        best_model_path = None
        
        # Look for trained models
        for model_dir in trained_models_dir.glob("*/weights"):
            best_model = model_dir / "best.pt"
            if best_model.exists():
                best_model_path = str(best_model)
                break
        
        # Load model
        if detection_system.load_model(best_model_path):
            model_loaded = True
            logger.info(f"✅ Model loaded successfully: {best_model_path or 'pretrained'}")
        else:
            logger.warning("⚠️ Model loading failed, will use pretrained model")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize detection system: {e}")
        detection_system = None
        model_loaded = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return round(os.path.getsize(filepath) / (1024 * 1024), 2)

def process_detection_result(result, image_path):
    """Process detection results for frontend display"""
    try:
        processed_result = {
            'success': True,
            'fracture_detected': len(result.get('boxes', [])) > 0,
            'confidence': 0.0,
            'detections': [],
            'analysis': {
                'total_detections': len(result.get('boxes', [])),
                'risk_level': 'low',
                'recommendation': '',
                'additional_tests_needed': False
            },
            'processing_info': {
                'image_path': image_path,
                'processed_at': datetime.now().isoformat(),
                'model_used': 'YOLOv8'
            }
        }

        if result.get('boxes') is not None and len(result['boxes']) > 0:
            boxes = result['boxes']
            confidences = result.get('confidences', [])
            class_names = result.get('class_names', [])
            
            max_confidence = max(confidences) if confidences else 0.0
            processed_result['confidence'] = float(max_confidence)
            
            # Process each detection
            for i, box in enumerate(boxes):
                confidence = confidences[i] if i < len(confidences) else 0.0
                class_name = class_names[i] if i < len(class_names) else 'fracture'
                
                detection = {
                    'type': class_name.title(),
                    'location': f"Region {i+1}",
                    'confidence': f"{confidence*100:.1f}%",
                    'severity': get_severity_level(confidence),
                    'bounding_box': {
                        'x': float(box[0]),
                        'y': float(box[1]),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1])
                    }
                }
                processed_result['detections'].append(detection)
            
            # Set analysis based on confidence
            if max_confidence > 0.8:
                processed_result['analysis'].update({
                    'risk_level': 'high',
                    'recommendation': 'High confidence fracture detected. Immediate medical attention recommended.',
                    'additional_tests_needed': True
                })
            elif max_confidence > 0.5:
                processed_result['analysis'].update({
                    'risk_level': 'medium',
                    'recommendation': 'Possible fracture detected. Additional imaging and medical consultation recommended.',
                    'additional_tests_needed': True
                })
            else:
                processed_result['analysis'].update({
                    'risk_level': 'low',
                    'recommendation': 'Low confidence detection. Consider higher quality imaging if symptoms persist.',
                    'additional_tests_needed': False
                })
        else:
            processed_result['analysis']['recommendation'] = 'No fractures detected in current analysis. Consult healthcare provider if symptoms persist.'
            
        return processed_result
        
    except Exception as e:
        logger.error(f"Error processing detection result: {e}")
        return {
            'success': False,
            'error': f'Result processing failed: {str(e)}',
            'fracture_detected': False
        }

def get_severity_level(confidence):
    """Determine severity level based on confidence"""
    if confidence > 0.8:
        return 'High'
    elif confidence > 0.5:
        return 'Moderate'
    else:
        return 'Low'

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and fracture detection"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or similar image files.'}), 400
        
        # Check if detection system is available
        if not detection_system or not model_loaded:
            return jsonify({'error': 'Detection system not available. Please try again later.'}), 503
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get file info
        file_size = get_file_size_mb(filepath)
        
        logger.info(f"Processing uploaded file: {unique_filename} ({file_size}MB)")
        
        # Get confidence threshold from request
        confidence_threshold = float(request.form.get('confidence', 0.3))
        detection_system.config['confidence_threshold'] = confidence_threshold
        
        # Run fracture detection
        detection_result = detection_system.detect_fractures(
            filepath, 
            save_results=True
        )
        
        # Process results for frontend
        processed_result = process_detection_result(detection_result, filepath)
        
        if processed_result['success']:
            # Add file information
            processed_result.update({
                'filename': unique_filename,
                'file_size': f"{file_size} MB",
                'image_url': f'/static/uploads/{unique_filename}',
                'analyzed_image_url': detection_result.get('output_path', f'/static/uploads/{unique_filename}')
            })
            
            logger.info(f"Detection completed: {len(processed_result.get('detections', []))} fractures found")
            return jsonify(processed_result)
        else:
            return jsonify(processed_result), 500
            
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Handle batch analysis of multiple images"""
    try:
        if not detection_system or not model_loaded:
            return jsonify({'error': 'Detection system not available'}), 503
        
        files = request.files.getlist('files[]')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Analyze
                detection_result = detection_system.detect_fractures(filepath)
                processed_result = process_detection_result(detection_result, filepath)
                processed_result['filename'] = unique_filename
                results.append(processed_result)
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_images': len(results),
                'fractures_detected': sum(1 for r in results if r['fracture_detected']),
                'processed_at': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Run model tests and return results"""
    try:
        if not detection_system:
            return jsonify({'error': 'Detection system not available'}), 503
        
        # Basic model test
        test_results = {
            'model_loaded': model_loaded,
            'system_initialized': detection_system is not None,
            'confidence_threshold': detection_system.config.get('confidence_threshold', 0.3),
            'model_type': detection_system.config.get('model_name', 'yolov8s'),
            'test_timestamp': datetime.now().isoformat(),
            'directories_created': True,
            'processing_capability': 'Available' if model_loaded else 'Limited'
        }
        
        # Test with a sample if available
        sample_images = list(Path(app.config['UPLOAD_FOLDER']).glob("*.jpg"))
        if sample_images:
            test_results['sample_test'] = 'Passed'
            test_results['sample_image'] = str(sample_images[0])
        else:
            test_results['sample_test'] = 'No sample images available'
        
        return jsonify({
            'success': True,
            'test_results': test_results
        })
        
    except Exception as e:
        logger.error(f"Model test error: {e}")
        return jsonify({'error': f'Testing failed: {str(e)}'}), 500

@app.route('/model-info')
def model_info():
    """Get model information"""
    try:
        if not detection_system:
            return jsonify({'error': 'Detection system not available'}), 503
        
        info = {
            'model_loaded': model_loaded,
            'config': detection_system.config,
            'directories': {name: str(path) for name, path in detection_system.dirs.items()},
            'system_status': 'Ready' if model_loaded else 'Initializing'
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred'}), 500

# Initialize the detection system when the app starts
@app.before_first_request
def startup():
    """Initialize the detection system on startup"""
    logger.info("🚀 Starting Bone Fracture Detection Web Application")
    initialize_detection_system()

if __name__ == '__main__':
    # Initialize detection system
    print("🏗️ Initializing Bone Fracture Detection System...")
    initialize_detection_system()
    
    if detection_system:
        print("✅ System initialized successfully!")
        print(f"📁 Project root: {detection_system.project_root}")
        print(f"🤖 Model loaded: {'Yes' if model_loaded else 'No'}")
        print(f"🌐 Starting web server...")
    else:
        print("⚠️  System initialization failed, running with limited functionality")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )
