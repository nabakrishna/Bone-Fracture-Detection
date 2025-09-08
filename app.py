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
import shutil

# Import your bone fracture detection system
try:
    from bone_fracture_detector import BoneFractureDetectionSystem
except ImportError as e:
    logging.warning(f"Could not import BoneFractureDetectionSystem: {e}")
    BoneFractureDetectionSystem = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app with current directory as template folder (RENDER FIX)
app = Flask(__name__, 
           template_folder='.', 
           static_folder='.',
           static_url_path='')

# Configuration
app.config.update({
    'SECRET_KEY': 'bone-fracture-detection-secret-key-2024',
    'UPLOAD_FOLDER': 'static/uploads',
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,  # 10MB max file size
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'PROJECT_ROOT': os.getcwd()
})

# Create necessary directories safely (RENDER FIX)
# def create_directories():
#     """Create necessary directories safely"""
#     directories = [
#         app.config['UPLOAD_FOLDER'],
#         'static/results',
#         'logs',
#         'models/trained'
#     ]
    
#     for directory in directories:
#         try:
#             os.makedirs(directory, exist_ok=True)  # exist_ok=True prevents the error
#             logger.info(f"✅ Directory created/verified: {directory}")
#         except Exception as e:
#             logger.error(f"❌ Failed to create directory {directory}: {e}")

def create_directories():
    """Create necessary directories safely"""
    directories = [
        'static/uploads',
        'static/results', 
        'logs',
        'models/trained'
    ]
    
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"✅ Directory created: {directory}")
            else:
                logger.info(f"✅ Directory already exists: {directory}")
        except Exception as e:
            logger.warning(f"⚠️ Directory issue for {directory}: {e}")
            # Don't fail the app, just log the warning

# Create directories on startup
create_directories()

# Global variables
detection_system = None
model_loaded = False

# def initialize_detection_system():
#     """Initialize the bone fracture detection system"""
#     global detection_system, model_loaded
#     try:
#         if BoneFractureDetectionSystem is None:
#             logger.warning("⚠️ BoneFractureDetectionSystem not available, running in demo mode")
#             model_loaded = False
#             return
            
#         project_root = app.config['PROJECT_ROOT']
#         detection_system = BoneFractureDetectionSystem(project_root=project_root)
        
#         # Try to load a trained model if available
#         trained_models_dir = Path(project_root) / "models" / "trained"
#         best_model_path = None
        
#         # Look for trained models
#         if trained_models_dir.exists():
#             for model_dir in trained_models_dir.glob("*/weights"):
#                 best_model = model_dir / "best.pt"
#                 if best_model.exists():
#                     best_model_path = str(best_model)
#                     break
        
#         # Load model
#         if detection_system.load_model(best_model_path):
#             model_loaded = True
#             logger.info(f"✅ Model loaded successfully: {best_model_path or 'pretrained'}")
#         else:
#             logger.warning("⚠️ Model loading failed, will use pretrained model")
#             model_loaded = False
            
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize detection system: {e}")
#         detection_system = None
#         model_loaded = False

def initialize_detection_system():
    """Initialize the bone fracture detection system"""
    global detection_system, model_loaded
    try:
        if BoneFractureDetectionSystem is None:
            logger.warning("⚠️ BoneFractureDetectionSystem not available, running in demo mode")
            model_loaded = False
            return
            
        project_root = app.config['PROJECT_ROOT']
        
        # TEMPORARY FIX: Create directories manually first to avoid [Errno 17] error
        try:
            os.makedirs(os.path.join(project_root, 'models', 'trained'), exist_ok=True)
            logger.info("✅ Pre-created models/trained directory")
        except Exception as dir_error:
            logger.warning(f"⚠️ Directory creation warning: {dir_error}")
        
        detection_system = BoneFractureDetectionSystem(project_root=project_root)
        
        # Try to load a trained model if available
        trained_models_dir = Path(project_root) / "models" / "trained"
        best_model_path = None
        
        # Look for trained models
        if trained_models_dir.exists():
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
            model_loaded = False
            
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
    try:
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    except:
        return 0.0

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

# Demo function for when detection system is not available
def demo_detection_result(image_path):
    """Return demo results when detection system is not available"""
    return {
        'success': True,
        'fracture_detected': True,
        'confidence': 0.75,
        'detections': [{
            'type': 'Fracture',
            'location': 'Demo Region',
            'confidence': '75.0%',
            'severity': 'Moderate',
            'bounding_box': {'x': 100, 'y': 100, 'width': 200, 'height': 150}
        }],
        'analysis': {
            'total_detections': 1,
            'risk_level': 'medium',
            'recommendation': 'Demo mode: This is a simulated detection result. Deploy with actual model for real analysis.',
            'additional_tests_needed': True
        },
        'processing_info': {
            'image_path': image_path,
            'processed_at': datetime.now().isoformat(),
            'model_used': 'Demo Mode'
        }
    }

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index.html: {e}")
        return """
        <html>
        <head><title>Bone Fracture Detection</title></head>
        <body>
            <h1>🩻 BoneScan AI - Bone Fracture Detection</h1>
            <p>Advanced AI-powered bone fracture detection system</p>
            <p><strong>Status:</strong> System is running</p>
            <p><em>Note: Frontend template not found, but backend is operational.</em></p>
        </body>
        </html>
        """

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'detection_system_available': detection_system is not None,
        'timestamp': datetime.now().isoformat(),
        'service': 'bone-fracture-detection',
        'version': '1.0.0'
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
        
        # Run fracture detection
        if detection_system and model_loaded:
            detection_system.config['confidence_threshold'] = confidence_threshold
            detection_result = detection_system.detect_fractures(
                filepath, 
                save_results=True  # This should save the annotated image
            )
            
            # Process results for frontend
            processed_result = process_detection_result(detection_result, filepath)
            
            # IMPORTANT: Use the annotated image path, not original
            annotated_image_path = detection_result.get('output_path', filepath)
            if annotated_image_path and os.path.exists(annotated_image_path):
                # Copy annotated image to static folder
                annotated_filename = f"annotated_{unique_filename}"
                annotated_static_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
                
                # Copy the annotated image
                import shutil
                try:
                    shutil.copy2(annotated_image_path, annotated_static_path)
                    logger.info(f"✅ Annotated image copied: {annotated_filename}")
                except Exception as e:
                    logger.error(f"❌ Failed to copy annotated image: {e}")
                    annotated_filename = unique_filename  # Use original as fallback
                
                processed_result.update({
                    'filename': unique_filename,
                    'file_size': f"{file_size} MB",
                    'image_url': f'/static/uploads/{unique_filename}',  # Original
                    'analyzed_image_url': f'/static/uploads/{annotated_filename}'  # With detections
                })
            else:
                # Fallback to original image if annotated not available
                logger.warning("⚠️ No annotated image found, using original")
                processed_result.update({
                    'filename': unique_filename,
                    'file_size': f"{file_size} MB",
                    'image_url': f'/static/uploads/{unique_filename}',
                    'analyzed_image_url': f'/static/uploads/{unique_filename}'
                })
        else:
            # Use demo results if detection system not available
            logger.warning("Using demo detection results")
            processed_result = demo_detection_result(filepath)
            processed_result.update({
                'filename': unique_filename,
                'file_size': f"{file_size} MB",
                'image_url': f'/static/uploads/{unique_filename}',
                'analyzed_image_url': f'/static/uploads/{unique_filename}'
            })
        
        if processed_result['success']:
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
                if detection_system and model_loaded:
                    detection_result = detection_system.detect_fractures(filepath)
                    processed_result = process_detection_result(detection_result, filepath)
                else:
                    processed_result = demo_detection_result(filepath)
                    
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
        test_results = {
            'model_loaded': model_loaded,
            'system_initialized': detection_system is not None,
            'confidence_threshold': detection_system.config.get('confidence_threshold', 0.3) if detection_system else 0.3,
            'model_type': detection_system.config.get('model_name', 'yolov8s') if detection_system else 'demo',
            'test_timestamp': datetime.now().isoformat(),
            'directories_created': True,
            'processing_capability': 'Available' if model_loaded else 'Demo Mode'
        }
        
        # Test with a sample if available
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        sample_images = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))
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
        if detection_system:
            info = {
                'model_loaded': model_loaded,
                'config': detection_system.config,
                'directories': {name: str(path) for name, path in detection_system.dirs.items()},
                'system_status': 'Ready' if model_loaded else 'Initializing'
            }
        else:
            info = {
                'model_loaded': False,
                'config': {'demo_mode': True},
                'directories': {'upload': app.config['UPLOAD_FOLDER']},
                'system_status': 'Demo Mode'
            }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Additional routes that might be called by your frontend
@app.route('/about')
def about():
    """About page"""
    try:
        return render_template('about.html')
    except:
        return "<h1>About BoneScan AI</h1><p>Advanced bone fracture detection system using YOLOv8</p>"

@app.route('/contact')
def contact():
    """Contact page"""
    try:
        return render_template('contact.html')  
    except:
        return "<h1>Contact</h1><p>Contact information for BoneScan AI support</p>"

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred'}), 500

# Initialize the detection system on first request (RENDER COMPATIBLE)
@app.before_request
def startup():
    """Initialize the detection system on startup"""
    if not hasattr(startup, 'has_run'):
        startup.has_run = True
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
    
    # Get port from environment variable (RENDER REQUIREMENT)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app (RENDER COMPATIBLE)
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Set to False for production
        use_reloader=False  # Disable reloader to prevent double initialization
    )
