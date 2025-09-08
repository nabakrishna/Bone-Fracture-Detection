##!/usr/bin/env python3
"""
Complete Bone Fracture Detection System
Single file solution with all functionality included
"""

import os
import sys
import yaml
import shutil
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from sklearn.model_selection import train_test_split

# Install required packages if not available
def install_requirements():
    """Install required packages"""
    packages = [
        'ultralytics>=8.0.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'pillow>=9.5.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'pandas>=2.0.0',
        'pyyaml>=6.0',
        'scikit-learn>=1.3.0'
    ]
    
    try:
        import ultralytics
        print("✅ Required packages already installed")
    except ImportError:
        print("📦 Installing required packages...")
        for package in packages:
            os.system(f"pip install {package}")
        print("✅ All packages installed!")

class BoneFractureDetectionSystem:
    """Complete Bone Fracture Detection System"""
    
    def __init__(self, project_root: str = None):
        """Initialize the detection system"""
        
        # Project configuration
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.setup_directories()
        self.setup_config()
        
        # Initialize components
        self.model = None
        
        print(f"🏗️  Bone Fracture Detection System initialized at: {self.project_root}")
    
    def setup_directories(self):
        """Setup all project directories"""
        
        # Define directory structure
        self.dirs = {
            'data': self.project_root / "data",
            'raw_images': self.project_root / "data" / "raw" / "images",
            'raw_labels': self.project_root / "data" / "raw" / "labels",
            'processed': self.project_root / "data" / "processed",
            'models': self.project_root / "models",
            'pretrained': self.project_root / "models" / "pretrained",
            'trained': self.project_root / "models" / "trained",
            'results': self.project_root / "results",
            'predictions': self.project_root / "results" / "predictions",
            'metrics': self.project_root / "results" / "metrics"
        }
        
        # Create all directories
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created successfully!")
    
    def setup_config(self):
        """Setup configuration parameters"""
        
        self.config = {
            # Training parameters
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.01,
            
            # Model configuration
            'model_name': 'yolov8s',
            'confidence_threshold': 0.3, #change from 0.5 to 0.01
            'iou_threshold': 0.45,
            
            # Data split ratios
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            
            # Class names (modify according to your dataset)
            'class_names': ['fracture']
        }
    
    def create_data_yaml(self) -> str:
        """Create data.yaml file for YOLO training"""
        
        data_yaml_content = {
            'path': str(self.dirs['processed']),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.config['class_names']),
            'names': {i: name for i, name in enumerate(self.config['class_names'])}
        }
        
        yaml_path = self.dirs['processed'] / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        print(f"✅ Created data.yaml at {yaml_path}")
        return str(yaml_path)
    
    def split_dataset(self) -> bool:
        """Split dataset into train, validation, and test sets"""
        
        images_dir = self.dirs['raw_images']
        labels_dir = self.dirs['raw_labels']
        output_dir = self.dirs['processed']
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
        
        if not image_files:
            print(f"❌ No image files found in {images_dir}")
            print(f"   Please place your images in: {images_dir}")
            print(f"   Please place your labels in: {labels_dir}")
            return False
        
        print(f"📊 Found {len(image_files)} images")
        
        # Split into train, val, test
        train_files, temp_files = train_test_split(
            image_files, 
            train_size=self.config['train_ratio'], 
            random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, 
            train_size=self.config['val_ratio']/(1-self.config['train_ratio']), 
            random_state=42
        )
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy files
        splits = {'train': train_files, 'val': val_files, 'test': test_files}
        
        for split_name, files in splits.items():
            copied_images = 0
            copied_labels = 0
            
            for img_file in files:
                # Copy image
                shutil.copy2(img_file, output_dir / split_name / 'images' / img_file.name)
                copied_images += 1
                
                # Copy corresponding label file
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, output_dir / split_name / 'labels' / label_file.name)
                    copied_labels += 1
            
            print(f"   {split_name.capitalize()}: {copied_images} images, {copied_labels} labels")
        
        print("✅ Dataset split completed successfully!")
        return True
    
    def validate_dataset(self) -> bool:
        """Validate dataset structure and files"""
        
        data_dir = self.dirs['processed']
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        
        for dir_path in required_dirs:
            full_path = data_dir / dir_path
            if not full_path.exists():
                print(f"❌ Missing directory: {full_path}")
                return False
        
        # Check if data.yaml exists
        if not (data_dir / 'data.yaml').exists():
            print("❌ Missing data.yaml file")
            return False
        
        print("✅ Dataset structure is valid")
        return True
    
    def prepare_dataset(self) -> bool:
        """Complete dataset preparation pipeline"""
        
        print("📊 Preparing dataset...")
        
        # Check if raw data exists
        if not self.dirs['raw_images'].exists() or not any(self.dirs['raw_images'].iterdir()):
            print(f"❌ No images found in: {self.dirs['raw_images']}")
            print("   Please place your images and labels in the raw data directories")
            return False
        
        # Split dataset
        if not self.split_dataset():
            return False
        
        # Create data.yaml
        self.create_data_yaml()
        
        # Validate dataset
        return self.validate_dataset()
    
    def load_model(self, model_path: str = None) -> bool:
        """Load YOLO model"""
        
        try:
            from ultralytics import YOLO
            
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                print(f"✅ Loaded custom model from {model_path}")
            else:
                self.model = YOLO(f"{self.config['model_name']}.pt")
                print(f"✅ Loaded pretrained {self.config['model_name']} model")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def train_model(self, **kwargs) -> bool:
        """Train the bone fracture detection model"""
        
        print("🎯 Training bone fracture detection model...")
        
        # Load model
        if not self.load_model():
            return False
        
        # Check dataset
        data_yaml_path = self.dirs['processed'] / 'data.yaml'
        if not data_yaml_path.exists():
            print("❌ data.yaml not found. Please prepare dataset first.")
            return False
        
        # Training parameters
        train_params = {
            'data': str(data_yaml_path),
            'epochs': kwargs.get('epochs', self.config['epochs']),
            'batch': kwargs.get('batch', self.config['batch_size']),
            'imgsz': kwargs.get('imgsz', self.config['img_size']),
            'lr0': kwargs.get('lr0', self.config['learning_rate']),
            'project': str(self.dirs['trained']),
            'name': 'bone_fracture_detector',
            'save_period': 10,
            'device': 'cpu',  # Change to 'cuda' if GPU available
            'patience': 20,
            'save': True,
            'plots': True
        }
        
        print("🚀 Starting training with parameters:")
        for key, value in train_params.items():
            print(f"   {key}: {value}")
        
        try:
            results = self.model.train(**train_params)
            print("✅ Training completed successfully!")
            
            # Save best model path
            best_model_path = self.dirs['trained'] / 'bone_fracture_detector' / 'weights' / 'best.pt'
            if best_model_path.exists():
                print(f"💾 Best model saved at: {best_model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    # def detect_fractures(self, image_path: str, model_path: str = None, save_results: bool = True) -> Dict:
    #     """Detect fractures in a single image"""
        
    #     # Load model if not already loaded
    #     if not self.model:
    #         if not self.load_model(model_path):
    #             return {}
        
    #     try:
    #         # Run inference
    #         results = self.model(
    #             image_path, 
    #             conf=self.config['confidence_threshold'], 
    #             iou=self.config['iou_threshold']
    #         )
            
    #         # Extract results
    #         result = results[0]
    #         detections = {
    #             'image_path': image_path,
    #             'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
    #             'confidences': result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
    #             'classes': result.boxes.cls.cpu().numpy() if result.boxes is not None else [],
    #             'class_names': [result.names[int(cls)] for cls in result.boxes.cls] if result.boxes is not None else []
    #         }
            
    #         print(f"🔍 Detected {len(detections['boxes'])} fractures in {Path(image_path).name}")
            
    #         if save_results and len(detections['boxes']) > 0:
    #             # Save annotated image
    #             output_path = self.dirs['predictions'] / f"detected_{Path(image_path).name}"
    #             annotated_img = result.plot()
    #             cv2.imwrite(str(output_path), annotated_img)
    #             detections['output_path'] = str(output_path)
    #             print(f"💾 Saved results to: {output_path}")
            
    #         return detections
            
    #     except Exception as e:
    #         print(f"❌ Error during detection: {e}")
    #         return {}

    def detect_fractures(self, image_path: str, model_path: str = None, save_results: bool = True) -> Dict:
        """Detect fractures in a single image"""
    
        # Load model if not already loaded
        if not self.model:
            if not self.load_model(model_path):
                return {'boxes': [], 'confidences': [], 'class_names': [], 'output_path': None}
    
        try:
            # Run inference
            results = self.model(
                image_path, 
                conf=self.config['confidence_threshold'], 
                iou=self.config['iou_threshold']
            )
        
            # Extract results
            result = results[0]
        
            # Convert numpy arrays to lists for JSON compatibility
            boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
            confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []
            classes = result.boxes.cls.cpu().numpy().tolist() if result.boxes is not None else []
            class_names = [result.names[int(cls)] for cls in result.boxes.cls] if result.boxes is not None else []
        
            detections = {
                'image_path': image_path,
                'boxes': boxes,
                'confidences': confidences,
                'classes': classes,
                'class_names': class_names,
                'output_path': None  # Initialize as None
            }
        
            print(f"🔍 Detected {len(boxes)} fractures in {Path(image_path).name}")
        
            # Save annotated image
            if save_results:
                try:
                    # Ensure predictions directory exists
                    self.dirs['predictions'].mkdir(parents=True, exist_ok=True)
                
                    # Create output path
                    output_path = self.dirs['predictions'] / f"detected_{Path(image_path).name}"
                
                    # Generate annotated image with bounding boxes
                    annotated_img = result.plot()
                
                    # Convert RGB to BGR for cv2 (YOLO outputs RGB, cv2 expects BGR)
                    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                
                    # Save the image
                    success = cv2.imwrite(str(output_path), annotated_img_bgr)
                
                    if success:
                        detections['output_path'] = str(output_path)
                        print(f"💾 Saved annotated image to: {output_path}")
                    else:
                        print(f"❌ Failed to save image to: {output_path}")
                    
                except Exception as save_error:
                    print(f"❌ Error saving annotated image: {save_error}")
        
            return detections
        
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return {'boxes': [], 'confidences': [], 'class_names': [], 'output_path': None}

    
    def batch_detect(self, images_path: str, model_path: str = None) -> List[Dict]:
        """Detect fractures in multiple images"""
        
        images_dir = Path(images_path)
        
        if not images_dir.exists():
            print(f"❌ Directory not found: {images_dir}")
            return []
        
        # Get all image files
        image_files = (list(images_dir.glob("*.jpg")) + 
                      list(images_dir.glob("*.png")) + 
                      list(images_dir.glob("*.jpeg")))
        
        if not image_files:
            print(f"❌ No images found in {images_dir}")
            return []
        
        print(f"🔍 Processing {len(image_files)} images...")
        
        results = []
        for i, img_file in enumerate(image_files, 1):
            print(f"Processing ({i}/{len(image_files)}): {img_file.name}")
            detection_result = self.detect_fractures(str(img_file), model_path)
            results.append(detection_result)
        
        # Summary
        total_detections = sum(len(r.get('boxes', [])) for r in results)
        print(f"✅ Batch processing completed!")
        print(f"   Processed: {len(results)} images")
        print(f"   Total fractures detected: {total_detections}")
        
        return results
    
    def visualize_results(self, detection_result: Dict) -> None:
        """Visualize detection results"""
        
        if 'output_path' in detection_result and Path(detection_result['output_path']).exists():
            img = cv2.imread(detection_result['output_path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f"Bone Fracture Detection Results\nDetected: {len(detection_result['boxes'])} fractures")
            plt.tight_layout()
            plt.show()
        else:
            print("❌ No visualization available. No fractures detected or results not saved.")
    
    def get_model_info(self) -> None:
        """Display information about available models"""
        
        print("\n📋 Model Information:")
        print("Available pretrained models:")
        models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        for model in models:
            print(f"   - {model}.pt (size: {'nano' if 'n' in model else 'small' if 's' in model else 'medium' if 'm' in model else 'large' if 'l' in model else 'extra-large'})")
        
        # Check for trained models
        trained_models = list(self.dirs['trained'].glob("*/weights/*.pt"))
        if trained_models:
            print("\nTrained models:")
            for model_path in trained_models:
                print(f"   - {model_path}")
        else:
            print("\nNo trained models found.")
    
    def create_sample_structure(self) -> None:
        """Create sample directory structure with instructions"""
        
        # Create README file
        readme_content = ""
# Bone Fracture Detection System

## Directory Structure

