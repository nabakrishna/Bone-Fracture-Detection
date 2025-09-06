
from bone_fracture_detector import BoneFractureDetectionSystem  # Note the change from bone_fracture_detection to bone_fracture_detector

# Example usage
project_root = "C:/Users/N. K. Hazarika/OneDrive/Desktop/bone-fracture-detection"
system = BoneFractureDetectionSystem(project_root=project_root)
# image_path = f"{project_root}/data/raw/images/fracture-of-the-humeral-capitellum-milch-type-1-1-1-_jpg.rf.f40a3ca2a57d511a40839bd1ca615d54.jpg"
image_path = "C:/Users/N. K. Hazarika/Downloads/fractured3.jpg"  # Example image path
system.config['confidence_threshold'] = 0.3
result = system.detect_fractures(image_path)
# result = system.detect_fractures(image_path)
if result:
    system.visualize_results(result)

def detect_fractures(self, image_path: str, model_path: str = None, save_results: bool = True) -> Dict:
    # Add this debugging code after running inference
    results = self.model(image_path, conf=self.config['confidence_threshold'])
    result = results[0]
    
    # Debug: Print all detections regardless of class
    if result.boxes is not None:
        print(f"Raw detections: {len(result.boxes)}")
        for i, cls in enumerate(result.boxes.cls):
            print(f"Class {int(cls)}: {result.names[int(cls)]} (conf: {result.boxes.conf[i]:.3f})")
    else:
        print("No boxes detected at all")


