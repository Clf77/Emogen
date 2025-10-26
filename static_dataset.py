import pandas as pd

from pathlib import Path
from deepface import DeepFace

def get_static_images(folder_path):
    """Get the folder containing the static images that are desired to be tested"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in: {folder_path}")
        return []

    print(f"Found {len(image_files)} images to process")

    return image_files

class StaticEvaluation():
    def __init__(self, data, output_csv:str=None, models:list=["ssd"]):
        self.data = data
        self.results = []
        self.valid_paths = []
        self.emotions = []
        self.emotion_criteria = ["Model","Image","Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.models = models
        self.output_csv = output_csv


    def record_result(self, analysis:list, image_path:list, model:str):
        """Records the desired, relevant data from the analysis for DeepFace"""
        self.valid_paths.append(image_path)
        values = [model, image_path]
        for val in list(analysis['emotion'].values()):
            values.append(val)
        # values.append(list(analysis['emotion'].values()))
        self.emotions.append(values)

    def assess_models(self):
        for model in self.models:
            self.analyze_images(model=model)

    def analyze_images(self,model):
        """Uses DeepFace Analysis to record the emotions"""
        for idx, img_path in enumerate(self.data, 1):
                print(f"\nProcessing {idx}/{len(self.data)}: {img_path.name}")
                
                try:
                    # Analyze emotion using DeepFace
                    analysis = DeepFace.analyze(
                        img_path=str(img_path),
                        actions=['emotion'],
                        enforce_detection=False,  # Continue even if no face detected
                        detector_backend=model
                    )
                    
                    # Handle both single face and multiple faces
                    if isinstance(analysis, list):
                        self.record_result(analysis[0], img_path, model)
                        
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {str(e)}")
            
            # Save to CSV if requested
        if self.output_csv:
            df = pd.DataFrame(self.emotions, columns=self.emotion_criteria)
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")
        


if __name__ == "__main__":
    # Example usage
    folder_path = "static_images"  # Change this to your folder
    output_csv = "emotion_results.csv"

    models_to_test = ["opencv","ssd","mtcnn","retinaface","mediapipe"]     # emotion backends for DeepFace include opencv, ssd, dlib, mtcnn, retinaface, mediapipe
    
    data = get_static_images(folder_path)

    # Detect emotions
    static_eval = StaticEvaluation(data, output_csv, models=models_to_test)
    static_eval.assess_models()
        