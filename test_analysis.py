#!/usr/bin/env python3
"""
Test script to see what analysis is actually being performed
"""

from deepface import DeepFace
import cv2
import numpy as np

def test_analysis():
    """Test what face analysis returns"""

    # Create a simple test image (black square with white dot)
    # This won't work for analysis, but let's try with a real image if available
    # For now, let's just check what the analyze function returns for a dummy image

    print("Testing DeepFace analysis...")

    # Create dummy image
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_img[100:124, 100:124] = [255, 255, 255]  # white square

    try:
        analysis = DeepFace.analyze(
            img_path=dummy_img,  # pass numpy array
            actions=['age', 'gender', 'emotion', 'race']
        )

        print("Analysis result:")
        print(f"Age: {analysis[0]['age']}")
        print(f"Gender: {analysis[0]['dominant_gender']}")
        print(f"Emotion: {analysis[0]['dominant_emotion']}")
        print(f"Race: {analysis[0]['dominant_race']}")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    test_analysis()
