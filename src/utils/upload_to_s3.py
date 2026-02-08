"""
Simple Model Upload to S3
Upload your model using credentials from .env file
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, "src")

from utils.s3_handler import S3ModelHandler

# Configuration from .env
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MODEL_NAME = os.getenv("S3_MODEL_NAME")
MODEL_FILE = "models/model_threshold_0.35_recall_0.9723.pkl"

def main():
    print("=" * 60)
    print("Uploading Model to S3")
    print("=" * 60)
    
    print(f"\nBucket: {BUCKET_NAME}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Local File: {MODEL_FILE}")
    
    # Check if model exists
    if not Path(MODEL_FILE).exists():
        print(f"\nERROR: Model file not found: {MODEL_FILE}")
        return
    
    # Upload to S3
    try:
        handler = S3ModelHandler(bucket_name=BUCKET_NAME)
        
        s3_uri = handler.upload_model_from_file(
            file_path=MODEL_FILE,
            model_name=MODEL_NAME,
            metadata={
                'threshold': '0.35',
                'recall': '0.9723'
            }
        )
        
        print(f"\n✓ SUCCESS! Model uploaded to: {s3_uri}")
        print("\nYou can now run: python main.py")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



