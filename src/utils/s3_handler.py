"""
S3 Handler for Model Storage and Retrieval
===========================================
Handles uploading, downloading, and managing ML models in AWS S3.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

import boto3
import joblib
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3ModelHandler:
    """
    Manages ML model storage and retrieval from AWS S3.
    
    Environment Variables:
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: AWS secret key
        AWS_REGION: AWS region (default: us-east-1)
        S3_BUCKET_NAME: S3 bucket name for model storage
        S3_MODEL_PREFIX: Prefix/folder path in S3 (default: models/)
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        model_prefix: str = "models/",
    ):
        """
        Initialize S3 handler.

        Args:
            bucket_name: S3 bucket name (reads from env if not provided)
            region: AWS region (reads from env if not provided)
            model_prefix: Prefix for model objects in S3
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.model_prefix = model_prefix.rstrip("/") + "/"

        if not self.bucket_name:
            raise ValueError(
                "S3 bucket name must be provided or set via S3_BUCKET_NAME env variable"
            )

        # Initialize S3 client
        try:
            self.s3_client = boto3.client("s3", region_name=self.region)
            logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.error(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            )
            raise

    def upload_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload a trained model to S3.

        Args:
            model: The trained model object
            model_name: Name for the model file (e.g., 'churn_model_v1.pkl')
            metadata: Optional metadata to store with the model

        Returns:
            S3 URI of the uploaded model
        """
        try:
            # Ensure model name ends with .pkl
            if not model_name.endswith(".pkl"):
                model_name = f"{model_name}.pkl"

            s3_key = f"{self.model_prefix}{model_name}"

            # Serialize model to bytes
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)

            # Prepare metadata
            extra_args = {}
            if metadata:
                # Convert metadata to string format for S3
                extra_args["Metadata"] = {
                    str(k): str(v) for k, v in metadata.items()
                }

            # Upload to S3
            logger.info(f"Uploading model to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_fileobj(buffer, self.bucket_name, s3_key, extra_args)

            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Model successfully uploaded to {s3_uri}")

            return s3_uri

        except ClientError as e:
            logger.error(f"Failed to upload model to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model upload: {e}")
            raise

    def upload_model_from_file(
        self,
        file_path: str,
        model_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload a model file to S3.

        Args:
            file_path: Path to the model file
            model_name: Name for the model in S3 (uses filename if not provided)
            metadata: Optional metadata

        Returns:
            S3 URI of the uploaded model
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model_name = model_name or file_path.name
        
        if not model_name.endswith(".pkl"):
            model_name = f"{model_name}.pkl"

        s3_key = f"{self.model_prefix}{model_name}"

        try:
            # Prepare metadata
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {
                    str(k): str(v) for k, v in metadata.items()
                }

            logger.info(f"Uploading model from {file_path} to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(
                str(file_path), self.bucket_name, s3_key, ExtraArgs=extra_args
            )

            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Model successfully uploaded to {s3_uri}")

            return s3_uri

        except ClientError as e:
            logger.error(f"Failed to upload model file to S3: {e}")
            raise

    def download_model(self, model_name: str) -> Tuple[Any, dict]:
        """
        Download and load a model from S3.

        Args:
            model_name: Name of the model file in S3

        Returns:
            Tuple of (model object, metadata dict)
        """
        try:
            if not model_name.endswith(".pkl"):
                model_name = f"{model_name}.pkl"

            s3_key = f"{self.model_prefix}{model_name}"

            logger.info(f"Downloading model from s3://{self.bucket_name}/{s3_key}")

            # Download model to memory
            buffer = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, s3_key, buffer)
            buffer.seek(0)

            # Load model
            model = joblib.load(buffer)

            # Get metadata
            try:
                response = self.s3_client.head_object(
                    Bucket=self.bucket_name, Key=s3_key
                )
                metadata = response.get("Metadata", {})
            except Exception as e:
                logger.warning(f"Could not retrieve metadata: {e}")
                metadata = {}

            logger.info(f"Model successfully downloaded from S3")

            return model, metadata

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(
                    f"Model not found in S3: s3://{self.bucket_name}/{s3_key}"
                )
            logger.error(f"Failed to download model from S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model download: {e}")
            raise

    def download_model_to_file(self, model_name: str, local_path: str) -> str:
        """
        Download a model from S3 to a local file.

        Args:
            model_name: Name of the model in S3
            local_path: Local path to save the model

        Returns:
            Path to the downloaded file
        """
        try:
            if not model_name.endswith(".pkl"):
                model_name = f"{model_name}.pkl"

            s3_key = f"{self.model_prefix}{model_name}"

            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Downloading model from s3://{self.bucket_name}/{s3_key} to {local_path}"
            )
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)

            logger.info(f"Model downloaded to {local_path}")
            return local_path

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(
                    f"Model not found in S3: s3://{self.bucket_name}/{s3_key}"
                )
            logger.error(f"Failed to download model to file: {e}")
            raise

    def list_models(self, prefix: Optional[str] = None) -> list:
        """
        List all models in S3 bucket.

        Args:
            prefix: Optional prefix to filter models

        Returns:
            List of model names
        """
        try:
            search_prefix = self.model_prefix
            if prefix:
                search_prefix = f"{self.model_prefix}{prefix}"

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=search_prefix
            )

            if "Contents" not in response:
                return []

            models = []
            for obj in response["Contents"]:
                # Extract model name (remove prefix)
                model_name = obj["Key"].replace(self.model_prefix, "")
                if model_name:  # Skip if empty (prefix itself)
                    models.append(
                        {
                            "name": model_name,
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                        }
                    )

            logger.info(f"Found {len(models)} models in S3")
            return models

        except ClientError as e:
            logger.error(f"Failed to list models: {e}")
            raise

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from S3.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if successful
        """
        try:
            if not model_name.endswith(".pkl"):
                model_name = f"{model_name}.pkl"

            s3_key = f"{self.model_prefix}{model_name}"

            logger.info(f"Deleting model from s3://{self.bucket_name}/{s3_key}")
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)

            logger.info(f"Model deleted successfully")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete model: {e}")
            raise

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in S3.

        Args:
            model_name: Name of the model

        Returns:
            True if exists, False otherwise
        """
        try:
            if not model_name.endswith(".pkl"):
                model_name = f"{model_name}.pkl"

            s3_key = f"{self.model_prefix}{model_name}"

            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_model_info(self, model_name: str) -> dict:
        """
        Get metadata and info about a model in S3.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model info
        """
        try:
            if not model_name.endswith(".pkl"):
                model_name = f"{model_name}.pkl"

            s3_key = f"{self.model_prefix}{model_name}"

            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)

            return {
                "name": model_name,
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "metadata": response.get("Metadata", {}),
                "s3_uri": f"s3://{self.bucket_name}/{s3_key}",
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(f"Model not found: {model_name}")
            logger.error(f"Failed to get model info: {e}")
            raise


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_s3_handler(
    bucket_name: Optional[str] = None,
    region: Optional[str] = None,
    model_prefix: str = "models/",
) -> S3ModelHandler:
    """
    Create an S3 handler instance.

    Args:
        bucket_name: S3 bucket name
        region: AWS region
        model_prefix: Prefix for models in S3

    Returns:
        Initialized S3ModelHandler
    """
    return S3ModelHandler(
        bucket_name=bucket_name, region=region, model_prefix=model_prefix
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example: Initialize handler
    handler = create_s3_handler(bucket_name="my-ml-models-bucket")

    # List models
    models = handler.list_models()
    print(f"\nAvailable models: {len(models)}")
    for model in models:
        print(f"  - {model['name']} ({model['size']} bytes)")
