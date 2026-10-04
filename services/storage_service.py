import logging
import os
from typing import Optional
import boto3
import botocore
from botocore.exceptions import ClientError

logger = logging.getLogger("services.storage_service")

class StorageService:
    """Service to manage uploads, downloads, deletions, and pre-signed URLs on Cloudflare R2."""

    def __init__(self):
        self.access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self.endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        self.bucket_name = os.environ.get("R2_BUCKET_NAME")

        self.is_configured = bool(self.access_key_id and self.secret_access_key and self.endpoint_url and self.bucket_name)

        # Validate configuration
        if not self.is_configured:
            missing = []
            if not self.access_key_id: missing.append("R2_ACCESS_KEY_ID")
            if not self.secret_access_key: missing.append("R2_SECRET_ACCESS_KEY")
            if not self.endpoint_url: missing.append("R2_ENDPOINT_URL")
            if not self.bucket_name: missing.append("R2_BUCKET_NAME")
            logger.warning(f"R2 Storage missing environment variables: {', '.join(missing)}. R2 storage fallback will be used.")
            self.s3 = None
            return

        # Configure boto3 client for Cloudflare R2
        self.config = botocore.config.Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"}
        )
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            region_name="auto",
            config=self.config
        )

    def upload_file(self, local_path: str, r2_key: str) -> bool:
        """Upload a local file to Cloudflare R2."""
        if not self.is_configured:
            logger.warning("R2 storage not configured. Bypassing upload.")
            return False

        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist to upload: {local_path}")
            return False

        try:
            logger.info(f"Uploading local file {local_path} to R2 key {r2_key}...")
            self.s3.upload_file(local_path, self.bucket_name, r2_key)
            logger.info(f"Successfully uploaded {local_path} to R2 key {r2_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload file to R2: {e}")
            return False

    def download_file(self, r2_key: str, local_path: str) -> bool:
        """Download an object from Cloudflare R2 to a local file path."""
        if not self.is_configured:
            logger.warning("R2 storage not configured. Bypassing download.")
            return False
        try:
            logger.info(f"Downloading R2 key {r2_key} to local path {local_path}...")
            # Ensure local folder exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket_name, r2_key, local_path)
            logger.info(f"Successfully downloaded R2 key {r2_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download R2 key {r2_key}: {e}")
            return False

    def exists(self, r2_key: str) -> bool:
        """Check if an object exists in Cloudflare R2."""
        if not self.is_configured:
            return False
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=r2_key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            logger.error(f"Error checking existence for R2 key {r2_key}: {e}")
            # Assume it doesn't exist if error occurs
            return False

    def delete_file(self, r2_key: str) -> bool:
        """Delete an object from Cloudflare R2."""
        if not self.is_configured:
            return False
        try:
            logger.info(f"Deleting R2 key {r2_key}...")
            self.s3.delete_object(Bucket=self.bucket_name, Key=r2_key)
            logger.info(f"Successfully deleted R2 key {r2_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete R2 key {r2_key}: {e}")
            return False

    def delete_folder(self, r2_prefix: str) -> bool:
        """Delete all objects matching a key prefix (logical folder) from Cloudflare R2."""
        if not self.is_configured:
            return False
        try:
            logger.info(f"Deleting R2 objects matching prefix {r2_prefix}...")
            paginator = self.s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=r2_prefix)
            
            delete_batch = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        delete_batch.append({"Key": obj["Key"]})
                        
                        # Delete in chunks of 1000 keys (S3 API limit)
                        if len(delete_batch) >= 1000:
                            self.s3.delete_objects(Bucket=self.bucket_name, Delete={"Objects": delete_batch})
                            delete_batch = []
                            
            if delete_batch:
                self.s3.delete_objects(Bucket=self.bucket_name, Delete={"Objects": delete_batch})
                
            logger.info(f"Successfully deleted objects matching prefix {r2_prefix}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete objects under prefix {r2_prefix}: {e}")
            return False

    def generate_presigned_url(self, r2_key: str, expiration_seconds: int = 3600) -> Optional[str]:
        """Generate a pre-signed URL for direct GET access to an object."""
        if not self.is_configured:
            return None
        try:
            url = self.s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket_name, "Key": r2_key},
                ExpiresIn=expiration_seconds
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate pre-signed URL for key {r2_key}: {e}")
            return None
