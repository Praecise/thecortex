import base64
import os
from typing import Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class EncryptionManager:
    """Handles encryption/decryption of model weights and sensitive data"""
    
    def __init__(self, encryption_key: str = None):
        """Initialize with encryption key or generate new one"""
        if encryption_key:
            self.key = base64.urlsafe_b64decode(encryption_key)
        else:
            self.key = self._generate_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.key))
    
    @staticmethod
    def _generate_key() -> bytes:
        """Generate a new encryption key"""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(os.urandom(32))
        return base64.urlsafe_b64decode(key)
    
    def encrypt_weights(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt model weights"""
        encrypted_weights = {}
        for key, value in weights.items():
            if isinstance(value, bytes):
                encrypted_value = self.fernet.encrypt(value)
            else:
                serialized = str(value).encode()
                encrypted_value = self.fernet.encrypt(serialized)
            encrypted_weights[key] = encrypted_value
        return encrypted_weights
    
    def decrypt_weights(self, encrypted_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt model weights"""
        decrypted_weights = {}
        for key, encrypted_value in encrypted_weights.items():
            try:
                decrypted_value = self.fernet.decrypt(encrypted_value)
                decrypted_weights[key] = decrypted_value
            except Exception as e:
                raise DecryptionError(f"Failed to decrypt weight {key}: {str(e)}")
        return decrypted_weights
    
    def get_encryption_key(self) -> str:
        """Get base64 encoded encryption key"""
        return base64.urlsafe_b64encode(self.key).decode()

class DecryptionError(Exception):
    """Raised when decryption fails"""
    pass