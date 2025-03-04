"""
Base models for the paper writing system.

This module provides base model classes with common functionality
such as internationalization support, versioning, and validation.
"""

from typing import Dict, Optional, Any, ClassVar, Type, TypeVar, Generic, List
from datetime import datetime
from pydantic import BaseModel, Field, validator

T = TypeVar('T')

class I18nString(BaseModel):
    """Internationalized string with support for multiple languages"""
    en: str
    ko: Optional[str] = None
    
    def get(self, lang: str = "en") -> str:
        """Get string in the specified language, falling back to English"""
        if lang == "ko" and self.ko:
            return self.ko
        return self.en


class VersionedModel(BaseModel):
    """Base model with versioning support"""
    model_version: ClassVar[str] = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def update(self, **kwargs) -> 'VersionedModel':
        """Update model fields and set updated_at timestamp"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.now()
        return self


class I18nModel(VersionedModel):
    """Base model with internationalization support"""
    i18n_fields: ClassVar[List[str]] = []
    
    def set_i18n(self, field: str, value: str, lang: str = "en") -> None:
        """Set internationalized value for a field"""
        if field not in self.i18n_fields:
            self.i18n_fields.append(field)
        elif lang == "en":
            self.i18n_fields[field] = value
        elif lang == "ko":
            self.i18n_fields[field] = value
    
    def get_i18n(self, field: str, lang: str = "en") -> str:
        """Get internationalized value for a field"""
        if field in self.i18n_fields:
            return self.i18n_fields[field]
        return getattr(self, field, "")


class StateTransitionModel(VersionedModel, Generic[T]):
    """Base model with state transition support"""
    status: str = "initialized"
    error: Optional[str] = None
    
    def transition(self, new_status: str, error: Optional[str] = None) -> T:
        """Transition to a new status"""
        self.status = new_status
        if error:
            self.error = error
        self.updated_at = datetime.now()
        return self
    
    def is_in_status(self, *statuses: str) -> bool:
        """Check if the model is in one of the specified statuses"""
        return self.status in statuses
    
    def validate_transition(self, current_status: str, allowed_statuses: List[str]) -> None:
        """Validate that a transition is allowed"""
        if current_status not in allowed_statuses:
            allowed = ", ".join(allowed_statuses)
            raise ValueError(f"Cannot transition from '{current_status}'. Allowed statuses: {allowed}") 