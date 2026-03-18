"""
Pydantic Schemas for Authentication
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Request model for login"""
    email: str = Field(..., min_length=3, max_length=254, description="User email address")
    password: str = Field(..., min_length=6, max_length=128, description="User password")
    mode: Literal["home", "institution"] = Field(
        ..., description="Selected mode: home or institution"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "email": "parent@kiddoland.local",
                "password": "Parent123!",
                "mode": "home",
            }
        }


class AuthRegisterRequest(BaseModel):
    """Request model for registration"""
    email: str = Field(..., min_length=3, max_length=254, description="User email address")
    password: str = Field(..., min_length=6, max_length=128, description="User password")
    name: Optional[str] = Field(None, min_length=1, max_length=120, description="User name")
    mode: Literal["home", "institution"] = Field(
        ..., description="Selected mode: home or institution"
    )
    role: Literal["Parent", "Teacher", "Admin", "Librarian"] = Field(
        "Teacher", description="User role"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "email": "parent@kiddoland.local",
                "password": "Parent123!",
                "name": "Parent User",
                "mode": "home",
                "role": "Parent",
            }
        }


class AuthTokenResponse(BaseModel):
    """Response model for issued auth token"""
    access_token: str = Field(..., description="Access token")
    token_type: Literal["bearer"] = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Seconds until token expiration")
    role: Literal["Parent", "Teacher", "Admin", "Librarian"] = Field(
        ..., description="User role"
    )
    mode: Literal["home", "institution"] = Field(
        ..., description="User mode"
    )
    email: Optional[str] = Field(None, description="User email")
    name: Optional[str] = Field(None, description="User display name")
    username: Optional[str] = Field(None, description="User handle")
    first_name: Optional[str] = Field(None, description="User first name")
    last_name: Optional[str] = Field(None, description="User last name")
    full_name: Optional[str] = Field(None, description="User full name")


class AuthUser(BaseModel):
    """Authenticated user payload"""
    user_id: str = Field(..., description="User identifier")
    email: Optional[str] = Field(None, description="User email")
    name: Optional[str] = Field(None, description="User display name")
    username: Optional[str] = Field(None, description="User handle")
    first_name: Optional[str] = Field(None, description="User first name")
    last_name: Optional[str] = Field(None, description="User last name")
    full_name: Optional[str] = Field(None, description="User full name")
    role: Literal["Parent", "Teacher", "Admin", "Librarian"] = Field(
        ..., description="User role"
    )
    mode: Literal["home", "institution"] = Field(
        ..., description="User mode"
    )
