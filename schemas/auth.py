"""
Pydantic Schemas for Authentication
"""
from typing import Literal
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


class AuthUser(BaseModel):
    """Authenticated user payload"""
    user_id: str = Field(..., description="User identifier")
    role: Literal["Parent", "Teacher", "Admin", "Librarian"] = Field(
        ..., description="User role"
    )
    mode: Literal["home", "institution"] = Field(
        ..., description="User mode"
    )
