"""
Authentication Router
Handles login and token validation endpoints.
"""
from fastapi import APIRouter, Depends

from schemas.auth import AuthLoginRequest, AuthRegisterRequest, AuthTokenResponse, AuthUser
from utils.auth_service import authenticate_user, create_access_token, get_current_user, register_user

router = APIRouter()


@router.post("/login", response_model=AuthTokenResponse)
def login(request: AuthLoginRequest) -> AuthTokenResponse:
    user = authenticate_user(request.email, request.password, request.mode)
    token_data = create_access_token(user, request.mode)

    return AuthTokenResponse(
        access_token=token_data["token"],
        expires_in=token_data["expires_in"],
        role=user["role"],
        mode=request.mode,
    )


@router.post("/register", response_model=AuthTokenResponse)
def register(request: AuthRegisterRequest) -> AuthTokenResponse:
    user = register_user(request.email, request.password, request.mode, request.role)
    token_data = create_access_token(user, request.mode)

    return AuthTokenResponse(
        access_token=token_data["token"],
        expires_in=token_data["expires_in"],
        role=user["role"],
        mode=request.mode,
    )


@router.get("/validate", response_model=AuthUser)
def validate_token(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
    return current_user
