"""
Authentication Router
Handles login and token validation endpoints.
"""
from fastapi import APIRouter, Depends

from schemas.auth import AuthLoginRequest, AuthRegisterRequest, AuthTokenResponse, AuthUser
from utils.auth_service import (
    authenticate_user,
    create_access_token,
    extract_user_profile_fields,
    get_current_user,
    get_user_by_id,
    register_user,
)

router = APIRouter()


@router.post("/login", response_model=AuthTokenResponse)
def login(request: AuthLoginRequest) -> AuthTokenResponse:
    user = authenticate_user(request.email, request.password, request.mode)
    token_data = create_access_token(user, request.mode)
    profile_fields = extract_user_profile_fields(user)

    return AuthTokenResponse(
        access_token=token_data["token"],
        expires_in=token_data["expires_in"],
        role=user["role"],
        mode=request.mode,
        email=profile_fields["email"],
        name=profile_fields["name"],
        username=profile_fields["username"],
        first_name=profile_fields["first_name"],
        last_name=profile_fields["last_name"],
        full_name=profile_fields["full_name"],
    )


@router.post("/register", response_model=AuthTokenResponse)
def register(request: AuthRegisterRequest) -> AuthTokenResponse:
    user = register_user(
        request.email,
        request.password,
        request.mode,
        request.role,
        request.name,
    )
    token_data = create_access_token(user, request.mode)
    profile_fields = extract_user_profile_fields(user)

    return AuthTokenResponse(
        access_token=token_data["token"],
        expires_in=token_data["expires_in"],
        role=user["role"],
        mode=request.mode,
        email=profile_fields["email"],
        name=profile_fields["name"],
        username=profile_fields["username"],
        first_name=profile_fields["first_name"],
        last_name=profile_fields["last_name"],
        full_name=profile_fields["full_name"],
    )


@router.get("/validate", response_model=AuthUser)
def validate_token(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
    user = get_user_by_id(current_user.user_id)
    if not user:
        return current_user

    profile_fields = extract_user_profile_fields(user)
    full_name = profile_fields["full_name"] or profile_fields["name"]

    return AuthUser(
        user_id=current_user.user_id,
        role=current_user.role,
        mode=current_user.mode,
        email=profile_fields["email"],
        name=profile_fields["name"],
        username=profile_fields["username"],
        first_name=profile_fields["first_name"],
        last_name=profile_fields["last_name"],
        full_name=full_name,
    )


@router.post("/refresh", response_model=AuthTokenResponse)
def refresh_token(current_user: AuthUser = Depends(get_current_user)) -> AuthTokenResponse:
    user = get_user_by_id(current_user.user_id)
    token_user = user or {"id": current_user.user_id, "role": current_user.role}
    token_data = create_access_token(token_user, current_user.mode)

    profile_fields = extract_user_profile_fields(user) if user else {
        "email": current_user.email,
        "name": current_user.name,
        "username": current_user.username,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "full_name": current_user.full_name,
    }
    full_name = profile_fields.get("full_name") or profile_fields.get("name")

    return AuthTokenResponse(
        access_token=token_data["token"],
        expires_in=token_data["expires_in"],
        role=current_user.role,
        mode=current_user.mode,
        email=profile_fields.get("email"),
        name=profile_fields.get("name"),
        username=profile_fields.get("username"),
        first_name=profile_fields.get("first_name"),
        last_name=profile_fields.get("last_name"),
        full_name=full_name,
    )
