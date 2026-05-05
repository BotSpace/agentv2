from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


DEFAULT_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAl8Oq208sLKpkR1XZuGWs
QeGHzinPikp6dLG6aI/0QLRs3QxgiDVZEdaMa+FiGrJ0DYmqzNya5hoa6fCEeTr2
N2Cyf1sATi166/qZooq1sWFgaogQDhhsi7NcAboopU5sI5fxMUsKvQTuoZ0TakKx
ohCn2Iyb/p4HFnwqsl5fT0J+jq5L0/rt3wM9H9cpdrEXZeSl9jbdolvhvMW7/z4L
c/XJ6dDInRKAgmhrlOpUFeqoP1pAhOnWoq8KFXClB69wmcT/XNdE4hUmbRqTaYKD
9VsDZ3+rFOyr0KMhyHMgdH84YT/2AGS5ECp7RyOwuoA8AMji9Bhvy6DhHh74Yj58
owIDAQAB
-----END PUBLIC KEY-----"""


bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    claims: dict[str, Any]


def get_jwt_public_key() -> str:
    return os.getenv("AGENT_JWT_PUBLIC_KEY") or DEFAULT_PUBLIC_KEY


def get_jwt_algorithms() -> list[str]:
    raw = os.getenv("AGENT_JWT_ALGORITHMS", "RS256")
    return [item.strip() for item in raw.split(",") if item.strip()]


def decode_token(token: str, *, public_key: str | None = None) -> dict[str, Any]:
    return jwt.decode(
        token,
        public_key or get_jwt_public_key(),
        algorithms=get_jwt_algorithms(),
        options={"verify_aud": False},
    )


def user_from_claims(claims: dict[str, Any]) -> CurrentUser:
    user_id = claims.get("sub") or claims.get("user_id") or claims.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="JWT payload must include sub, user_id, or id")
    return CurrentUser(user_id=str(user_id), claims=claims)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        claims = decode_token(credentials.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc
    return user_from_claims(claims)
