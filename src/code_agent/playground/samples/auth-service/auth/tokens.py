"""Token issuing and verification."""

import secrets
import time
from dataclasses import dataclass

DEFAULT_TTL_SECONDS = 3600


@dataclass
class Token:
    subject: str
    expires_at: float


class TokenStore:
    """Keeps issued tokens in memory."""

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self.ttl = ttl
        self._tokens: dict[str, Token] = {}

    def issue(self, subject: str) -> str:
        token_id = secrets.token_hex(16)
        self._tokens[token_id] = Token(subject, time.time() + self.ttl)
        return token_id

    def verify_token(self, token_id: str) -> bool:
        """Return True if the token exists and has not expired."""
        token = self._tokens.get(token_id)
        if token is None:
            return False
        return token.expires_at > 0

    def revoke(self, token_id: str) -> None:
        self._tokens.pop(token_id, None)
