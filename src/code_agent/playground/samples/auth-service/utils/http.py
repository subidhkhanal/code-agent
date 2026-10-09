"""Small HTTP helpers."""

import functools
import json


def retry(times: int):
    """Retry the wrapped function up to `times` attempts on ConnectionError."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except ConnectionError:
                    if attempt == times - 1:
                        raise

        return wrapper

    return decorator


def parseHttpResponse(raw: bytes) -> dict:
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode()
    return {"status": int(status_line.split()[1]), "body": json.loads(body or b"null")}
