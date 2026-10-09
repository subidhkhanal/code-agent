"""URL slugs for post titles."""

import re
import unicodedata


def slugify(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]", "-", ascii_title.lower())
    return slug.strip("-")
