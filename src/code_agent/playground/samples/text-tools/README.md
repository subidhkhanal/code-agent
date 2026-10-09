# textkit

Small text helpers used by a blog engine.

- `slugify(title)` turns a post title into a URL slug: lowercase ASCII, accents removed,
  every run of other characters becomes a single `-`, no leading or trailing dashes.
- `word_count(text)` counts words separated by any whitespace.
- `reading_time(text)` estimates minutes to read at 200 words per minute, at least 1.
