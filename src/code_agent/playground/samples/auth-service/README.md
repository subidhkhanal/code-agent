# Sample service

A tiny service used as a test fixture. It issues session tokens, computes invoices,
and parses raw HTTP responses.

## Tokens

Tokens expire after `DEFAULT_TTL_SECONDS`. Expired tokens must be rejected by `verify_token`.
