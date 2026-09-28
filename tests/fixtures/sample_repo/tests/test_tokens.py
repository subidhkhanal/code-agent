from auth.tokens import TokenStore


def test_issue_then_verify():
    store = TokenStore()
    token_id = store.issue("alice")
    assert store.verify_token(token_id)


def test_expired_token_rejected():
    store = TokenStore(ttl=-1)
    token_id = store.issue("alice")
    assert not store.verify_token(token_id)
