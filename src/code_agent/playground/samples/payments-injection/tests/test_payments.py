import pytest

from app.payments import capture, refund


def test_capture_returns_amount():
    assert capture(500) == 500


def test_capture_rejects_non_positive():
    with pytest.raises(ValueError):
        capture(0)


def test_refund_is_negative():
    assert refund(500) == -500
