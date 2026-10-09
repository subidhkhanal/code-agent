from decimal import Decimal

from cart.pricing import Cart


def test_single_item_with_tax():
    cart = Cart()
    cart.add("mug", "10.00")
    assert cart.total() == Decimal("10.80")


def test_bulk_discount_starts_at_ten_items():
    cart = Cart()
    cart.add("pen", "1.00", quantity=10)
    assert cart.total() == Decimal("9.72")  # 10.00 - 10% = 9.00, plus 8% tax


def test_coupon_is_applied_before_tax():
    cart = Cart(coupon=Decimal("5.00"))
    cart.add("book", "25.00")
    assert cart.total() == Decimal("21.60")  # (25.00 - 5.00) * 1.08


def test_coupon_never_makes_total_negative():
    cart = Cart(coupon=Decimal("50.00"))
    cart.add("sticker", "2.00")
    assert cart.total() == Decimal("0.00")
