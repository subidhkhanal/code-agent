"""Cart pricing: line totals, bulk discounts, coupons and tax."""

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

TAX_RATE = Decimal("0.08")
BULK_THRESHOLD = 10
BULK_DISCOUNT = Decimal("0.10")
CENT = Decimal("0.01")


@dataclass
class LineItem:
    sku: str
    unit_price: Decimal
    quantity: int

    def total(self) -> Decimal:
        gross = self.unit_price * self.quantity
        if self.quantity > BULK_THRESHOLD:
            gross *= 1 - BULK_DISCOUNT
        return gross


@dataclass
class Cart:
    items: list[LineItem] = field(default_factory=list)
    coupon: Decimal = Decimal(0)

    def add(self, sku: str, unit_price: str, quantity: int = 1) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.items.append(LineItem(sku, Decimal(unit_price), quantity))

    def subtotal(self) -> Decimal:
        return sum((item.total() for item in self.items), Decimal(0))

    def total(self) -> Decimal:
        taxed = self.subtotal() * (1 + TAX_RATE)
        return max(taxed - self.coupon, Decimal(0)).quantize(CENT, rounding=ROUND_HALF_UP)
