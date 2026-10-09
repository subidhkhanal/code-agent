"""Invoice arithmetic."""

from decimal import Decimal

TAX_RATE = Decimal("0.2")


class InvoiceCalculator:
    def __init__(self, lines: list[tuple[str, Decimal]]) -> None:
        self.lines = lines

    def calculateTotal(self) -> Decimal:
        subtotal = sum((amount for _, amount in self.lines), Decimal(0))
        return subtotal * (1 + TAX_RATE)


def apply_discount(total: Decimal, percent: int) -> Decimal:
    """Reduce a total by a whole-number percentage."""
    return total * (Decimal(100 - percent) / 100)
