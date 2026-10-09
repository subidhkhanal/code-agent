# Shopping cart

Prices a cart for checkout.

## Rules

- **Bulk discount:** buying 10 or more of the same item takes 10% off that line.
- **Coupons** are a fixed amount off, subtracted from the subtotal **before** tax.
  A coupon can never make the total negative.
- **Tax** is 8%, applied after the coupon.
- Totals are rounded to the cent, halves rounding up.
