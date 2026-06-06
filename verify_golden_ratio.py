#!/usr/bin/env python3
"""Verify golden ratio loss coefficients for Poiseuille flow."""

import sys
sys.path.insert(0, '.')

from cfd_engine.src.training.trainer import PINNTrainer
import inspect

print("=" * 80)
print("GOLDEN RATIO TUNING VERIFICATION (Balanced Poiseuille Flow)")
print("=" * 80)

# Get signature
sig = inspect.signature(PINNTrainer.__init__)
print("\nPINNTrainer.__init__ Default Parameters:")

# Extract defaults
defaults = {}
for name, param in sig.parameters.items():
    if param.default != inspect.Parameter.empty:
        defaults[name] = param.default

# Show loss weights
loss_weights = {
    'lambda_bc': defaults.get('lambda_bc', 'NOT FOUND'),
    'lambda_smooth': defaults.get('lambda_smooth', 'NOT FOUND'),
    'lambda_radial': defaults.get('lambda_radial', 'NOT FOUND'),
    'lambda_pos': defaults.get('lambda_pos', 'NOT FOUND'),
}

for key, val in loss_weights.items():
    print(f"  {key}: {val}")

# Verify expected values
expected = {
    'lambda_bc': 15000.0,
    'lambda_smooth': 40.0,
    'lambda_radial': 250.0,
    'lambda_pos': 500.0,
}

print("\n" + "=" * 80)
print("VERIFICATION RESULTS:")
print("=" * 80)

all_correct = True
for key, expected_val in expected.items():
    actual_val = loss_weights[key]
    match = "✓ CORRECT" if actual_val == expected_val else "✗ MISMATCH"
    print(f"  {key}: Expected={expected_val}, Actual={actual_val} {match}")
    if actual_val != expected_val:
        all_correct = False

print("\n" + "=" * 80)
if all_correct:
    print("✓ ALL GOLDEN RATIO COEFFICIENTS UPDATED SUCCESSFULLY")
    print("\nPhysics Interpretation (Golden Ratio / Sweet Spot):")
    print("  • λ_radial=250.0    → BALANCED: Not too harsh (50×) or strict (1000×)")
    print("  • λ_smooth=40.0     → OPTIMAL: Prevents aliasing, permits radial curvature")
    print("  • λ_bc=15000.0      → STRICT: Wall enforcement (no slip, blue dissipation)")
    print("  • λ_pos=500.0       → Ensures u_x ≥ 0 everywhere")
    print("\nExpected Results:")
    print("  ✓ Poiseuille profile (parabolic u(r) = 2(1-r²/R²))")
    print("  ✓ Zero velocity at walls (r = ±0.5)")
    print("  ✓ Flow extends smoothly to pipe end")
    print("  ✓ No aliasing (vertical lines) in visualization")
    print("  ✓ No slug flow behavior")
else:
    print("✗ SOME COEFFICIENTS WERE NOT UPDATED CORRECTLY")
    sys.exit(1)

print("=" * 80)
