#!/usr/bin/env python3
"""Verify loss coefficient updates for Poiseuille flow."""

import sys
sys.path.insert(0, '.')

from cfd_engine.src.training.trainer import PINNTrainer
import inspect

print("=" * 70)
print("LOSS COEFFICIENT VERIFICATION (Tuned for Poiseuille Flow)")
print("=" * 70)

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
    'lambda_smooth': 10.0,
    'lambda_radial': 50.0,
    'lambda_pos': 500.0,
}

print("\n" + "=" * 70)
print("VERIFICATION RESULTS:")
print("=" * 70)

all_correct = True
for key, expected_val in expected.items():
    actual_val = loss_weights[key]
    match = "✓ CORRECT" if actual_val == expected_val else "✗ MISMATCH"
    print(f"  {key}: Expected={expected_val}, Actual={actual_val} {match}")
    if actual_val != expected_val:
        all_correct = False

print("\n" + "=" * 70)
if all_correct:
    print("✓ ALL COEFFICIENTS UPDATED SUCCESSFULLY")
    print("\nPhysics Interpretation:")
    print("  • λ_bc=15000.0     → Strict wall BC (u=0 at r=R, NO SLIP)")
    print("  • λ_radial=50.0    → Relaxed guide (allows natural parabolic profile)")
    print("  • λ_smooth=10.0    → Light filtering (permits radial curvature)")
    print("  • λ_pos=500.0      → Ensures u_x ≥ 0 everywhere")
    print("\n  Result: Poiseuille (parabolic) flow with no wall slip")
else:
    print("✗ SOME COEFFICIENTS WERE NOT UPDATED CORRECTLY")
    sys.exit(1)

print("=" * 70)
