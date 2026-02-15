import numpy as np

pi = np.pi
xi_range_20 = np.linspace(-pi/18, pi/18, 20)
xi_range_5 = np.linspace(-pi/18, pi/18, 5)

print("Xi Range 5 (small):")
print(xi_range_5)
print("\nXi Range 20 (large):")
print(xi_range_20)

# Check for near-zero values
print("\nMin absolute value in Small (5):", np.min(np.abs(xi_range_5)))
print("Min absolute value in Large (20):", np.min(np.abs(xi_range_20)))

# Check intersection
intersection = np.intersect1d(xi_range_5, xi_range_20)
print("\nExact intersection values:", intersection)
