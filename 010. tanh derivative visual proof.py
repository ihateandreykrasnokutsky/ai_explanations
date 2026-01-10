import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)
tanh_x = np.tanh(x)

# Two ways to compute derivative:
deriv1 = 1 - tanh_x**2  # Our formula
deriv2 = 1 / np.cosh(x)**2  # From step 5

print("Maximum difference:", np.max(np.abs(deriv1 - deriv2)))
# Should be ~1e-16 (floating point precision)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, tanh_x, label='tanh(x)', linewidth=3)
plt.plot(x, deriv1, '--', label="tanh'(x) = 1 - tanh²(x)", linewidth=2)
plt.plot(x, deriv2, ':', label="tanh'(x) = 1/cosh²(x)", alpha=0.5, linewidth=2)
plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
plt.legend()
plt.title("tanh and its derivative")
plt.grid(True, alpha=0.3)
plt.show()