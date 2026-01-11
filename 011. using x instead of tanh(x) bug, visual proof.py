import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
tanh_x = np.tanh(x)

# Correct gradient
correct_grad = 1 - tanh_x**2

# Buggy gradient (what happens if you use x instead of tanh_x)
buggy_grad = 1 - x**2

plt.figure(figsize=(10, 6))
plt.plot(x, correct_grad, 'b-', label='Correct: 1 - tanh²(x)', linewidth=3)
plt.plot(x, buggy_grad, 'r--', label='Buggy: 1 - x²', linewidth=2)
plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
plt.ylim(-5, 1.5)
plt.xlabel('x')
plt.ylabel('Gradient')
plt.title('The Dangerous Bug: Using x instead of tanh(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("At x = 2.0:")
print(f"  tanh(2.0) = {np.tanh(2.0):.4f}")
print(f"  Correct gradient = 1 - {np.tanh(2.0):.4f}² = {1 - np.tanh(2.0)**2:.4f}")
print(f"  Buggy gradient = 1 - {2.0}² = {1 - 2.0**2:.4f}")
print(f"  Error: {abs((1-2.0**2) - (1-np.tanh(2.0)**2))/(1-np.tanh(2.0)**2)*100:.0f}%!")