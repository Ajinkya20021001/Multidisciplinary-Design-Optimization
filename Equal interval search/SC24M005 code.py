import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f(x, y):
    return 2 * x ** 2 + 3 * y ** 2 + 4 * x * y + 10

# Input parameters
x_k = np.array([1, 1])                               # Initial point
d_k = np.array([-8, -10]) / np.sqrt(164)             # Descent Direction
h = 1e-5                                             # Step size for numerical gradient
tol = 1e-15                                          # Tolerance for floating point comparisons
q_upper = 5                                          # Upper bound for step size
q_lower = 0                                          # Lower bound for step size
n = 10                                               # Number of intervals

# Generate contour plot
xi = np.linspace(-2, 2, 100)
yi = np.linspace(-2, 2, 100)
a, b = np.meshgrid(xi, yi)



def f_alpha(q):
    return f(x_k[0] + q * d_k[0], x_k[1] + q * d_k[1])


# Numerical gradient at x_k
df_dx = (f(x_k[0] + h, x_k[1]) - f(x_k[0] - h, x_k[1])) / (2 * h)
df_dy = (f(x_k[0], x_k[1] + h) - f(x_k[0], x_k[1] - h)) / (2 * h)
gradf_at_x_k = np.array([df_dx, df_dy])

gradf_dot_dk = np.dot(gradf_at_x_k, d_k)

if gradf_dot_dk < 0:
    print("The direction selected is a descent direction.")
elif gradf_dot_dk > 0:
    raise ValueError("The direction selected is an ascent direction.")
else:
    raise ValueError("The point is the optimum (minima) point.")

# # Error check for decreasing function
f_lower = f_alpha(q_lower)
f_upper = f_alpha(q_upper)

if f_lower > f_upper:
    raise ValueError("Change the limits of step size. No local minima in the interval.")

# Using scipy.optimize's minimize_scalar to find optimal step size
res = minimize_scalar(f_alpha, bounds=(q_lower, q_upper), method='bounded')
opt_q = res.x
min_f = res.fun

print(f'stepsize: {opt_q:.15f}')
print(f'Minimum function value: {min_f:.15f}')
print(f'Minimum function occurs at: ({x_k[0] + opt_q * d_k[0]:.6f}, {x_k[1] + opt_q * d_k[1]:.6f})')

min_point = np.array([x_k[0] + opt_q * d_k[0], x_k[1] + opt_q * d_k[1]])
#Test case 1
# Calculate the slope between x_k and the minimum point
slope_min_point = (min_point[1] - x_k[1]) / (min_point[0] - x_k[0])

# Calculate the slope of the point x_k in direction d_k
slope_dk = d_k[1] / d_k[0]

# Check whether the slopes are equal or not
if np.isclose(slope_min_point, slope_dk, atol=1e-12):
    print("Test passed by slopes check.")
else:
    print("Test failed by slopes check.")
    
#Test case 2
# Check if the minimum point lies on the function
f_min_point = f(min_point[0], min_point[1])
if np.isclose(f_min_point, min_f, atol=1e-12):
    print("Test passed: The minimum point lies on the function.")
else:
    print("Test failed: The minimum point does not lie on the function.")

plt.figure()
plt.contour(a, b, f(a, b), levels=200)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('Contour Plot of f(x, y)')
plt.colorbar()
plt.plot(x_k[0], x_k[1], 'ro', label='Initial Point x_k')
min_point = np.array([x_k[0] + opt_q * d_k[0], x_k[1] + opt_q * d_k[1]])
plt.plot(min_point[0], min_point[1], 'go', label='Minimum Point')
plt.arrow(x_k[0], x_k[1], (min_point[0] - x_k[0]) * 0.9, (min_point[1] - x_k[1]) * 0.9, 
          head_width=0.1, head_length=0.14, fc='blue', ec='blue', label='alpha_k')
plt.legend()
plt.show()