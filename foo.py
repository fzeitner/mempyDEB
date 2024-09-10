import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE system
def odes(t, y, params_y1, params_y2):
    # Extract parameters for y1 and y2 from the dictionaries
    a = params_y1['a']
    b = params_y2['b']
    c = params_y2['c']
    
    dy1_dt = -a * y[0]
    dy2_dt = b * y[0] - c * y[1]
    
    return [dy1_dt, dy2_dt]

# Define two parameter dictionaries
params_y1 = {
    'a': 0.5,  # parameter for y1
}

params_y2 = {
    'b': 1.0,  # parameter for y2
    'c': 0.3   # parameter for y2
}

# Initial conditions
y0 = [1, 0]  # y1(0) = 1, y2(0) = 0

# Time span for the integration
t_span = [0, 10]  # from t=0 to t=10

# Time points at which to store the solution
t_eval = np.linspace(0, 10, 100)

# Solve the ODE system
solution = solve_ivp(odes, t_span, y0, args=(params_y1, params_y2), t_eval=t_eval)

# Plot the solution
plt.plot(solution.t, solution.y[0], label='y1(t)')
plt.plot(solution.t, solution.y[1], label='y2(t)')
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.legend()
plt.show()
