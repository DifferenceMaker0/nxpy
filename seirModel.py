import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.5, 0.1
t = np.linspace(0, 100, 1000)

solution = odeint(sir_model, [S0, I0, R0], t, args=(beta, gamma, N))
S, I, R = solution.T

plt.plot(t, S/N, label='Susceptible')
plt.plot(t, I/N, label='Infectious')
plt.plot(t, R/N, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Proportion')
plt.title('SIR Model Simulation (R0=5)')
plt.legend()
plt.show()