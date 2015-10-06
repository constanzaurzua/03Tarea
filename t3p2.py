import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D


def lorenz(t,A):
    return [si*(A[1]-A[0]), A[0]*(rho-A[2])-A[1],A[0]*A[1]-beta*A[2]]

si=10
rho=28
beta=8/3

t0=1e-3
A0=[1,1,1]#condiciones iniciales de x y z

r = ode(lorenz)
r.set_integrator('dopri5')
r.set_initial_value(A0)

#t=1000
t = 10000
t_values = np.linspace(t0, 10* np.pi,t)
x = np.zeros(t)
y = np.zeros(t)
z = np.zeros(t)

for i in range(len(t_values)):
    r.integrate(t_values[i])
    x[i], y[i] , z[i]= r.y

fig = plt.figure(4)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')


ax.plot(x,y,z)
plt.show()
