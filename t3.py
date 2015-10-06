import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
#---------------------
#     parte 1
#---------------------


e=1.487
def f(y, v,eta=e):
    return v, -y-eta*(y**2-1)*v

def get_k1(y_n, v_n, h, f):
    f_eval = f(y_n, v_n)
    return h * f_eval[0], h * f_eval[1]

def get_k2(y_n, v_n, h, f):
    k1 = get_k1(y_n, v_n, h, f)
    f_eval = f(y_n + k1[0]/2, v_n + k1[1]/2)
    return h * f_eval[0], h * f_eval[1]

def get_k3(y_n, v_n, h,f):
    k1=get_k1(y_n, v_n, h, f)
    k2=get_k2(y_n, v_n, h, f)
    f_eval=f(y_n-k1[0]-2*k2[0],v_n -k1[1]-2*k2[0])
    return h*f_eval[0],h*f_eval[1]

def rk3_step(y_n, v_n, h, f):
    k1 = get_k1(y_n, v_n, h, f)
    k2 = get_k2(y_n, v_n, h, f)
    k3 = get_k1(y_n, v_n, h, f)
    y_n1 = y_n + (1/6.)*(k1[0] + k3[0] + k2[0])
    v_n1 = v_n + (1/6.)*(k1[1] + k3[1] + k2[1])
    return y_n1, v_n1


N_steps = 40000
h = 20*np.pi / N_steps
y= np.zeros(N_steps)
v = np.zeros(N_steps)


y[0] = 0.1
v[0] = 0
for i in range(1, N_steps):
    y[i], v[i] = rk3_step(y[i-1], v[i-1], h, f)



t_rk= [h * i for i in range(N_steps)]
plt.figure(1)
plt.plot(y, v, 'g')
plt.show()
plt.figure (2)
plt.plot(t_rk,y)
plt.show()

plt.figure(3)

y[0]=4
v[0]=0
for i in range(1,N_steps):
    y[i],v[i]= rk3_step(y[i-1], v[i-1], h, f)

plt.plot(y,v,'r')
plt.show()

#---------------------
#     parte 2
#---------------------

def lorenz(t,A):
    x,y,z=A
    return si*(A[1]-A[0]), A[0]*(rho-A[2])-A[1],A[0]*A[1]-beta*A[2]

si=10
rho=28
beta=8/3

t0=1e-3
ci=[1,1,1]#condiciones iniciales de x y z

r = ode(lorenz)
r.set_integrator('dopri5')
r.set_initial_value(t0,ci)

#t=1000
t = np.arange(0, 15 * np.pi, 0.01)
t_values = np.linspace(t0, 2 * np.pi, 1000)
x_values = np.zeros(len(t))
y_values = np.zeros(len(t))
z_values = np.zeros(len(t))

for i in range(len(t_values)):
    r.integrate(t_values[i])
    x_values[i], y_values[i] , z_values[i]= r.y

fig = plt.figure(4)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

x=x_values
y=y_values
z=z_values

ax.plot(x,y,z)
plt.show()
