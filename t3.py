import numpy as np
import matplotlib.pyplot as plt


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
h = 10*np.pi / N_steps
y= np.zeros(N_steps)
v = np.zeros(N_steps)


y[0] = 0.1
v[0] = 0
for i in range(1, N_steps):
    y[i], v[i] = rk3_step(y[i-1], v[i-1], h, f)



#t_rk = [h * i for i in range(N_steps)]
plt.figure(1)
plt.plot(y, v, 'g')
plt.show()

plt.figure(2)

y[0]=4
v[0]=0
for i in range(1,N_steps):
    y[i],v[i]= rk3_step(y[i-1], v[i-1], h, f)

plt.plot(y,v,'r')
plt.show()
