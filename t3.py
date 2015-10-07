import numpy as np
import matplotlib.pyplot as plt

#---------------------
#     parte 1
#---------------------


e=1.487


'''Lo primero es definirse a traves de funciones la obtencion
de k1 k2 y k3 para luego obtener yn+1'''
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


'''Nos definimos el largo de pasos para asi obtener el
tamano de y dy/ds = v'''

N_steps = 40000
h = 20*np.pi / N_steps
y= np.zeros(N_steps)
v = np.zeros(N_steps)
y2= np.zeros(N_steps)
v2= np.zeros(N_steps)

y[0] = 0.1
v[0] = 0
for i in range(1, N_steps):
    y[i], v[i] = rk3_step(y[i-1], v[i-1], h, f)

'''Ploteamos'''

t_rk= [h * i for i in range(N_steps)]
plt.figure(1)
plt.figure(1).clf()
plt.plot(y, v, 'g', label = "$\ Condiciones \ iniciales \ y(s)=0.1 \; \ dy/ds=0$")
plt.legend()
plt.title("$ \ y(s) \ vs \ dy/ds$")
plt.xlabel('$y(s)$')
plt.ylabel('$dy/ds$')
plt.savefig("fig1.png")
plt.show()

plt.figure(2)
plt.figure(2).clf()

y2[0]=4
v2[0]=0
for i in range(1,N_steps):
    y2[i],v2[i]= rk3_step(y2[i-1], v2[i-1], h, f)

plt.plot(y2,v2,'r', label = "$\ Condiciones \ iniciales \ y(s)=4 \ ; \ dy/ds=0$")
plt.legend()
plt.title("$ \ y(s) \ vs \ dy/ds$", fontsize=20)
plt.xlabel('$y(s)$',fontsize=20)
plt.ylabel('$dy/ds$',fontsize=20)
plt.savefig("fig2.png")
plt.show()

plt.figure(3)
plt.figure(3).clf()

plt.plot(t_rk,y2,'r',label = " $\ Condiciones \ iniciales \ y(s)=4 \ dy/ds= 0$")

plt.legend()
plt.plot(t_rk,y,'b', label =" $\ Condiciones \ iniciales \ y(s)=0.1 \ dy/ds= 0$")
plt.legend()
#plt.xlabel('$y(s)$')
plt.title(" $ \ y \ vs \ s$",fontsize=20)
plt.ylabel('$y(s)$',fontsize=20)
plt.xlabel("$s$",fontsize=20)
plt.savefig("fig3.png")

plt.show()
