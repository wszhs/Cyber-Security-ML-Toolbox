import autograd.numpy as np
import autograd
import os
from autograd import grad
from autograd import jacobian
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from func import f1, f2, f3, saddle,mop1
from algo import gda, eg, ogda, co,follow,sga

import argparse
parser = argparse.ArgumentParser()
# 1 2 3 saddle
parser.add_argument("--function", default=1, help="choose from three low dimensional example functions, 1-3")
opt = parser.parse_args()
function = opt.function

# Select target function
if function==1:
    target = f1              # (0,0) is local minimax and global minimax
    z_0 = np.array([5., 7.]) # Set initial point
    plot_width = 12          # Set range of the plot
elif function==2:
    target = f2         # (0,0) is not local minimax and not global minimax
    z_0 = np.array([6., 5.])
    plot_width = 12
elif function==3:
    target = f3         # (0,0) is local minimax
    z_0 = np.array([7., 5.])
    plot_width = 8
elif function=='saddle':
    target = saddle         # (0,0) is local minimax
    z_0 = np.array([5.0, 5.0])
    plot_width = 6

elif function=='mop1':
    target = mop1         # (0,0) is local minimax
    z_0 = np.array([5.0, 5.0])
    plot_width = 10


# Run all algorithms on target
zfr=follow(z_0, num_iter = 1000, alpha = 0.05, target=target)
zgda=gda(z_0, num_iter = 1000, alpha = 0.05, target=target)
zogda=ogda(z_0, num_iter = 1000, alpha = 0.05, target=target)
zeg=eg(z_0, num_iter = 1000, alpha = 0.05, target=target)
zco=co(z_0, num_iter = 1000, alpha = 0.05, gamma=0.1, target=target)
zsga=sga(z_0, num_iter = 1000, alpha = 0.01, lamb=1.0, target=target)


# Plot trajectory with contour
plt.rcParams.update({'font.size': 14})
def_colors=(plt.rcParams['axes.prop_cycle'].by_key()['color'])

#plot_width=12
plt.figure(figsize=(8,5))
axes = plt.gca()
axes.set_xlim([-plot_width,plot_width])
axes.set_ylim([-plot_width,plot_width])

x1 = np.arange(-plot_width,plot_width,0.1)
y1 = np.arange(-plot_width,plot_width,0.1)
X,Y = np.meshgrid(x1,y1)
Z = np.zeros_like(X)
for i in range(len(x1)):
    for j in range(len(y1)):
        Z[j][i] = target(np.array([x1[i] ,y1[j]]))

cset = plt.contourf(X,Y,Z,20) 

contour = plt.contour(X,Y,Z,30,colors='k')
plt.clabel(contour,fontsize=10,colors='k')
plt.colorbar(cset)
init=plt.plot(z_0[0],z_0[1],'o',zorder=20,ms=12.0,color='r')

lw = 2
hw = 0.7
line6,=plt.plot(zfr[:,0],zfr[:,1],'-',color='r',linewidth=lw,zorder=10)
line1,=plt.plot(zgda[:,0],zgda[:,1],'-',linewidth=lw,color=def_colors[9],zorder=2)
line2,=plt.plot(zogda[:,0],zogda[:,1],'-',linewidth=lw,color=def_colors[1])
line3,=plt.plot(zeg[:,0],zeg[:,1],'--',linewidth=lw,color=def_colors[2])
line4,=plt.plot(zsga[:,0],zsga[:,1],'--',color=def_colors[0],linewidth=lw)
line5,=plt.plot(zco[:,0],zco[:,1],'--',color='xkcd:violet',linewidth=lw)
plt.legend((line6,line1, line2, line3, line4, line5), ('FR','GDA', 'OGDA', 'EG', 'SGA', 'CO'), loc=4)

plt.show()