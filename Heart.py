#PROGRAM    that makes a 3d heart using matplot, animation and numpy libraries
#for abby but 3d
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.set_facecolor("black")
ax.set_facecolor("black")
ax.set_axis_off()

#function that calls heard_3d to return the solution of the arithmetic equation
def heart_3d(x, y, z):
    a = (x**2 + (9/4)*(y**2) + z**2 - 1)**3
    b = (x**2) * (z**3)
    c = (9/80) * (y**2) * (z**3)
    return a - b - c

bbox = (-2, 2)
xmin, xmax, ymin, ymax, zmin, zmax = bbox * 3

A = np.linspace(xmin, xmax, 60)
B = np.linspace(xmin, xmax, 60)
A1, A2 = np.meshgrid(A, A)

#for loops for each axis 
for z in B:
    X, Y = A1, A2
    Z = heart_3d(X, Y, z)
    cset = ax.contour(X, Y, Z + z, [z], zdir="z", colors="#FF0000")

for y in B:
    X, Z = A1, A2
    Y = heart_3d(X, y, Z)
    cset = ax.contour(X, Y + y, Z, [y], zdir="y", colors="#FF0000")

for x in B:
    Y, Z = A1, A2
    X = heart_3d(x, Y, Z)
    cset = ax.contour(X + x, Y, Z, [x], zdir="x", colors="#FF0000")

ax.set_zlim3d(zmin, zmax)
ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)

#function that animates the heard
def animate(i):
    ax.view_init(elev=10, azim=i)

amin = animation.FuncAnimation(fig, animate, frames=360, interval=2)

plt.show()
