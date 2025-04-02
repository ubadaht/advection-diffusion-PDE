from math import *
import matplotlib.pyplot as plt
import numpy as np

ftseries = open("timeseries.txt", "a")
ftseries.write("time fluxtot Tmax  \n")

timeswidtheps = 3 # limit turbulent diffusivity to 3 half widths

# started from diffusion_2D_inst.py
imax = 32
jmax = 32

Ubig = 1.

# parameters for obstacle wake
diameter = .01 # diameter of the obstacle creating the wake (SH)
VO       = 0.5 # distance of obstacle to the left of x = 0
ymp      = 0.0 # centerpoint obstacle

# free 2D jet
Kbig = diameter*Ubig**2
print("Kbig = ", Kbig)

# fill x with coordinates of points
# the x-points are at positions -0.5*dx, 0.5*dx, 1.5*dx,  ..., 1+0.5
x = np.zeros((imax,1))
y = np.zeros((jmax,1))
xplot = np.zeros((imax,jmax))
yplot = np.zeros((imax,jmax))

#Grid type 1
dx = 1./(imax-2)
ysize = 1.
dy = ysize/(jmax-2)

yshift = 0.5*ysize # shift so that jet centerline is in the middle

for j in range(0,jmax):
   for i in range(0,imax):
      x[i] = (i-0.5)*dx
      y[j] = (j-0.5)*dy - yshift
      xplot[i,j] = x[i,0]
      yplot[i,j] = y[j,0]

# initialise arrays Tn1 and To1 are filled with zero's
Cn1  = np.zeros((imax,jmax))
Co1  = np.zeros((imax,jmax))
Cn2  = np.zeros((imax,jmax))
Co2  = np.zeros((imax,jmax))
K    = np.zeros((imax,jmax))
U    = np.zeros((imax,jmax))
V    = np.zeros((imax,jmax))
S1   = np.zeros((imax,jmax))
S2   = np.zeros((imax,jmax))
Kfilt= np.zeros((imax,jmax))

for j in range(0,jmax):
   for i in range(0,imax):
      if( 0.8 <= x[i] and x[i] <= 0.9 and -0.05 <= y[j] and y[j] <= 0.05):
         S2[i,j] = 1.0

#jet
# half width fillows from tanh(eta)^2 = 0.5
for j in range(0,jmax):
   for i in range(0,imax):
      sigma    = 7.67
      eta_half = 0.88137 # value of eta = sigma*y/x for which (1-tanh(eta)**2) = 0.5
      b_half   = eta_half*(x[i,0]+VO)/sigma
      eta = sigma*(y[j,0]-ymp)/(x[i,0]+VO)
    
      U[i,j] = 0.5*sqrt(3.)*sqrt(Kbig*sigma/(x[i,0]+VO))*(1-tanh(eta)**2)
      V[i,j] = 0.25*sqrt(3.)*sqrt(Kbig/(sigma*(x[i,0]+VO)))*(2*eta*(1-tanh(eta)**2)-tanh(eta))

      if(abs(y[j]-ymp)<timeswidtheps*b_half):
         Ucl = 0.5*sqrt(3.)*sqrt(Kbig*sigma/(x[i,0]+VO))
         epsilon_tau = 0.037*b_half*Ucl
      else:
         epsilon_tau = 0.
      K[i,j] = 2.*epsilon_tau + 0.001

# smooth out K
nfilt = 3
fc  = 0.5
foc = (1.-fc)/4
for ifilt in range(0,nfilt):
   Kfilt = K.copy()
   for j in range(1,jmax-1):
      for i in range(1,imax-1):
          K[i,j] = fc*Kfilt[i,j]+foc*(Kfilt[i-1,j]+Kfilt[i+1,j]+Kfilt[i,j-1]+Kfilt[i,j+1])


#Plotting stuff (deletable later)
plt.figure(0)
ax1 = plt.axes(projection ='3d')
ax1.plot_surface(xplot, yplot, Cn1)

plt.figure(1)
ax2 = plt.axes(projection ='3d')
ax2.plot_surface(xplot, yplot, Cn2)
plt.pause(0.1)

input("Enter to start")


time     = 0.
dt = 0.04
nstep    = 250
CritMax = 0.
for istep in range(1,nstep):
   time = time + dt
   print(istep)
   Co1 = Cn1.copy()
   Co2 = Cn2.copy()

   for j in range(1,jmax-1):
    for i in range(1,imax-1):
       if(Co1[i,j] > 0 and Co2[i,j] > 0):
          Ar = 20.
       else:
          Ar = 0.
          
       a = dt*(U[i,j]+K[i,j]/dx-(K[i+1,j]-K[i-1,j])/4/dx)/dx
       b = dt*(V[i,j]+K[i,j]/dy-(K[i,j+1]-K[i,j-1])/4/dy)/dy
       c1 = 1+dt*((U[i-1,j]-2*U[i,j])/dx+(V[i,j-1]-2*V[i,j])/dy-2*K[i,j]*(1/dx**2+1/dy**2)+Ar*Co2[i,j])
       c2 = 1+dt*((U[i-1,j]-2*U[i,j])/dx+(V[i,j-1]-2*V[i,j])/dy-2*K[i,j]*(1/dx**2+1/dy**2)-Ar*Co1[i,j])
       d = dt*(K[i,j]+(K[i+1,j]-K[i-1,j])/4)/dx**2
       e = dt*(K[i,j]+(K[i,j+1]-K[i,j-1])/4)/dy**2

       Cn1[i,j] = a*Co1[i-1,j]+b*Co1[i,j-1]+c1*Co1[i,j]+d*Co1[i+1,j]+e*Co1[i,j+1]
       Cn2[i,j] = a*Co2[i-1,j]+b*Co2[i,j-1]+c2*Co2[i,j]+d*Co2[i+1,j]+e*Co2[i,j+1]+dt*S2[i,j]

       Crit1 = 0.4/(abs(U[i,j])/dx+abs(V[i,j])/dy+K[i,j]*(1/dx**2+1/dy**2)+Ar*Cn1[i,j])
       Crit2 = 0.4/(abs(U[i,j])/dx+abs(V[i,j])/dy+K[i,j]*(1/dx**2+1/dy**2)+Ar*Cn2[i,j])
       if(Crit1 > CritMax):
          CritMax = Crit1
       if(Crit2 > CritMax):
          CritMax = Crit2

   for i in range(1,imax-1):
      Cn1[i,0] = - Cn1[i,1]
      Cn2[i,0] = - Cn2[i,1]
      Cn1[i,jmax-1] = - Cn1[i,jmax-2]
      Cn2[i,jmax-1] = - Cn2[i,jmax-2]

   for j in range(1,jmax-1):
      Cn1[imax-1,j] = Cn1[imax-2,j]
      Cn2[imax-1,j] = Cn2[imax-2,j]
      
      eta = sigma*(y[j,0]-ymp)/(VO)
      Cn1[0,j] = 2*1*sqrt(1-tanh(eta)**2) - Cn1[1,j]
      Cn2[0,j] = - Cn2[1,j]
      
#for i in range(1,imax-1): #Isolation
#Tn[i,0]      = Tn[i,1]
#Tn[i,jmax-1] = Tn[i,jmax-2]
   plt.figure(0)
   plt.clf()
   ax1 = plt.axes(projection ='3d')
   ax1.plot_surface(xplot, yplot, Cn1)
   ax1.set_zlim3d(0, 1)
   
   plt.figure(1)
   plt.clf()
   ax2 = plt.axes(projection ='3d')
   ax2.plot_surface(xplot, yplot, Cn2)
   ax2.set_zlim3d(0, 0.4)
   #plt.grid()
   plt.pause(0.01)

print(CritMax)

input("Enter to close")

#plt.plot(xplot[:,10],Tn[:,10],'bo-')
#   plt.clf()
#  plt.contour(xplot,yplot,Tn1,20)
#   ax = plt.axes(projection='3d')
#   ax.plot_surface(xplot,yplot,V)
#axis sets limits z-axis
#   ax.set_zlim3d(0,0.5)
#   plt.plot_surface(xplot,yplot,Tn1)
#   surf(xplot,yplot,Tn1)
