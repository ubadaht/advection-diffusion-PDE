from math import *
import matplotlib.pyplot as plt
import numpy as np

ftseries = open("timeseries.txt", "a")
ftseries.write("time fluxtot Tmax  \n")

def apply_boundary_conditions():
    # West boundary (inlet)
    for j in range(jmax):
        eta = sigma * (y[j] - ymp) / (x[0] + VO)
        Tn1[0,j] = (1 - tanh(eta)**2)**0.5  # C1 profile
        Tn2[0,j] = 0.0                      # C2 = 0
    
    # East boundary (outflow: zero-gradient)
    Tn1[-1,:] = Tn1[-2,:]
    Tn2[-1,:] = Tn2[-2,:]
    
    # North/South boundaries (Dirichlet)
    Tn1[:,-1] = 0.0
    Tn2[:,-1] = 0.0
    Tn1[:,0] = 0.0
    Tn2[:,0] = 0.0

def upwind_flux(C, velocity_face, direction):
    # Returns upwind value based on flow direction
    if direction == 'x':
        return C[i-1,j] if velocity_face > 0 else C[i,j]
    else:  # y-direction
        return C[i,j-1] if velocity_face > 0 else C[i,j]
    
def harmonic_mean(K1, K2):
    return 2 * K1 * K2 / (K1 + K2) if (K1 + K2) != 0 else 0.0

def diffusion_term(C, K, i, j, imax, jmax, dx, dy):
    """
    Computes the diffusion term for a scalar C at cell (i,j) using harmonic averaging for K.
    Handles all boundaries safely.
    
    Args:
        C: 2D array of scalar concentrations (C₁ or C₂)
        K: 2D array of diffusivities
        i, j: Current cell indices
        imax, jmax: Grid dimensions
        dx, dy: Grid spacing
    Returns:
        Diffusion term for the cell (i,j)
    """
    # --- East face ---
    if i == imax-1:  # East boundary (zero gradient)
        Ke = K[i,j]
        dCdx_e = 0.0  # Zero gradient: C[i+1,j] = C[i,j]
    else:
        Ke = 2 * K[i,j] * K[i+1,j] / (K[i,j] + K[i+1,j] + 1e-12)  # Harmonic mean
        dCdx_e = (C[i+1,j] - C[i,j]) / dx
    
    # --- West face ---
    if i == 0:  # West boundary (Dirichlet handled elsewhere)
        Kw = K[i,j]
        dCdx_w = 0.0  # Placeholder (BC applied separately)
    else:
        Kw = 2 * K[i,j] * K[i-1,j] / (K[i,j] + K[i-1,j] + 1e-12)
        dCdx_w = (C[i,j] - C[i-1,j]) / dx
    
    # --- North face ---
    if j == jmax-1:  # North boundary (Dirichlet C=0)
        Kn = K[i,j]
        dCdy_n = (0.0 - C[i,j]) / dy  # C[i,j+1] = 0
    else:
        Kn = 2 * K[i,j] * K[i,j+1] / (K[i,j] + K[i,j+1] + 1e-12)
        dCdy_n = (C[i,j+1] - C[i,j]) / dy
    
    # --- South face ---
    if j == 0:  # South boundary (Dirichlet C=0)
        Ks = K[i,j]
        dCdy_s = (C[i,j] - 0.0) / dy  # C[i,j-1] = 0
    else:
        Ks = 2 * K[i,j] * K[i,j-1] / (K[i,j] + K[i,j-1] + 1e-12)
        dCdy_s = (C[i,j] - C[i,j-1]) / dy
    
    # Combine all terms
    diffusion = (Ke*dCdx_e - Kw*dCdx_w)/dx + (Kn*dCdy_n - Ks*dCdy_s)/dy
    return diffusion

def calculate_stable_dt(U, V, K, C1, C2, dx, dy, Ar):
    max_U = np.max(np.abs(U))
    max_V = np.max(np.abs(V))
    max_K = np.max(K)
    max_C = max(np.max(C1), np.max(C2))
    
    inv_dt = (max_U/dx + max_V/dy + 2*max_K/dx**2 + 2*max_K/dy**2 + Ar*max_C)
    return 0.4 / inv_dt  # Safety factor 0.4 from PDF

timeswidtheps = 3 # limit turbulent diffusivity to 3 half widths
# started from diffusion_2D_inst.py
imax = 32
jmax = 32

Ubig = 1.
# parameters for obstacle wake
diameter = .01 # diameter of the obstacle creating the wake
VO       = 0.5 # distance of obstacle to the left of x = 0
ymp      = 0.0 # centerpoint obstacle
# free 2D jet
Kbig = diameter*Ubig**2
print("Kbig = ", Kbig)

# fill x with coordinates of points
# the x-points are at positions 0, dx, 2dx,  ... 1-dx, 1
x = np.zeros((imax,1))
y = np.zeros((jmax,1))
xplot = np.zeros((imax,jmax))
yplot = np.zeros((imax,jmax))

dx = 1./(imax-2)
ysize = 1.
dy = ysize/(jmax-2)

yshift = 0.5*ysize # shift so that jet centerline is in the middle
for j in range(0,jmax):
   for i in range(0,imax):
      x[i] = (i-0.5)*dx
      y[j] = (j-0.5)*dy - yshift
      xplot[i,j] = x[i]
      yplot[i,j] = y[j]


# initialise arrays Tn1 and To1 are filled with zero's
Tn1  = np.zeros((imax,jmax))
To1  = np.zeros((imax,jmax))
Tn2  = np.zeros((imax,jmax))
To2  = np.zeros((imax,jmax))
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
      b_half   = eta_half*(x[i]+VO)/sigma
      eta = sigma*(y[j]-ymp)/(x[i]+VO)
    
      U[i,j] = 0.5*sqrt(3.)*sqrt(Kbig*sigma/(x[i]+VO))*(1-tanh(eta)**2)
      V[i,j] = 0.25*sqrt(3.)*sqrt(Kbig/(sigma*(x[i]+VO)))*(2*eta*(1-tanh(eta)**2)-tanh(eta))

      if(abs(y[j]-ymp)<timeswidtheps*b_half):
         Ucl = 0.5*sqrt(3.)*sqrt(Kbig*sigma/(x[i]+VO))
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





# Parameters from vel.py
Ar = 20.0  # Reaction constant (0 or 20)
dt = 0.01   # Initial time step (will be adjusted)
t_end = 5.0  # Total simulation time

# Main time-stepping loop
for t in np.arange(0, t_end, dt):
    # Store old values
    To1[:, :] = Tn1[:, :]
    To2[:, :] = Tn2[:, :]
    
    # Calculate stable time step
    dt = calculate_stable_dt(U, V, K, To1, To2, dx, dy, Ar)
    
    # Update C₁ and C₂ for all interior points
    for i in range(1, imax-1):
        for j in range(1, jmax-1):
            # --- Common terms ---
            # Face velocities (arithmetic average)
            U_e = 0.5 * (U[i,j] + U[i+1,j])  # East face
            U_w = 0.5 * (U[i,j] + U[i-1,j])  # West face
            V_n = 0.5 * (V[i,j] + V[i,j+1])  # North face
            V_s = 0.5 * (V[i,j] + V[i,j-1])  # South face
            
            # --- Advection fluxes (Upwind) ---
            # C₁ advection
            C1_e = To1[i,j] if U_e < 0 else To1[i+1,j]  # East face
            C1_w = To1[i-1,j] if U_w > 0 else To1[i,j]  # West face
            C1_n = To1[i,j] if V_n < 0 else To1[i,j+1]  # North face
            C1_s = To1[i,j-1] if V_s > 0 else To1[i,j]  # South face
            
            adv_C1 = (U_e * C1_e - U_w * C1_w) / dx + (V_n * C1_n - V_s * C1_s) / dy
            
            # C₂ advection (same logic)
            C2_e = To2[i,j] if U_e < 0 else To2[i+1,j]
            C2_w = To2[i-1,j] if U_w > 0 else To2[i,j]
            C2_n = To2[i,j] if V_n < 0 else To2[i,j+1]
            C2_s = To2[i,j-1] if V_s > 0 else To2[i,j]
            
            adv_C2 = (U_e * C2_e - U_w * C2_w) / dx + (V_n * C2_n - V_s * C2_s) / dy
            
            # --- Diffusion terms (Harmonic K) ---
            # Harmonic averages at faces
            K_e = harmonic_mean(K[i,j], K[i+1,j])  # East face
            K_w = harmonic_mean(K[i,j], K[i-1,j])  # West face
            K_n = harmonic_mean(K[i,j], K[i,j+1])  # North face
            K_s = harmonic_mean(K[i,j], K[i,j-1])  # South face
            
            # C₁ diffusion
            diff_C1 = (K_e * (To1[i+1,j] - To1[i,j]) / dx**2 -
                       K_w * (To1[i,j] - To1[i-1,j]) / dx**2 +
                       K_n * (To1[i,j+1] - To1[i,j]) / dy**2 -
                       K_s * (To1[i,j] - To1[i,j-1]) / dy**2)
            
            # C₂ diffusion (same stencil)
            diff_C2 = (K_e * (To2[i+1,j] - To2[i,j]) / dx**2 -
                       K_w * (To2[i,j] - To2[i-1,j]) / dx**2 +
                       K_n * (To2[i,j+1] - To2[i,j]) / dy**2 -
                       K_s * (To2[i,j] - To2[i,j-1]) / dy**2)
            
            # --- Reaction terms ---
            reaction = Ar * To1[i,j] * To2[i,j]
            
            # --- Update equations ---
            Tn1[i,j] = To1[i,j] + dt * (
                - adv_C1    # Advection
                + diff_C1   # Diffusion
                + reaction  # Source: +Ar C₁ C₂
            )
            
            Tn2[i,j] = To2[i,j] + dt * (
                - adv_C2    # Advection
                + diff_C2   # Diffusion
                - reaction  # Sink: -Ar C₁ C₂
                + S2[i,j]   # Source term (from vel.py)
            )
    
    # Apply boundary conditions
    apply_boundary_conditions()
    
    # Visualization (every 0.1s)
    if t % 0.1 == 0:
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.plot_surface(xplot, yplot, Tn1, cmap='viridis')
        ax.set_title(f'C₁ at t={t:.2f}s')
        plt.pause(0.01)
#   plt.clf()
#  plt.contour(xplot,yplot,Tn1,20)
#   ax = plt.axes(projection='3d')
#   ax.plot_surface(xplot,yplot,V)
# axis sets limits z-axis
#  ax.set_zlim3d(0,0.5)
#   plt.plot_surface(xplot,yplot,Tn1)
#   surf(xplot,yplot,Tn1)

# pause 0.1 seconds after plotting

def plot_surface(x, y, C, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, C, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.show()

# Usage (after simulation):
plot_surface(xplot, yplot, Tn1, "C1 Concentration")
plot_surface(xplot, yplot, Tn2, "C2 Concentration")

def plot_slices(x, y, C1, C2):
    # Plot C1 along centerline (y=0)
    plt.figure()
    plt.plot(x, C1[:, jmax//2], 'b-', label='C1 (y=0)')
    plt.xlabel('x')
    plt.ylabel('C1')
    plt.title("C1 Along Jet Centerline")
    plt.legend()
    plt.show()

    # Plot C1 and C2 at x=0.5 (example)
    idx_x = int(0.5 / dx)  # Find index for x=0.5
    plt.figure()
    plt.plot(y, C1[idx_x, :], 'r-', label='C1 at x=0.5')
    plt.plot(y, C2[idx_x, :], 'g-', label='C2 at x=0.5')
    plt.xlabel('y')
    plt.title("Vertical Profiles at x=0.5")
    plt.legend()
    plt.show()

# Usage:
plot_slices(x, y, Tn1, Tn2)

def plot_contour(x, y, C, title):
    plt.figure()
    plt.contourf(x, y, C, levels=20, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

# Usage:
plot_contour(xplot, yplot, Tn1, "C1 Contour")
plot_contour(xplot, yplot, Tn2, "C2 Contour")

plt.show()
