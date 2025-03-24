import numpy as np
import matplotlib.pyplot as plt
from pyrcel import AerosolSpecies, ParcelModel, Lognorm, binned_activation

# Define the log-normal distribution function
def md(A, x):
    return A[0] / (np.sqrt(2 * np.pi) * np.log(A[2])) * np.exp(-(np.log(x / A[1]))**2 / (2 * np.log(A[2])**2))

# Define the function to sum contributions of multiple log-normal distributions
def logn(A, x):
    return sum(md(a, x) for a in A)

# Define the surface tension of water as a function of temperature
def sigma_w(T):
    return 0.0761 - 1.55e-4 * (T - 273.15)

# Set the temperature in Kelvin
T = 280.01

# Calculate the coefficient used in Köhler theory
Acoeff = (4.0 * 0.018 * sigma_w(T)) / (8.315 * T * 997.0)

# Define the critical supersaturation function
def Sc(Dd, kappa):
    return ((4.0 * (Acoeff)**3.0) / (27.0 * Dd**3 * kappa))**0.5

# Define the supersaturation percentage function
def sc(Dd, kappa):
    return (np.exp(Sc(Dd, kappa)) - 1) * 100

# Set the hygroscopicity parameter
kappa = 0.5

# Define parameters for an aerosol mode
p = [1000.0, 60.0, 1.9]

# Create an aerosol mode object
p1 = AerosolSpecies('m1', distribution=Lognorm(mu=p[1] / 2000, sigma=p[2], N=p[0]), kappa=kappa, bins=100)


# Create an array of aerosol modes
aerosol = [p1]

# Define an array of updraft velocities
ws = [5.0]

# Run the pyrcel model for each updraft velocity
out1 = []
for w in ws:
    model = ParcelModel(aerosol, w, 280.0, -0.02, 1000e2, accom=1.0, console=False)
    parcel_trace, aer_out = model.run(t_end=100.0 / w, output_dt=1.0 / w, solver="cvode", output="dataframes", terminate=True)
    
    Smax = max(parcel_trace['S']) * 100.0
    T_final = parcel_trace['T'].values[-1]
    CDNC = 0
    N01 = 0
    N1 = 0
    Nt = 0
    for i, (_, atrace) in enumerate(aer_out.items()):
        this_total_N = aerosol[i].total_N
        Nt += this_total_N
        f, _, _, _ = binned_activation(Smax / 100.0, T_final, atrace.iloc[-1], aerosol[i])
        CDNC += f * this_total_N
        f, _, _, _ = binned_activation(0.1 / 100.0, T_final, atrace.iloc[-1], aerosol[i])
        N01 += f * this_total_N
        f, _, _, _ = binned_activation(1.0 / 100.0, T_final, atrace.iloc[-1], aerosol[i])
        N1 += f * this_total_N
    out1.append((Smax, CDNC, N01, N1, Nt))

# Extract the maximum supersaturation values
Smax1 = [x[0] for x in out1]

# Extract the cloud droplet number concentration (CDNC) values
CDNC1 = [x[1] for x in out1]

# Extract specific outputs from the first result
N01_1, N1_1, Nt_1 = out1[0][2], out1[0][3], out1[0][4]

# Wrap the aerosol parameters in an array
A = [p]

# Define the edges of diameter bins in nanometers
Dedge = np.logspace(np.log10(10), np.log10(10000), 100)

# Calculate the logarithmic width of each diameter bin
ΔlnD = np.log(Dedge[1:] / Dedge[:-1])

# Calculate the geometric mean diameter for each bin
Dmid = np.sqrt(Dedge[:-1] * Dedge[1:])

# Calculate the number distribution function for each diameter bin
dNdlnDp = np.array([logn(A, D) for D in Dmid])

# Calculate the total number concentration by summing over all bins
N = dNdlnDp * ΔlnD
total = np.sum(N)

# Calculate the critical supersaturation for each diameter bin
ss = np.array([sc(Dd * 1e-9, kappa) for Dd in Dmid])

# Calculate the cumulative number concentration from the largest to smallest bin
cumulative = np.array([np.sum(N[i:]) for i in range(len(N))])

# Create the first plot: number distribution vs. diameter
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(Dmid, dNdlnDp)
plt.xscale("log")
plt.xlim([10, 1000])
plt.xlabel("Diameter (nm)")
plt.ylabel("dN/dlnD (cm⁻³)")
plt.grid(which="both", linestyle="--", linewidth=0.5)

# Create the second plot: CCN number vs. diameter
plt.subplot(1, 3, 2)
plt.plot(Dmid, p[0] - cumulative)
plt.xscale("log")
plt.yscale("log")
plt.xlim([10, 1000])
plt.ylim([10, 2000])
plt.xlabel("Diameter (nm)")
plt.ylabel("CCN Number (cm⁻³)")
plt.grid(which="both", linestyle="--", linewidth=0.5)

# Create the third plot: CCN number vs. critical supersaturation
plt.subplot(1, 3, 3)
plt.plot(ss, cumulative)
plt.scatter([0.1, 1.0], [N01_1, N1_1], color="red")
plt.xscale("log")
plt.yscale("log")
plt.xlim([0.01, 2.0])
plt.ylim([10, 2000])
plt.xlabel("sc (%)")
plt.ylabel("CCN Number (cm⁻³)")
plt.grid(which="both", linestyle="--", linewidth=0.5)

# Save the combined plot to a file
plt.tight_layout()
plt.savefig("testparcel.png")
