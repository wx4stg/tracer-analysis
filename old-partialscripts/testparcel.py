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
T = 298.15

# Calculate the coefficient used in Köhler theory
Acoeff = (4.0 * 0.018 * sigma_w(T)) / (8.315 * T * 997.0)

# Define the critical supersaturation function
def Sc(Dd, kappa):
    return ((4.0 * (Acoeff)**3.0) / (27.0 * Dd**3 * kappa))**0.5

# Define the supersaturation percentage function
def sc(Dd, kappa):
    return (np.exp(Sc(Dd, kappa)) - 1) * 100

# Set the hygroscopicity parameter
pkappas = [0.04, 0.25875964523372785]

# Define parameters for an aerosol mode
# N, μ, σ
A = []
p1 = [2274.66078738224, 36.9919734655319, 1.96212333365330]
A.append(p1)
p2 = [317.667960078024, 51.7260638738924, 1.24343755856041]
A.append(p2)
p3 = [345.720047063653, 162.61005715237533, 1.4963594692121904]
A.append(p3)
p4 = [0.5364362582682796, 800, 2.0777829253133966]
A.append(p4)

# Create an aerosol mode object
aerosol = []
kappas = []
for i, p in enumerate(A):
    # Create an AerosolSpecies object with the given parameters
    if i > 1:
        this_kappa = pkappas[1]
    else:
        this_kappa = pkappas[0]
    aerosol.append(AerosolSpecies(f'mode{i+1}', distribution=Lognorm(mu=p[1] / 2000, sigma=p[2], N=p[0]), kappa=this_kappa, bins=100))
    if this_kappa not in kappas:
        kappas.append(this_kappa)

# Define an array of updraft velocities
ws = [5.0]

# Run the pyrcel model for each updraft velocity
out1 = []
for w in ws:
    model = ParcelModel(aerosol, w, T, -0.02, 1000e2, accom=1.0, console=False)
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

out1arr = np.array(out1)

# Extract the maximum supersaturation values
Smax1 = out1arr[:, 0]

# Extract the cloud droplet number concentration (CDNC) values
CDNC1 = out1arr[:, 1]

# Extract specific outputs from the first result
N01_1 = np.nansum(out1arr[:, 2])
N1_1 = np.nansum(out1arr[:, 3])
Nt_1 = np.nansum(out1arr[:, 4])


# Define the edges of diameter bins in nanometers
Dedge = np.logspace(np.log10(1), np.log10(10000), 100)

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
ss = np.zeros(len(Dmid))
for kappa in kappas:
    ss = ss + np.array([sc(Dd * 1e-9, kappa) for Dd in Dmid])

# Calculate the cumulative number concentration from the largest to smallest bin
cumulative = np.array([np.sum(N[i:]) for i in range(len(N))])

# Create the first plot: number distribution vs. diameter
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(Dmid, dNdlnDp)
plt.xscale("log")
plt.xlim([1, 1000])
plt.xlabel("Diameter (nm)")
plt.ylabel("dN/dlnD (cm⁻³)")
plt.grid(which="both", linestyle="--", linewidth=0.5)

# Create the second plot: CCN number vs. diameter
plt.subplot(1, 3, 2)
plt.plot(Dmid, np.sum(A, axis=0)[0] - cumulative)
plt.xscale("log")
plt.yscale("log")
plt.xlim([.1, 1000])
plt.xlabel("Diameter (nm)")
plt.ylabel("CCN Number (cm⁻³)")
plt.grid(which="both", linestyle="--", linewidth=0.5)

# Create the third plot: CCN number vs. critical supersaturation
plt.subplot(1, 3, 3)
plt.plot(ss, cumulative)
print(Smax1, N01_1, N1_1)
plt.scatter([0.1, 1.0], [N01_1, N1_1], color="red")
plt.scatter([0.1, 1.0], [105.4672808, 1453.627972], color="blue")
plt.xscale("log")
plt.yscale("log")
plt.xlim([0.01, 2.0])
plt.ylim([.1, 3000])
plt.xlabel("sc (%)")
plt.ylabel("CCN Number (cm⁻³)")
plt.grid(which="both", linestyle="--", linewidth=0.5)

# Save the combined plot to a file
plt.tight_layout()
plt.savefig("testparcel.png")
