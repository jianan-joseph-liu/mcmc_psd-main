import numpy as np
from src.sgvb_psd.psd_estimator import PSDEstimator
import matplotlib.pyplot as plt
import time

L = 4096*4
FMIN, FMAX = 0.0001, 10 ** -1

# import the file path of one year lisa noise data
file_path = r'C:\lisa_noise_psd\data\lisa_data.npz'
with np.load(file_path, allow_pickle=True) as z:
    data = z["data"]
    freq_true = z["freq_true"]
    true_matrix_freq = z["true_matrix_freq"] 
    true_matrix = z['true_matrix']
    delta_t = float(z["delta_t"])
    model = z["model"]
    use_freq_units = z["use_freq_units"]
    Nb = z["Nb"]
    Lb = z["Lb"]
    block_seconds = z["block_seconds"]

# window the noise dataset by chunks
def chunk_hann_reconcat(time_series: np.ndarray, L: int = 4096):
    ts = np.asarray(time_series)
    N, C = ts.shape

    n_chunks = N // L
    N_used = n_chunks * L

    # Cut to full chunks
    ts_use = ts[:N_used, :]
    # Reshape to (n_chunks, L, C)
    chunks = ts_use.reshape(n_chunks, L, C)
    
    # Mean-center EACH chunk, EACH channel
    chunk_means = chunks.mean(axis=1, keepdims=True)
    chunks_mc = chunks - chunk_means

    # Hann window (length L), apply along time axis
    w = np.hanning(L).reshape(1, L, 1)
    Ew = np.mean(w**2)
    chunks_win = chunks_mc * w

    # Concatenate back to (N_used, C)
    ts_win = chunks_win.reshape(N_used, C)

    return ts_win, Ew, n_chunks

ts_win, Ew, n_chunks = chunk_hann_reconcat(data, L)

# welch PSD
def welch_spectral_matrix_xyz(x, y, z, L, dt, overlap=0.5):
    N = len(x)
    step = int(L * (1 - overlap))
    w = np.hanning(L)
    U = np.mean(w**2)

    Sxx = Syy = Szz = 0
    Sxy = Syz = Szx = 0
    count = 0

    for start in range(0, N - L + 1, step):
        xs = x[start:start+L] * w
        ys = y[start:start+L] * w
        zs = z[start:start+L] * w

        Xf = np.fft.rfft(xs)
        Yf = np.fft.rfft(ys)
        Zf = np.fft.rfft(zs)

        scale = 2.0 * dt / (L * U)

        Sxx += scale * (np.abs(Xf)**2)
        Syy += scale * (np.abs(Yf)**2)
        Szz += scale * (np.abs(Zf)**2)

        Sxy += scale * (Xf * np.conj(Yf))
        Syz += scale * (Yf * np.conj(Zf))
        Szx += scale * (Zf * np.conj(Xf))

        count += 1

    Sxx /= count
    Syy /= count
    Szz /= count
    Sxy /= count
    Syz /= count
    Szx /= count

    freq = np.fft.rfftfreq(L, d=dt)
    return freq, Sxx, Syy, Szz, Sxy, Syz, Szx

# SGVB PSD
lr = 0.001
N_theta = 250
start_time = time.time()
optim = PSDEstimator(
    x=ts_win,
    N_theta=N_theta,
    nchunks=n_chunks,
    fs=1/delta_t,
    ntrain_map=10000,
    n_elbo_maximisation_steps=600,
    fmin_for_analysis=FMIN,
    fmax_for_analysis=FMAX,
    degree_fluctuate=N_theta,
    seed=0,
)

optim.run(lr=lr)
end_time = time.time()
estimation_time = end_time - start_time
print(f'The estimation time is {estimation_time:.2f}s')

psd_matrices = optim.pointwise_ci
sgvb_med = psd_matrices[1]*2/Ew
freq_sgvb = optim.freq

# plot the SGVB PSD, true PSD, welch PSD
def coherence(Sii, Sjj, Sij):
    return np.abs(Sij) / np.sqrt(Sii * Sjj)

def plot_psd_coherence(freq_true, S_true, freq_emp, S_emp, freq_sgvb, S_sgvb):
    # ---------- True ----------
    Sxx_true, Syy_true, Szz_true = S_true[:,0,0], S_true[:,1,1], S_true[:,2,2]
    Sxy_true, Syz_true, Szx_true = S_true[:,0,1], S_true[:,1,2], S_true[:,2,0]

    coh_xy_true = coherence(Sxx_true, Syy_true, Sxy_true)
    coh_yz_true = coherence(Syy_true, Szz_true, Syz_true)
    coh_zx_true = coherence(Szz_true, Sxx_true, Szx_true)

    # ---------- Welch empirical ----------
    Sxx_emp, Syy_emp, Szz_emp = S_emp["Sxx"], S_emp["Syy"], S_emp["Szz"]
    Sxy_emp, Syz_emp, Szx_emp = S_emp["Sxy"], S_emp["Syz"], S_emp["Szx"]

    coh_xy_emp = coherence(Sxx_emp, Syy_emp, Sxy_emp)
    coh_yz_emp = coherence(Syy_emp, Szz_emp, Syz_emp)
    coh_zx_emp = coherence(Szz_emp, Sxx_emp, Szx_emp)

    # ---------- SGVB ----------
    Sxx_sgvb, Syy_sgvb, Szz_sgvb = S_sgvb[:,0,0], S_sgvb[:,1,1], S_sgvb[:,2,2]
    Sxy_sgvb, Syz_sgvb, Szx_sgvb = S_sgvb[:,0,1], S_sgvb[:,1,2], S_sgvb[:,2,0]
    
    coh_xy_sgvb = coherence(Sxx_sgvb, Syy_sgvb, Sxy_sgvb)
    coh_yz_sgvb = coherence(Syy_sgvb, Szz_sgvb, Syz_sgvb)
    coh_zx_sgvb = coherence(Szz_sgvb, Sxx_sgvb, Szx_sgvb)
    
    # ---------- Plotting ----------
    channels = ["X", "Y", "Z"]
    true_psd = [Sxx_true, Syy_true, Szz_true]
    emp_psd = [Sxx_emp,  Syy_emp,  Szz_emp]
    sgvb_psd = [Sxx_sgvb, Syy_sgvb, Szz_sgvb]
    
    true_coh = [[None, coh_xy_true, coh_zx_true],
                [coh_xy_true, None, coh_yz_true],
                [coh_zx_true, coh_yz_true, None]]
    emp_coh = [[None, coh_xy_emp, coh_zx_emp],
               [coh_xy_emp, None, coh_yz_emp],
               [coh_zx_emp, coh_yz_emp, None]]
    sgvb_coh = [[None, coh_xy_sgvb, coh_zx_sgvb],
                [coh_xy_sgvb, None, coh_yz_sgvb],
                [coh_zx_sgvb, coh_yz_sgvb, None]]

    fig, ax = plt.subplots(3, 3, figsize=(12,10))

    for i in range(3):
        for j in range(3):
            a = ax[i,j]

            if i < j:
                a.axis("off")
                continue

            if i == j:
                a.loglog(freq_true, true_psd[i], label="True", color="blue", alpha=0.6)
                a.loglog(freq_emp, emp_psd[i], label="Welch", color="green", alpha=0.6)
                a.loglog(freq_sgvb, sgvb_psd[i], label="SGVB",  color="red", alpha=1.0)
                #a.set_ylim(10**-50, 10**-32)
                a.set_title(f"{channels[i]} PSD")
                a.grid(True, which="both", alpha=0.3)
                if i == 0:
                    a.legend()
                continue

            a.semilogx(freq_true, true_coh[i][j], label="True coh", color="blue", alpha=0.6)
            a.semilogx(freq_emp, emp_coh[i][j], label="Welch coh", color="green", alpha=0.6)
            a.semilogx(freq_sgvb, sgvb_coh[i][j], label="SGVB coh",  color="red", alpha=1.0)
            a.set_ylim(0,1.05)
            a.grid(True, which="both", alpha=0.3)
            a.set_title(f"{channels[i]}–{channels[j]}")
            if i == 1 and j == 0:
                a.legend()

    plt.tight_layout()
    fig.savefig("lisa_one_year_psd_eigen.pdf", bbox_inches="tight")
    plt.show()


# Welch
freq_emp, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
    data[:,0], data[:,1], data[:,2], L=4096*4, dt=delta_t, overlap=0.5
)

S_emp = dict(Sxx=Sxx, Syy=Syy, Szz=Szz, Sxy=Sxy, Syz=Syz, Szx=Szx)

plot_psd_coherence(freq_true, true_matrix, freq_emp, S_emp, freq_sgvb, sgvb_med)

np.savez_compressed(
    "lisa_one_year_psd_eigen.npz",
    sgvb_med=sgvb_med,
    freq_sgvb=freq_sgvb,
)

