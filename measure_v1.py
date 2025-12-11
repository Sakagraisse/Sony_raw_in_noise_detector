import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Paramètres ----------
FILENAME = "sample_1.tiff"
# taille du crop central (en pixels). Ajuste si besoin.
CROP_SIZE = 1024


def load_image_as_float(path):
    """Charge l'image TIFF et renvoie un tableau float64 2D (un seul canal)."""
    img = Image.open(path)
    arr = np.array(img)

    # Si l'image est RGB ou RGBA, on prend le canal vert
    if arr.ndim == 3:
        # arr.shape = (H, W, C)
        # Canal vert: index 1 si format standard RGB
        arr = arr[:, :, 1]
    elif arr.ndim != 2:
        raise ValueError(f"Format d'image non supporté: dimensions {arr.shape}")

    # Conversion en float64
    arr = arr.astype(np.float64)
    return arr


def center_crop(arr, size):
    """Retourne un crop central carré de taille size x size."""
    h, w = arr.shape
    size = min(size, h, w)  # sécurité si l'image est plus petite que CROP_SIZE

    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr[y0:y0+size, x0:x0+size]


def compute_lag1_correlations(noise):
    """
    Calcule la corrélation de lag 1 horizontale et verticale :
    rho_x(1) = E[n(x,y)n(x+1,y)] / E[n(x,y)^2]
    rho_y(1) = E[n(x,y)n(x,y+1)] / E[n(x,y)^2]
    """
    var = np.mean(noise**2)
    if var == 0:
        return 0.0, 0.0

    # horizontale : voisins (x, y) et (x+1, y)
    prod_x = noise[:, :-1] * noise[:, 1:]
    rho_x = np.mean(prod_x) / var

    # verticale : voisins (x, y) et (x, y+1)
    prod_y = noise[:-1, :] * noise[1:, :]
    rho_y = np.mean(prod_y) / var

    return rho_x, rho_y


def compute_autocorr_2d(noise):
    """
    Autocorrélation 2D via FFT :
    R = iFFT( |FFT(noise)|^2 ), normalisée telle que R(0,0)=1.
    On renvoie la version centrée (fftshift).
    """
    # FFT
    F = np.fft.fft2(noise)
    # spectre de puissance
    S = np.abs(F)**2
    # autocorr brute
    R = np.fft.ifft2(S)
    R = np.fft.fftshift(np.real(R))

    # normalisation
    center = R[noise.shape[0] // 2, noise.shape[1] // 2]
    if center != 0:
        R /= center

    return R, S


def main():
    # 1. Chargement
    print(f"Chargement de l'image : {FILENAME}")
    img = load_image_as_float(FILENAME)
    print(f"Image chargée, shape = {img.shape}")

    # 2. Crop central
    img_crop = center_crop(img, CROP_SIZE)
    print(f"Crop central utilisé, shape = {img_crop.shape}")

    # 3. Extraction du bruit (soustraction de la moyenne)
    mean_val = np.mean(img_crop)
    noise = img_crop - mean_val
    var = np.mean(noise**2)
    print(f"Valeur moyenne du crop : {mean_val:.3f}")
    print(f"Variance du bruit (approx.) : {var:.3f}")

    # 4. Corrélations lag-1
    rho_x, rho_y = compute_lag1_correlations(noise)
    print(f"Corrélation horizontale lag-1 (rho_x): {rho_x:.4f}")
    print(f"Corrélation verticale   lag-1 (rho_y): {rho_y:.4f}")

    # 5. Autocorr 2D + spectre
    R, S = compute_autocorr_2d(noise)

    # 6. Visualisation
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # a) Bruit
    im0 = axs[0].imshow(noise, cmap="gray")
    axs[0].set_title("Bruit (crop, centré)")
    axs[0].axis("off")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # b) Autocorr 2D centrée
    im1 = axs[1].imshow(R, cmap="gray")
    axs[1].set_title("Autocorrélation 2D (centrée, normalisée)")
    axs[1].axis("off")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # c) Spectre de puissance (log)
    S_log = np.log10(np.fft.fftshift(S) + 1e-12)
    im2 = axs[2].imshow(S_log, cmap="gray")
    axs[2].set_title("Spectre de puissance (log10)")
    axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
