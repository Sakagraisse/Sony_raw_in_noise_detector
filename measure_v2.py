import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Paramètres ----------
FILENAME = "sample_1.tiff"
# Taille du crop central (en pixels)
CROP_SIZE = 1024

# Seuils heuristiques pour détecter une NR probable
CORR_THRESHOLD = 0.05  # au-delà, corrélation spatiale notable


def load_image_as_float(path):
    """
    Charge l'image TIFF et renvoie un tableau float64 3D (H, W, C).
    - Si image mono (2D), on ajoute un canal (C=1).
    - Si plus de 3 canaux, on ne garde que les 3 premiers.
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        # Image mono -> (H, W, 1)
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[2] > 3:
        # On ne garde que les 3 premiers canaux
        arr = arr[:, :, :3]
    elif arr.ndim != 3:
        raise ValueError(f"Format d'image non supporté: dimensions {arr.shape}")

    return arr.astype(np.float64)


def center_crop(arr2d, size):
    """Retourne un crop central carré de taille size x size sur une image 2D."""
    h, w = arr2d.shape
    size = min(size, h, w)  # sécurité si l'image est plus petite que CROP_SIZE

    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr2d[y0:y0+size, x0:x0+size]


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
    On renvoie la version centrée (fftshift), ainsi que le spectre S.
    """
    F = np.fft.fft2(noise)
    S = np.abs(F)**2
    R = np.fft.ifft2(S)
    R = np.fft.fftshift(np.real(R))

    center = R[noise.shape[0] // 2, noise.shape[1] // 2]
    if center != 0:
        R /= center

    return R, S


def guess_noise_reduction(rho_x, rho_y):
    """
    Heuristique très simple :
    - si |rho_x| ou |rho_y| > CORR_THRESHOLD → corrélation spatiale notable → NR probable
    - sinon → pas de corrélation flagrante → NR agressive peu probable
    """
    max_corr = max(abs(rho_x), abs(rho_y))

    if max_corr > CORR_THRESHOLD:
        verdict = "Noise reduction PROBABLE (corrélation spatiale significative)"
    else:
        verdict = "Noise reduction PEU PROBABLE (corrélation proche de 0)"

    return verdict, max_corr


def channel_name_from_index(idx, nb_channels):
    """Nom convivial du canal en fonction de l'index."""
    if nb_channels >= 3:
        mapping = {0: "Rouge (R)", 1: "Vert (G)", 2: "Bleu (B)"}
        return mapping.get(idx, f"Canal {idx}")
    else:
        return "Canal unique (mono)"


def main():
    # 1. Chargement
    print(f"Chargement de l'image : {FILENAME}")
    img = load_image_as_float(FILENAME)  # (H, W, C)
    h, w, C = img.shape
    print(f"Image chargée, shape = {img.shape}")

    # On choisit un canal pour la visualisation :
    # - si RGB, on prend le vert (1),
    # - sinon le canal 0.
    if C >= 2:
        vis_channel = 1  # vert
    else:
        vis_channel = 0

    # ---------- Analyse par canal ----------
    print("\n=== Analyse par canal ===")
    results = []

    for c in range(C):
        chan_name = channel_name_from_index(c, C)
        img_c = img[:, :, c]
        img_crop = center_crop(img_c, CROP_SIZE)

        mean_val = np.mean(img_crop)
        noise = img_crop - mean_val
        var = np.mean(noise**2)
        rho_x, rho_y = compute_lag1_correlations(noise)
        verdict, max_corr = guess_noise_reduction(rho_x, rho_y)

        results.append(
            (chan_name, mean_val, var, rho_x, rho_y, verdict, max_corr)
        )

    # Affichage console
    for (chan_name, mean_val, var, rho_x, rho_y, verdict, max_corr) in results:
        print(f"\n--- {chan_name} ---")
        print(f"  Moyenne (crop) : {mean_val:.3f}")
        print(f"  Variance bruit : {var:.3f}")
        print(f"  Corrélation horizontale lag-1 (rho_x) : {rho_x:.4f}")
        print(f"  Corrélation verticale   lag-1 (rho_y) : {rho_y:.4f}")
        print(f"  Corrélation max(|rho_x|,|rho_y|)     : {max_corr:.4f}")
        print(f"  Verdict heuristique NR : {verdict}")

    # ---------- Visualisation sur un canal choisi ----------
    print(f"\nCanal utilisé pour les figures : {channel_name_from_index(vis_channel, C)}")

    img_vis = img[:, :, vis_channel]
    img_crop_vis = center_crop(img_vis, CROP_SIZE)
    mean_vis = np.mean(img_crop_vis)
    noise_vis = img_crop_vis - mean_vis

    R, S = compute_autocorr_2d(noise_vis)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # a) Bruit
    im0 = axs[0].imshow(noise_vis, cmap="gray")
    axs[0].set_title("Bruit (crop, centré)")
    axs[0].axis("off")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # b) Autocorr 2D centrée
    im1 = axs[1].imshow(R, cmap="gray")
    axs[1].set_title("Autocorrélation 2D (centrée, normalisée)")
    axs[1].axis("off")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # c) Spectre de puissance (log)
    S_shift = np.fft.fftshift(S)
    S_log = np.log10(S_shift + 1e-12)
    im2 = axs[2].imshow(S_log, cmap="gray")
    axs[2].set_title("Spectre de puissance (log10)")
    axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
