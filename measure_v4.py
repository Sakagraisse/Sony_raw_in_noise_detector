import numpy as np
import rawpy
import matplotlib.pyplot as plt

# ---------- Paramètres ----------
FILENAME = "10s.ARW"
# Taille du crop central (en pixels)
CROP_SIZE = 1024

# Seuil heuristique pour détecter une NR probable
CORR_THRESHOLD = 0.05  # au-delà, corrélation spatiale notable


# ---------- Lecture RAW & extraction CFA ----------

def load_raw_cfa_planes(path):
    """
    Lit un fichier RAW (.ARW) avec rawpy et extrait les plans R, G, B
    directement à partir de la mosaïque CFA, sans dématriçage.

    Retourne un dict:
      {
        'R': array2D,
        'G': array2D,
        'B': array2D
      }
    Chaque plan est de taille ≈ (H/2, W/2).
    Si un canal est absent dans le pattern (cas exotique), il n'est pas renvoyé.
    """
    raw = rawpy.imread(path)

    # Image RAW visible (2D) en float64
    data = raw.raw_image_visible.astype(np.float64)

    # Pattern Bayer 2x2, indices 0..3
    pattern = raw.raw_pattern       # shape (2, 2)
    # Mapping d'indice -> lettre ('R','G','B', éventuellement autre)
    colors = raw.color_desc.decode("ascii")  # ex: "RGBG"

    H, W = data.shape
    h2, w2 = H // 2, W // 2

    # Liste des sous-plans par couleur
    planes_lists = {'R': [], 'G': [], 'B': []}

    # Parcours des 2x2 positions du motif CFA
    for py in range(2):
        for px in range(2):
            cidx = pattern[py, px]
            cchar = colors[cidx].upper()  # 'R','G','B' ou autre

            # Sous-échantillonnage : on prend un pixel sur deux dans chaque direction
            sub = data[py:H:2, px:W:2]

            if cchar in planes_lists:
                planes_lists[cchar].append(sub)

    # Fusion des sous-plans (ex: G1, G2 -> moyenne)
    planes = {}
    for cchar, plist in planes_lists.items():
        if len(plist) == 0:
            continue
        elif len(plist) == 1:
            plane = plist[0]
        else:
            # Moyenne des sous-plans (par ex. G1, G2)
            acc = np.zeros_like(plist[0], dtype=np.float64)
            for p in plist:
                acc += p
            plane = acc / len(plist)
        planes[cchar] = plane

    return planes


# ---------- Fonctions d'analyse ----------

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
    - sinon → NR agressive peu probable
    """
    max_corr = max(abs(rho_x), abs(rho_y))

    if max_corr > CORR_THRESHOLD:
        verdict = "Noise reduction PROBABLE (corrélation spatiale significative)"
    else:
        verdict = "Noise reduction PEU PROBABLE (corrélation proche de 0)"

    return verdict, max_corr


def channel_cmap(color):
    """
    Colormap 'niveau de gris teinté' en fonction du canal :
    - 'R' -> 'Reds'
    - 'G' -> 'Greens'
    - 'B' -> 'Blues'
    """
    if color == 'R':
        return "Reds"
    elif color == 'G':
        return "Greens"
    elif color == 'B':
        return "Blues"
    else:
        return "gray"


# ---------- Main ----------

def main():
    print(f"Lecture du RAW et extraction CFA : {FILENAME}")
    planes = load_raw_cfa_planes(FILENAME)

    for c in ['R', 'G', 'B']:
        if c not in planes:
            continue

        plane = planes[c]
        h, w = plane.shape
        print(f"\n=== Canal {c} — taille plane CFA : {h} x {w} ===")

        # Crop central
        plane_crop = center_crop(plane, CROP_SIZE)

        # Bruit = image - moyenne
        mean_val = np.mean(plane_crop)
        noise = plane_crop - mean_val
        var = np.mean(noise**2)

        rho_x, rho_y = compute_lag1_correlations(noise)
        verdict, max_corr = guess_noise_reduction(rho_x, rho_y)

        print(f"  Moyenne (crop) : {mean_val:.3f}")
        print(f"  Variance bruit : {var:.3f}")
        print(f"  Corrélation horizontale lag-1 (rho_x) : {rho_x:.4f}")
        print(f"  Corrélation verticale   lag-1 (rho_y) : {rho_y:.4f}")
        print(f"  Corrélation max(|rho_x|,|rho_y|)     : {max_corr:.4f}")
        print(f"  Verdict heuristique NR : {verdict}")

        # Autocorr 2D + spectre
        R, S = compute_autocorr_2d(noise)
        cmap = channel_cmap(c)

        # Figures
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Canal {c} (CFA) — cmap: {cmap}", fontsize=14)

        # a) Bruit
        im0 = axs[0].imshow(noise, cmap=cmap)
        axs[0].set_title("Bruit (crop, centré)")
        axs[0].axis("off")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        # b) Autocorr 2D centrée
        im1 = axs[1].imshow(R, cmap=cmap)
        axs[1].set_title("Autocorrélation 2D (centrée, normalisée)")
        axs[1].axis("off")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # c) Spectre de puissance (log)
        S_shift = np.fft.fftshift(S)
        S_log = np.log10(S_shift + 1e-12)
        im2 = axs[2].imshow(S_log, cmap=cmap)
        axs[2].set_title("Spectre de puissance (log10)")
        axs[2].axis("off")
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
