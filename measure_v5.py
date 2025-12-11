import numpy as np
import rawpy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # pour retirer la basse fréquence

# ---------- Paramètres ----------
FILENAME = "10s.ARW"          # ton fichier RAW
CROP_SIZE = 1024              # taille du crop central
CORR_THRESHOLD = 0.05         # seuil heuristique pour la "NR probable"
GAUSS_SIGMA = 20.0            # échelle du flou pour la composante basse fréquence


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
    """
    raw = rawpy.imread(path)

    # Image RAW visible (2D) en float64
    data = raw.raw_image_visible.astype(np.float64)

    # Pattern Bayer 2x2, indices 0..3
    pattern = raw.raw_pattern       # shape (2, 2)
    # Mapping d'indice -> lettre ('R','G','B', éventuellement autre)
    colors = raw.color_desc.decode("ascii")  # ex: "RGBG"

    H, W = data.shape

    # Listes de sous-plans par couleur
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


def highpass_remove_lowfreq(img, sigma=GAUSS_SIGMA):
    """
    Retire la composante basse fréquence par flou gaussien large :
      img_hp = img - gaussian_filter(img, sigma)
    On peut l'appliquer soit au signal brut, soit au bruit.
    Ici on l'utilise sur le signal (plane_crop) avant recentrage.
    """
    low = gaussian_filter(img, sigma=sigma)
    high = img - low
    return high


def clean_dark_frame_pattern(img):
    """
    Prépare un Dark Frame (image noire) pour l'analyse de bruit spatial.
    Traite l'image comme un plan monochrome unique mais retire les artefacts fixes :
    1. Pattern 2x2 (différences de Black Level entre les canaux CFA).
    2. Banding horizontal et vertical (Row/Col noise).
    """
    img = img.astype(np.float64)
    h, w = img.shape

    # 1. Normalisation du Black Level par phase de Bayer (2x2)
    # Cela retire la grille fixe si les canaux ont des offsets différents
    for y in range(2):
        for x in range(2):
            sub = img[y::2, x::2]
            img[y::2, x::2] -= np.mean(sub)

    # 2. Destriping (Retrait du bruit de ligne et de colonne)
    # On retire la moyenne de chaque ligne et de chaque colonne.
    # C'est crucial car le banding crée une fausse corrélation très forte.
    
    # Retrait moyenne par ligne (Row Noise)
    row_means = np.mean(img, axis=1, keepdims=True)
    img -= row_means

    # Retrait moyenne par colonne (Col Noise)
    col_means = np.mean(img, axis=0, keepdims=True)
    img -= col_means

    return img


# ---------- Main ----------

def main():
    print(f"Lecture du RAW : {FILENAME}")
    
    # --- Analyse 1 : Canaux séparés (existante) ---
    print("\n--- MODE 1 : Analyse par Canaux CFA (R, G, B séparés) ---")
    planes = load_raw_cfa_planes(FILENAME)

    for c in ['R', 'G', 'B']:
        if c not in planes:
            continue

        plane = planes[c]
        h, w = plane.shape
        print(f"\n=== Canal {c} — taille plane CFA : {h} x {w} ===")

        # Crop central
        plane_crop = center_crop(plane, CROP_SIZE)

        # --- 1) Bruit "brut" (juste - moyenne globale) ---
        mean_val = np.mean(plane_crop)
        noise = plane_crop - mean_val
        var = np.mean(noise**2)

        rho_x, rho_y = compute_lag1_correlations(noise)
        verdict, max_corr = guess_noise_reduction(rho_x, rho_y)

        print(f"  [Brut] Moyenne (crop) : {mean_val:.3f}")
        print(f"  [Brut] Variance bruit : {var:.3f}")
        print(f"  [Brut] Corrélation horizontale lag-1 (rho_x) : {rho_x:.4f}")
        print(f"  [Brut] Corrélation verticale   lag-1 (rho_y) : {rho_y:.4f}")
        print(f"  [Brut] Corrélation max(|rho_x|,|rho_y|)     : {max_corr:.4f}")
        print(f"  [Brut] Verdict heuristique NR : {verdict}")

        # Autocorr 2D + spectre pour le bruit brut
        R_raw, S_raw = compute_autocorr_2d(noise)

        # --- 2) Bruit "high-pass" (structure basse fréquence retirée) ---
        plane_hp = highpass_remove_lowfreq(plane_crop, sigma=GAUSS_SIGMA)
        plane_hp = plane_hp - np.mean(plane_hp)   # recentrage
        var_hp = np.mean(plane_hp**2)

        rho_x_hp, rho_y_hp = compute_lag1_correlations(plane_hp)
        verdict_hp, max_corr_hp = guess_noise_reduction(rho_x_hp, rho_y_hp)

        print(f"  [HP]   Variance bruit HP : {var_hp:.3f}")
        print(f"  [HP]   Corrélation horizontale lag-1 (rho_x) : {rho_x_hp:.4f}")
        print(f"  [HP]   Corrélation verticale   lag-1 (rho_y) : {rho_y_hp:.4f}")
        print(f"  [HP]   Corrélation max(|rho_x|,|rho_y|)     : {max_corr_hp:.4f}")
        print(f"  [HP]   Verdict heuristique NR : {verdict_hp}")

        # Autocorr 2D + spectre pour le bruit high-pass
        R_hp, S_hp = compute_autocorr_2d(plane_hp)
        cmap = channel_cmap(c)

        # ---------- Figures : ligne du haut = brut, ligne du bas = high-pass ----------
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Canal {c} (CFA) — Brut vs High-pass (sigma={GAUSS_SIGMA})", fontsize=14)

        # a1) Bruit brut
        im0 = axs[0, 0].imshow(noise, cmap=cmap)
        axs[0, 0].set_title("Bruit brut (crop, centré)")
        axs[0, 0].axis("off")
        plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        # b1) Autocorr 2D brut
        im1 = axs[0, 1].imshow(R_raw, cmap=cmap)
        axs[0, 1].set_title("Autocorr 2D brut (centrée, norm.)")
        axs[0, 1].axis("off")
        plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

        # c1) Spectre brut
        S_raw_shift = np.fft.fftshift(S_raw)
        S_raw_log = np.log10(S_raw_shift + 1e-12)
        im2 = axs[0, 2].imshow(S_raw_log, cmap=cmap)
        axs[0, 2].set_title("Spectre brut (log10)")
        axs[0, 2].axis("off")
        plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

        # a2) Bruit high-pass
        im3 = axs[1, 0].imshow(plane_hp, cmap=cmap)
        axs[1, 0].set_title("Bruit high-pass (basse fréquence retirée)")
        axs[1, 0].axis("off")
        plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

        # b2) Autocorr 2D high-pass
        im4 = axs[1, 1].imshow(R_hp, cmap=cmap)
        axs[1, 1].set_title("Autocorr 2D high-pass (centrée, norm.)")
        axs[1, 1].axis("off")
        plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

        # c2) Spectre high-pass
        S_hp_shift = np.fft.fftshift(S_hp)
        S_hp_log = np.log10(S_hp_shift + 1e-12)
        im5 = axs[1, 2].imshow(S_hp_log, cmap=cmap)
        axs[1, 2].set_title("Spectre high-pass (log10)")
        axs[1, 2].axis("off")
        plt.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    # --- Analyse 2 : Full Sensor (Dark Frame optimisé) ---
    print("\n--- MODE 2 : Analyse Full Sensor (Dark Frame / Monochrome) ---")
    print("Ce mode ignore la matrice de Bayer et analyse les pixels voisins immédiats.")
    print("Il applique un 'Destriping' pour retirer le bruit de ligne/colonne.")
    
    raw = rawpy.imread(FILENAME)
    full_raw = raw.raw_image_visible
    
    # Crop d'abord pour gagner du temps, mais attention : 
    # pour bien retirer le banding, il vaut mieux avoir toute la largeur/hauteur.
    # Ici on travaille sur le crop pour la rapidité, mais idéalement on nettoie avant crop.
    # On va nettoyer le crop directement, en supposant que le banding est localement constant.
    full_crop = center_crop(full_raw, CROP_SIZE)
    
    # Nettoyage spécifique Dark Frame
    clean_noise = clean_dark_frame_pattern(full_crop)
    
    var_clean = np.mean(clean_noise**2)
    rho_x_cl, rho_y_cl = compute_lag1_correlations(clean_noise)
    verdict_cl, max_corr_cl = guess_noise_reduction(rho_x_cl, rho_y_cl)

    print(f"  [Full-Clean] Variance bruit : {var_clean:.3f}")
    print(f"  [Full-Clean] Corrélation horizontale lag-1 (rho_x) : {rho_x_cl:.4f}")
    print(f"  [Full-Clean] Corrélation verticale   lag-1 (rho_y) : {rho_y_cl:.4f}")
    print(f"  [Full-Clean] Corrélation max(|rho_x|,|rho_y|)     : {max_corr_cl:.4f}")
    print(f"  [Full-Clean] Verdict : {verdict_cl}")

    # Autocorr
    R_cl, S_cl = compute_autocorr_2d(clean_noise)

    # Figure Full Sensor
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Full Sensor (Dark Frame Cleaned) — Analyse pixels voisins", fontsize=14)

    im0 = axs[0].imshow(clean_noise, cmap='gray', vmin=-3*np.sqrt(var_clean), vmax=3*np.sqrt(var_clean))
    axs[0].set_title("Bruit nettoyé (Destriped)")
    axs[0].axis("off")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(R_cl, cmap='gray')
    axs[1].set_title("Autocorr 2D")
    axs[1].axis("off")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    S_cl_shift = np.fft.fftshift(S_cl)
    S_cl_log = np.log10(S_cl_shift + 1e-12)
    im2 = axs[2].imshow(S_cl_log, cmap='gray')
    axs[2].set_title("Spectre (log10)")
    axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
