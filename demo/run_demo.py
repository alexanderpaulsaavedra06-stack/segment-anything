# -*- coding: utf-8 -*-mport cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# 1. Cargar imagen
print("Cargando imagen...")
image = cv2.imread('input.png')
if image is None:
    print("Error: No se encontró 'input.png'")
    exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Configurar SAM
sam_checkpoint = "../checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Iniciando SAM en {device}...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 3. Generar máscaras automáticamente
mask_generator = SamAutomaticMaskGenerator(sam)
print("Generando máscaras automáticamente (esto puede tardar unos minutos en CPU)...")
masks = mask_generator.generate(image)

print(f"Proceso completado. Se encontraron {len(masks)} objetos.")

# 4. Guardar resultado visual
plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('output_masks.png', bbox_inches='tight')
print("¡Éxito! El resultado visual se ha guardado como 'output_masks.png' en la carpeta demo.")
