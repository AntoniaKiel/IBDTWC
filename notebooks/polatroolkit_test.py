import matplotlib.pyplot as plt
from polartoolkit import fetch
from pyproj import Transformer
import numpy as np
from pathlib import Path

# ==============================
# 1. Region definieren
# ==============================
lon_min, lon_max = -20, 0
lat_min, lat_max = -73, -70

transformer = Transformer.from_crs("epsg:4326", "epsg:3031", always_xy=True)
x_min, y_min = transformer.transform(lon_min, lat_min)
x_max, y_max = transformer.transform(lon_max, lat_max)
region_proj = (x_min, x_max, y_min, y_max)

# ==============================
# 2. BEDMAP3 Daten laden
# ==============================
surface = fetch.bedmap3("surface_topography", region=region_proj)
#bed = fetch.bedmap3("bed_topography", region=region_proj)
ice_thickness = fetch.bedmap3("ice_thickness", region=region_proj)

# Mask: 0=Land, 1=Ice, 2=Floating Ice Shelf (kann Grounding Line ableiten)
mask = fetch.bedmap3("mask", region=region_proj)

# ==============================
# 3. Stationsdaten
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
station_path = BASE_DIR / "data" / "stations_meta"
sta_coord = np.load(str(station_path) + '/station_coord.npy')
station_lons, station_lats = sta_coord[0], sta_coord[1]
station_x, station_y = transformer.transform(station_lons, station_lats)

# ==============================
# 4. Plot
# ==============================
fig, ax = plt.subplots(figsize=(12, 9))

# Eisoberfläche als Farbplot
im = ax.imshow(surface, extent=region_proj, origin="upper", cmap="Blues", alpha=0.6)

# Untergrund als Kontur
#ax.contour(bed, levels=20, extent=region_proj, colors="saddlebrown", linewidths=1)

# Grounding Line ableiten: Übergang Mask=1 (Ice) zu Mask=2 (Floating Ice)
# Wir markieren alle Mask==2 Kanten als Grounding Line
ax.contour(mask, levels=[1.5], extent=region_proj, colors="black", linewidths=2)

# Stationspunkte
ax.scatter(station_x, station_y, color="red", s=50, marker="^", label="Stations")

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
fig.colorbar(im, ax=ax, label="Surface Elevation [m]")
plt.title("Antarctic Ice Surfac and Grounding Line with Stations")
ax.legend()
plt.show()


