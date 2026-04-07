import matplotlib.pyplot as plt
from polartoolkit import fetch
import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ==============================
# 1. Stationsdaten
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
station_path = BASE_DIR / "data" / "stations_meta"
sta_coord = np.load(str(station_path) + '/station_coord.npy')
station_lons, station_lats = sta_coord[0], sta_coord[1]

# ==============================
# 2. Automatische Bounding Box für BEDMAP3
# ==============================
# Mittelpunkt der Stationen
center_lon = np.mean(station_lons)
center_lat = np.mean(station_lats)

# Puffer in Grad, damit die Region nicht zu klein wird
buffer_lon = 1.0  # 1° west/east
buffer_lat = 1.0  # 1° south/north

# Min/Max Koordinaten mit Puffer
lon_min = np.min(station_lons) - buffer_lon
lon_max = np.max(station_lons) + buffer_lon
lat_min = np.min(station_lats) - buffer_lat
lat_max = np.max(station_lats) + buffer_lat

# Sicherstellen, dass Lat/Lon gültig für Antarktika sind
lat_min = max(lat_min, -90)
lat_max = min(lat_max, -60)
lon_min = max(lon_min, -180)
lon_max = min(lon_max, 180)

region_lonlat = (lon_min, lon_max, lat_min, lat_max)

# ==============================
# 3. BEDMAP3 Daten laden in Lat/Lon
# ==============================
surface = fetch.bedmap3("surface_topography", region=region_lonlat, crs="epsg:4326")
mask = fetch.bedmap3("mask", region=region_lonlat, crs="epsg:4326")
mask = np.nan_to_num(mask, nan=-1)

# ==============================
# 4. Farbdefinition für Masken
# ==============================
mask_colors = ListedColormap([
    "#4A90E2",  # 0 Ocean
    "#B8E986",  # 1 Grounded Ice
    "#F5A623",  # 2 Floating Ice Shelf
    "#8B572A"   # 3 Rock / No Ice
])
mask_labels = ["Ocean", "Grounded Ice", "Floating Ice Shelf", "Rock / No Ice"]
bounds = [0, 1, 2, 3, 4]
norm = BoundaryNorm(bounds, mask_colors.N)

# ==============================
# 5. Plot: Lat/Lon + Zoom auf Stationen
# ==============================
fig, ax = plt.subplots(figsize=(12, 9))

# Eisoberfläche
ax.imshow(surface, extent=region_lonlat, origin="lower", cmap="Blues", alpha=0.5)

# Maskenfläche
ax.imshow(mask, extent=region_lonlat, origin="lower", cmap=mask_colors, norm=norm, alpha=0.4)

# Grounding Line (Übergang Mask=1 -> Mask=2)
ax.contour(mask, levels=[1.5], extent=region_lonlat, colors="black", linewidths=2)

# Stationen
ax.scatter(station_lons, station_lats, color="red", s=50, marker="^", label="Stations")

# Achsenlimits auf Stationen + Puffer setzen (Zoom)
zoom_buffer_lon = 1.0  # Grad
zoom_buffer_lat = 1.0
ax.set_xlim(center_lon - zoom_buffer_lon, center_lon + zoom_buffer_lon)
ax.set_ylim(center_lat - zoom_buffer_lat, center_lat + zoom_buffer_lat)

# Legend
handles = [Patch(facecolor=mask_colors(i), label=mask_labels[i]) for i in range(len(mask_labels))]
ax.legend(handles=handles + [Patch(facecolor='none', edgecolor='black', label='Grounding Line'),
                            Patch(facecolor='red', label='Stations')],
          loc="lower right")

ax.set_title("Zoomed Ice Surface + Mask + Grounding Line + Stations (Lat/Lon)")
ax.set_xlabel("Longitude [°]")
ax.set_ylabel("Latitude [°]")

plt.show()


