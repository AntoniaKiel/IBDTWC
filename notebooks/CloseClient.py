from obspy.clients.fdsn import RoutingClient
from pathlib import Path
from datetime import datetime, timedelta

# Prüfen, ob EIDA-Token aktuell ist
token_path = Path('./.eidatoken')
if token_path.exists():
    mtime = datetime.fromtimestamp(token_path.stat().st_mtime)
    cutoff = datetime.now() - timedelta(days=30)
    if mtime < cutoff:
        raise RuntimeError(f"{token_path} ist älter als 30 Tage (geändert am {mtime}). Skript wird abgebrochen.")
    else:
        print(f"{token_path} ist neuer oder genau 30 Tage alt (geändert am {mtime}).")

# Client öffnen
client = RoutingClient("eida-routing", credentials={'EIDA_TOKEN': './.eidatoken'})
print("Client erfolgreich geöffnet.")

# Hinweis: RoutingClient muss nicht explizit geschlossen werden
del client
print("Client-Referenz gelöscht. Session ist beendet.")
