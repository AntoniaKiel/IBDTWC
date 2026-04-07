from datetime import datetime, timezone
from pathlib import Path
import logging

def find_folders_to_process(raw_data_path: Path, events_base: Path,
                            start_date: str, end_date: str) -> list[Path]:
    """ collects all daily folders of raw_data, which do not related to any eventXYZ folders in /detected_events/

    Parameters
    ----------
    raw_data_path : Path
        base folder in raw_data
    events_base : Path
        base folder in detected_events
    start_date : str
        start date of entire time window to process (e.g. years) as a string, e.g. "2026-02-09T00:00:00"
    end_date : str
        end date in same string format as end_date

    Returns
    -------
    list[Path]
        list of daily folder, which have to be processed
    """
  
    raw_data_path = Path(raw_data_path)
    events_base = Path(events_base)
    start=datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end=datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    # find all daily files of pattern: raw_data/YYYY/monthMM/YYYY-MM-DDTHH:MM:SS_YYYY-MM-DDTHH:MM-SS/
    day_folders = [p for p in raw_data_path.glob("*/month*/*") if p.is_dir()]
    
    day_folders_filtered = []
    for folder in day_folders:
        # example of pattern: '2025-09-25T00-00-00_2025-09-25T23-59-59'
        folder_name = folder.name
        try:
            # keep only date, disregard time
            date_str = folder_name.split('T')[0]
            folder_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            
            if start <= folder_date <= end:
                day_folders_filtered.append(folder)
        except Exception:
            # folder name does not match expected pattern
            continue

    folders_to_process = []

    for folder in day_folders_filtered:

        # skip empty daily folders
        if not any(folder.iterdir()):
            continue

        folder_name = folder.name
        date_str = folder_name.split("T")[0]

        corresponding_event_folder = (
            events_base
            / folder.relative_to(raw_data_path).parents[0]
            / date_str
        )

        logging.debug(f"event folder check: {corresponding_event_folder}")

        if not corresponding_event_folder.exists():
            folders_to_process.append(folder)

    return folders_to_process

def get_day_folders(base_path: Path):
    """Find all YYYY/monthMM/day folders."""
    return sorted([p for p in base_path.glob("*/month*/*/") if p.is_dir()])

def folder_has_events(raw_folder: Path, events_base: Path) -> bool:
    """
    Prüft, ob ein entsprechender Event-Ordner für den Rohdatenordner existiert
    und ob er mindestens einen Unterordner 'eventXXX' enthält, 
    der auch mindestens eine Datei enthält.

    Args:
        raw_folder: Pfad zum Tagesordner in raw_data
        events_base: Basisordner detected_events

    Returns:
        True, wenn Events existieren und Dateien enthalten sind, False sonst
    """
    # Relativer Pfad von raw_data zu events_base
    relative_path = raw_folder.relative_to(raw_folder.parents[2])  
    corresponding_event_folder = events_base / relative_path

    if not corresponding_event_folder.exists():
        return False

    # Alle Unterordner, die mit "event" anfangen
    event_folders = [f for f in corresponding_event_folder.iterdir() if f.is_dir() and f.name.startswith("event")]

    if not event_folders:
        return False

    # Bedingung: Jeder Event-Ordner muss mindestens eine Datei enthalten
    for event_folder in event_folders:
        files = [file for file in event_folder.iterdir() if file.is_file()]
        if not files:
            return False

    return True

def find_events_to_process(events_base: Path,
                           start_date: str, end_date: str) -> list[Path]:
    """
    Returns a list of event folders that still need processing within desired time window.
    Only folders that exist AND do NOT contain 'data_red.npy' are considered unprocessed.
    

    Args:
        events_base: Base folder containing detected_events (year/month/day/eventXXX)
    Returns:
        List of Path objects for events to process
    """
    events_base = Path(events_base)
    # alle Event-Ordner finden (rekursiv)
    event_folders = [p for p in events_base.rglob("event*") if p.is_dir()]
    
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    
    event_folders_filtered = []
    for folder in event_folders:

        folder_name = folder.parent.name
        try:
            # Nur den ersten Datumsanteil nehmen (vor dem '_')
            date_str = str(folder_name)
            folder_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            
            if start <= folder_date <= end:
                event_folders_filtered.append(folder)
        except Exception:
            # Falls ein Ordnername nicht dem erwarteten Muster entspricht
            continue

    # nur Ordner zurückgeben, die noch nicht processed sind
    unprocessed = [p for p in event_folders_filtered if not (p / "data_red.npy").exists()]

    print(f"Found {len(event_folders_filtered)} event folders, {len(unprocessed)} still unprocessed.")
    return unprocessed

def find_days_to_process_for_events(event_base_path: Path, expected_events_per_day=208):
    """
    Gibt alle Tage zurück, bei denen noch mindestens ein Event fehlt.
    """
    days_to_process = []
    for day_folder in sorted(event_base_path.glob("*/month*/*")):
        if not day_folder.is_dir():
            continue
        existing_events = [p.name for p in day_folder.glob("event*") if p.is_dir()]
        if len(existing_events) < expected_events_per_day:
            days_to_process.append(day_folder)
    return days_to_process 
    
def find_raw_folder_for_event(event_folder: Path, raw_base_path: Path) -> Path:
    """
    Find the corresponding raw data folder for a given event folder.
    
    Assumes event folder is like:
        detected_events/YYYY/monthMM/DD/eventXXX
    and raw folder is like:
        raw_data/YYYY/MM/DD
    """
    event_folder = Path(event_folder)
    raw_base_path = Path(raw_base_path)
    
    # Extrahiere Jahr, Monat, Tag aus dem Event-Pfad
    year = event_folder.parts[-4]  # 'YYYY'
    month = event_folder.parts[-3]  # 'monthMM'
    day = event_folder.parts[-2]  # 'DD'
    
    day_folder_name= f"{day}T00-00-00+00-00_{day}T23-59-59+00-00"
    # Rohdaten-Pfad zusammensetzen
    month_int = int(month.replace("month", ""))
    raw_path = raw_base_path / year / f"month{str(month_int).zfill(2)}" / day_folder_name
    
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw folder found for {event_folder} for {raw_path}")
    
    return raw_path
    
    
    