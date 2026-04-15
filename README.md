# current_vis

GUI fuer Signal-Visualisierung und Demodulation (FFT, STFT, Lock-in) mit optionalem Oszilloskop-Import via VISA.

Standardformat fuer Laden und Speichern ist jetzt HDF5 (`.h5` / `.hdf5`). CSV/TXT kann weiterhin geladen werden, ist aber nur noch ein Legacy-Importpfad.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Hinweis fuer Oszilloskopzugriff:
- `pyvisa` ist in `requirements.txt` enthalten.
- Es wird zusaetzlich ein VISA-Backend benoetigt (z. B. NI-VISA oder `pyvisa-py`).
- Fuer HDF5-Dateien wird `h5py` verwendet.

## Start

```powershell
python signal_visualization_app_main.py
```

## Oszilloskopdaten in der GUI

Im Bereich `Data` gibt es den Abschnitt `Oscilloscope Input`:

1. `Refresh VISA Resources` klicken.
2. Gewuenschte Ressource auswaehlen.
3. Kanal, Punkte und Timeout setzen.
4. `Acquire from Oscilloscope` klicken.
5. Die erfassten Daten werden direkt als aktive Datenquelle geladen und koennen wie eine Datei verarbeitet werden.
6. Optional mit `Save Last Scope Capture` als HDF5 speichern.

