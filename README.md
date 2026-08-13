# current_vis

GUI fuer Signal-Visualisierung, Lock-in-Demodulation und eine integrierte
Modulationsfrequenz-Suche mit AWG und Tektronix-MDO-Oszilloskop.

Standardformat fuer Laden und Speichern ist jetzt HDF5 (`.h5` / `.hdf5`). CSV/TXT kann weiterhin geladen werden, ist aber nur noch ein Legacy-Importpfad.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Hinweis fuer Oszilloskopzugriff:
- `pyvisa` ist in `requirements.txt` enthalten.
- `pyvisa-py` ist als softwarebasiertes VISA-Backend enthalten; NI-VISA kann
  weiterhin verwendet werden.
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

## Modulationsfrequenz suchen

Der Tab `Frequency Sweep` integriert den Workflow aus
`modulation_freq_searcher` direkt in diese Anwendung:

1. Mit `Refresh VISA` die Ressourcen laden und AWG sowie RF-Oszilloskop
   auswaehlen. Die Oszilloskop-Auswahl des normalen Datenimports wird mit dem
   Sweep-Tab synchronisiert.
2. `Frequency Sweep` auswaehlen und Start, Stop, Schrittweite, Scope-Fenster,
   RBW, Mittelungen und Messungen pro Schritt einstellen. Mit
   `Evaluation Freq. Offset` kann die Messfrequenz gegenueber der jeweiligen
   AWG-Frequenz verschoben werden: `Messfrequenz = AWG-Frequenz + Offset`.
   Der Standardwert ist `0 Hz`.
3. `Start Scan` klicken. Fuer jeden AWG-Frequenzschritt wird das RF-Fenster des
   MDO3024 gesetzt, die Amplitude aus `CURVE?` gelesen und live geplottet.
4. Der beste Messpunkt wird unter dem Plot angezeigt. Mit
   `Use Best Frequency for Demodulation` wird diese Frequenz direkt als
   Lock-in-Referenz uebernommen.
5. Ergebnisse koennen automatisch oder ueber `Save Results...` als CSV
   gespeichert werden. `File > Export Graph...` exportiert den Sweep-Plot.

Zusaetzlich stehen Amplituden- und Offset-Sweeps aus dem Ursprungsprojekt zur
Verfuegung. `Mock mode` erzeugt eine synthetische Resonanzkurve und erlaubt
einen Funktionstest ohne angeschlossene Hardware.

