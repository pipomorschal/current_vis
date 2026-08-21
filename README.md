# current_vis

GUI fuer Signal-Visualisierung, Lock-in-Demodulation und eine integrierte
Modulationsfrequenz-Suche mit AWG und Tektronix-MDO-Oszilloskop.

Standardformat fuer Laden und Speichern ist jetzt HDF5 (`.h5` / `.hdf5`). Tektronix-Waveforms (`.wfm`) koennen direkt geladen werden. CSV/TXT kann weiterhin geladen werden, ist aber nur noch ein Legacy-Importpfad.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Hinweis fuer Oszilloskopzugriff:
- `pyvisa` ist in `requirements.txt` enthalten.
- `pyvisa-py` ist als softwarebasiertes VISA-Backend enthalten; NI-VISA kann
  weiterhin verwendet werden.
- Fuer HDF5-Dateien wird `h5py` verwendet.
- Tektronix-WFM-Dateien werden mit `tm_data_types` als kalibrierte Zeit- und Spannungswerte importiert.

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
2. Sweep-Typ und AWG-Wellenform auswaehlen und Start, Stop, Schrittweite, Scope-Fenster,
   RBW, Mittelungen und Messungen pro Schritt einstellen. Mit
   `Evaluation Freq. Offset` kann die Messfrequenz gegenueber der jeweiligen
   AWG-Frequenz verschoben werden: `Messfrequenz = AWG-Frequenz + Offset`.
   Der Standardwert ist `0 Hz`.
   Fuer den AFG1062 stehen die Sweep-Traeger Sine (bis 60 MHz), Square
   (bis 30 MHz) und Ramp (bis 2 MHz) zur Auswahl.
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

## Rectangular + Ramp fuer Sagnac FOCS

Im Feld `Carrier Waveform` steht zusaetzlich `Rectangular + Ramp` zur
Verfuegung. Dieser Modus ist ein fester ARB-Ausgang und veraendert die
bestehenden Sine-, Square- und Ramp-Sweeps nicht.

1. Rechteckfrequenz (Standard 395 kHz), Rechteckamplitude (Standard 2,4 Vpp),
   Rampensteigung in mV pro Rechteckperiode und die Anzahl ganzer Perioden im
   ARB-Record (Standard 10) einstellen.
2. Die GUI zeigt den vollstaendigen Record samt absichtlich wiederkehrendem
   Rampen-Reset, ARB-Wiederholfrequenz, effektiver Sample-Rate, Sample-Anzahl,
   Gesamtamplitude und notwendigem DC-Offset an.
3. `Upload / Apply` uebertraegt die 14-Bit-Samples direkt per VISA in den
   Edit-Speicher des AFG1062. ArbExpress ist nicht erforderlich.
   Fuer die automatische binaere Upload-Pruefung wird AFG1062-Firmware 1.0.3
   oder neuer benoetigt.

Ein vom AFG nach dem Binaertransfer gemeldetes `-201, Invalid while in local`
wird nur dann als Firmware-Warnung behandelt, wenn Recordlaenge, alle 14-Bit-
Samples, ARB-Frequenz, Amplitude, DC-Offset und Ausgangsstatus anschliessend
erfolgreich vom Instrument zurueckgelesen wurden.

Die Wiederholfrequenz wird als `f_ARB = f_mod / N_Perioden` gesetzt. Die
Sample-Anzahl wird automatisch so gewaehlt, dass jede Rechteckperiode exakt
50 % Tastgrad hat und weder 1.048.576 Punkte noch 300 MS/s ueberschritten
werden.
Vor dem Einschalten von Kanal 1 setzt die Software die berechnete gesamte
AFG-Amplitude und den DC-Offset, damit der gewaehlte Rechtecksprung trotz der
Rampenbewegung konstant bleibt.

