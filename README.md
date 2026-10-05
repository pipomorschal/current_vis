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
4. `Acquire Oscilloscope` klicken.
5. Die erfassten Daten werden direkt als aktive Datenquelle geladen und koennen wie eine Datei verarbeitet werden.
6. Optional mit `Save Last Scope Capture` als HDF5 speichern.

Fuer wiederholte Aufnahmen kann `Logging Mode` aktiviert werden. Dort werden
das Intervall zwischen den Startzeitpunkten zweier Akquisen in Sekunden, die
maximale Gesamtdauer in Minuten, HDF5 oder CSV sowie ein Ausgabeordner
gewaehlt. `Start Logging` nimmt sofort die erste Kurve auf und
speichert danach jede Akquise als eigene, zeitgestempelte Datei. Nach Ablauf
der Gesamtdauer wird keine neue Akquise mehr begonnen. `Stop Logging` beendet
die Serie sicher, nachdem eine bereits laufende Aufnahme gespeichert wurde.

Im Tab `Demodulation` koennen Referenzfrequenz und Lock-in-Tiefpass direkt in
`Hz`, `kHz`, `MHz` oder `GHz` eingegeben werden. Ein Wechsel des Praefixes
erhaelt dabei den physikalischen Frequenzwert.

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

## Serrodyne Dither

`Carrier Waveform > Serrodyne Dither` erzeugt einen reinen Serrodyne-Saegezahn
ohne Rechteckanteil und ohne additiv ueberlagerte Niederfrequenzspannung. Die
langsame Sinusschwingung moduliert ausschliesslich den Phaseninkrement bzw. die
momentane Saegezahnfrequenz:

`f_saw(t) = f_Q + Delta_f_d * sin(2*pi*f_d*t)`

Die GUI bietet `f_Q` (Standard 250 kHz), `Delta_f_d` (Standard 8 kHz), `f_d`
(Standard 7,3 Hz), ein vorzeichenbehaftetes `V_pi`, die Anzahl ganzer
Ditherperioden und optional die optische Laufzeit `tau`. Positives `V_pi`
erzeugt einen steigenden, negatives `V_pi` einen fallenden Saegezahn. Die
AFG-Ausgangsamplitude wird automatisch auf den positiven Betrag `2*abs(V_pi)`
bei 0 V DC-Offset gesetzt. Frequenzbereich, Recorddauer, ARB-Wiederholfrequenz,
effektive Sample-Rate, Punktzahl, Sampledichte und bei gesetztem `tau` auch
`beta_d = 2*pi*tau*Delta_f_d` werden angezeigt.

Der obere Vorschauplot zeigt echte Samples der ersten zehn schnellen
Saegezahnperioden; der untere Plot zeigt die momentane Frequenz ueber den
kompletten Dither-Record. `Upload / Apply` verwendet denselben direkt
verifizierten AFG1062-VISA-Upload wie `Rectangular + Ramp`.

# Recording visualization

## Optional recording inputs

In **Logging Mode**, each additional input has its own checkbox and defaults to
off. The main oscilloscope channel is the optical channel (normally CH1).

- **Record reference channel**: select a different channel (normally CH2).
  The scope is stopped while both channels are transferred from the same record,
  then its previous acquisition state is restored. Keep both channels enabled
  on the oscilloscope. If a hung driver must be killed during a paired transfer,
  check that the oscilloscope is running before continuing.
- **Read ITC4005 TEC temperature**: select or enter its USB VISA resource.
  **Refresh VISA Resources** also populates this list. `MEAS:TEMP?` reads the
  temperature and `UNIT:TEMP?` identifies its unit; results are converted to °C.
- **Read T4200 ambient temperature**: enter its COM port and select A or B.
  Serial settings are 9600 baud, RTS on and DTR off, following the supplied
  class. Set the instrument to °C. The parser expects a 9-byte response and a
  four-byte little-endian float at the configured byte offset (default 1).
  Verify this offset against your instrument: the supplied code removes one
  byte from nine, leaving eight bytes, which cannot be unpacked as one float.
  Each raw response is saved as `ambient_response_hex` for inspection.

Each recording remains a single file. HDF5 stores the optical waveform in
`time`/`amplitude`, and the reference waveform in
`reference_time`/`reference_amplitude`, with its own calibration metadata.
Paired CSV tables contain `TIME`, the optical channel, `REFERENCE_TIME`, and
the reference channel. CSV requires equal channel sample counts; use HDF5
otherwise. Existing default plots continue to use the optical waveform.
Both formats include temperature values, units, individual reading timestamps,
and error/status fields in their metadata. Sensor read errors do not discard
waveforms. Temperature readings occur sequentially after waveform transfer,
so their timestamps differ from the waveform capture timestamp.

Install `requirements.txt` for PySerial support. These options also run inside
the isolated acquisition process.

Long-running oscilloscope recording retries communication failures with fresh
VISA connections after 5, 10 and 20 seconds. After three unsuccessful retries,
recording stops and leaves the GUI available. Calibration query failures abort
the capture instead of saving data with default scaling. File-save failures stop
recording immediately. Diagnostics are written to `scope_recording.log` in the
recording directory (rotated at 2 MB with three backups). When launched through
`signal_visualization_app_main.py`, Python exception and fatal crash traces are
written to `%LOCALAPPDATA%/SignalVisualizationApp/crash.log`.
VISA acquisition runs in a separate Python process. A hard deadline of the
greater of 120 seconds or 12 times the configured VISA timeout per waveform
channel (plus a small budget for enabled temperature inputs) stops a hung
process and triggers the same bounded reconnect attempts. Stop cancels the
current acquisition. Incomplete recordings use a `.partial` suffix and are
published under the final filename only after saving completes.

**Update plots during recording** defaults to off: full-resolution filtering,
FFT and automatic demodulation otherwise run after every capture and can block
the GUI. All files retain full-resolution samples, and the latest capture remains
available through **Save Last Scope Capture**. An instrument that is itself
unresponsive may still require restarting, but its VISA worker can be stopped
without restarting the GUI.

Use the **Recordings** tab to select a directory containing GUI recordings
(`.csv`, `.h5`, or `.hdf5`), then click **Plot recordings / Refresh**.
Enable **Include nested subdirectories** to process a directory tree.
Each waveform contributes one 50 Hz **peak** amplitude, fitted using sine,
cosine and a DC offset. Capture timestamps provide actual local capture time
on the x-axis, with adaptive clock/date labels. Timestamps are read from capture
metadata or GUI filenames (`scope_YYYYMMDD_HHMMSS_mmm_CHANNEL_INDEX`);
files without timestamps use natural filename order and file index instead.
The table lists each file and its extracted value; failed files are listed below.

Optional high-pass and low-pass cutoffs are in Hz; **0 disables each filter**.
They filter each waveform before amplitude extraction, using a fourth-order
zero-phase Butterworth filter. Cutoffs near 50 Hz reduce the extracted amplitude.
Install `requirements.txt` to include SciPy, which provides the filters.
