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

Select **Optical frequency 1** (default 50 Hz). To evaluate a second component,
enable **Evaluate second optical frequency**, enter **Optical frequency 2**, and
click **Plot recordings / Refresh**. With both enabled, a joint sine/cosine fit
extracts the two peak amplitudes from each waveform. They appear as separate
curves with frequency labels and separate table columns. Optional filtering
applies before the fit. Optical/reference division uses the recorded reference's
50 Hz peak current for both optical components, and normalization divides each
displayed optical curve by its own maximum. Frequency changes require Refresh;
the second-frequency checkbox can hide a previously loaded second curve.

The **Recordings** sidebar has independent plot toggles for optical amplitude,
and a **Divide optical current by reference** checkbox. This divides each
optical 50 Hz peak current by the corresponding reference peak current before
optional normalization. Set **Optical ratio scale (A/V)** and **Reference scale
(A/V)** to the channel calibrations; both default to 1. The ratio is unitless.
Missing or zero references produce gaps and blank table values. Unchecking
restores optical amplitudes.

The overlay toggles also include
ambient temperature, TEC temperature and reference current. All enabled series
share the capture-time graph. Temperatures use a separate °C axis and their
individual reading timestamps when available. Reference current has its own
axis and is computed from the reference waveform's 50 Hz peak voltage using
**Reference scale (A/V)**; enter your current probe's calibration (default 1 A/V).
The optional waveform filters apply to both optical and reference amplitude
extraction. Normalization divides optical and reference values by their own
respective maxima; temperatures remain in °C. Missing readings appear as gaps
and unavailable series are reported in the sidebar. The table includes all
available values. Toggle and scale changes reuse loaded data.

## TEC temperature sine (USB only)

The **TEC Sine** tab continuously sends
`T(t) = center + amplitude * sin(2*pi*frequency*t)` to the ITC4005 over USB.
Set the center temperature, amplitude (**peak**, in °C), and frequency in Hz,
mHz or µHz. The preview shows one period and the last verified setpoint.
Defaults are 25 °C center, ±1 °C amplitude, 1 mHz (1000 s period), and a 5 s
setpoint update interval. The sine starts at the center and rises.

Select or enter the USB VISA resource (or **Use Time tab TEC**) and press
**Start sine**. No oscilloscope acquisition or recording is performed.
TEC readback only verifies accepted setpoints. The TEC must already
be enabled in temperature-control mode, and the complete center ± amplitude
range is checked against its configured limits before the first write.
Choose an update interval giving at least 20 setpoints per period. Writes run
sequentially in an isolated process; if USB is slow, missed updates are skipped
and the sine phase follows elapsed time. The actual temperature follows the
controller's thermal dynamics.

It runs until **Stop**, a communication/verification error, or application
shutdown. Stop retains the last commanded setpoint. Hardware access is shared
with PID, scope logging and frequency sweeps. The tab does not save data files.

### Keithley 2000 readings via RS232

The same tab can display live measurements from a **Keithley 2000** connected
through a USB-to-RS232 adapter. In the sidebar's Keithley section, use **Find
serial ports** or enter the COM port, and match the baud rate and flow control
to the meter's RS232 menu. Defaults are **4800 baud, 8 data bits, no parity,
one stop bit, no flow control**. XON/XOFF is also supported. Enable RS232 and
SCPI on the meter; hardware RTS/CTS handshaking is disabled. Protocol details
are in the [Keithley 2000 manual](https://download.tek.com/manual/2000-900_J-Aug2010_User.pdf).

Select DC voltage (default), AC RMS voltage, DC current, AC RMS current,
2/4-wire resistance or frequency, with the corresponding input connections.
**Start meter** works independently of the temperature sine. Alternatively,
check **Start/stop meter with sine** to start reading alongside the sine and
stop that reading session when the sine stops. A meter session already running
independently remains independent. Default read interval is 1 s and response
timeout is 5 s; increase the timeout for slow measurement settings.

The app verifies `*IDN?`, uses `CONF` to select the function with its default
settings, and requests a fresh single measurement with `READ?`. CR, LF and
combined response terminators are accepted. The numeric value, units and read
completion timestamp appear above a trend plot; only the last 5000 points are
kept in memory. No recording file is created. AC readings are meter RMS values,
not separate 50/100 Hz amplitudes. Meter I/O runs separately from TEC updates;
slow reads skip missed read slots. A timeout, invalid/overrange value or serial
error stops the meter and shows the reason without stopping the sine. Stop
cancels pending reads, returns local front-panel control and closes the port.

If identification fails, use **Copy serial details** after the meter stops to
copy the COM port, baud/flow settings and recent transmitted/received bytes.
The status text itself can also be selected and copied. Startup ignores echoed
commands and stale lines while waiting for a valid `*IDN?` response and retries
identification once. Quoted identity fields are supported; a different Keithley
model is still rejected. Garbled/non-ASCII data points to checking baud rate,
8N1 framing, COM selection and the RS232 cable before changing the identity check.

## TEC PID working-point optimization

The **TEC PID** tab, immediately before the mini-game, adjusts the ITC4005 TEC
temperature to improve the transducer's measured 50 Hz current signal while
suppressing its 100 Hz harmonic. It uses a bounded extremum-seeking loop:

`J = (A50 - weight * A100) / initial_A50`

Both amplitudes are **peak voltages at the scope input**, jointly fitted at
50 and 100 Hz with a DC offset. Configure the current probe/transducer scale
on the scope consistently. Weight defaults to 1; increase it to penalize 100 Hz
more strongly. The objectives can compete, so the loop seeks a **local weighted
optimum**, rather than guaranteeing both independent extrema or a global optimum.
An ordinary amplitude-setpoint PID cannot determine the direction toward an
unknown maximum. Instead, each cycle measures center-minus-probe and
center-plus-probe, estimates the score gradient, applies a PID correction to
the center, then commands and measures that new center. The ITC4005 continues
to run its own internal temperature-control loop.

1. Start with **Simulation** (default, no hardware; thermal time runs 60x faster).
2. For hardware, choose separate scope and ITC4005 VISA resources, or use
   **Use Time tab inputs**. Configure the scope timebase, input scale and trigger
   source/level first. The transferred waveform must span at least 0.2 s with
   a sample rate above 200 Hz. The harmonic fit uses up to the first two seconds,
   with anti-alias filtering before reducing large records.
3. Set suitable TEC bounds and starting temperature for your transducer.
   The example defaults are 20–30 °C, starting at 25 °C; they are **not measured
   device limits**. The center leaves room for symmetric ±0.2 °C probes.
   The default maximum center move is 0.1 °C/cycle (probe transitions are separate).
4. The initial gains are Kp=0.1 °C², Ki=0 °C²/s and Kd=0 °C²·s, acting on
   the normalized gradient in 1/°C. They are conservative guesses. Integral
   accumulation is conditional on the output and move limits; a gradient sign
   change or deadband resets it. The time step uses actual elapsed cycle time.
5. Set **Settle / step observation** (default 300 s) and **Minimum read interval**
   (default 60 s). Each verified setpoint write is followed by the settling wait,
   then a newly armed normal-triggered single acquisition. Missing triggers time
   out; stale records and forced auto triggers are not used. Scope trigger source,
   level and timebase remain those configured on the instrument. Acquisition/run
   and trigger mode are restored after each completed read. A default cycle takes
   at least 15 minutes plus I/O and analysis. Acquisition never overlaps another.
6. Optionally run **Measure calibration** first. It measures the settled baseline,
   samples a positive temperature step over the observation period, measures the
   negative probe and returns to the start. Use at least four sampling intervals
   and preferably enough observation time to see the response plateau. A resolved
   63% response crossing suggests a dwell of at least five response times;
   local concave score curvature suggests an initial Kp. These estimates include
   communication/acquisition delay. Flat, drifting or unresolved responses are
   reported rather than producing a timing estimate. **Use calibration estimates**
   explicitly copies the proposed gains and dwell; simulation estimates cannot
   be applied in hardware mode. Retune after changing the harmonic penalty.
7. **Start optimization** runs the configured number of cycles (default 100).
   Live plots show both harmonics, commanded/measured TEC temperature and score.
   Every measurement is flushed to a unique CSV in the selected log folder;
   adjacent JSON files preserve settings and calibration estimates. Raw waveforms
   are analyzed but are not saved by this tab.

Hardware access is shared with scope logging and frequency sweeps; a real PID
run locks those operations until its worker has finished. TEC output must already
be enabled in temperature mode, and every command respects controller limits.
Bad signal, failed readback, a measured temperature outside the chosen bounds,
or a settled temperature outside the configured tolerance stops the run.
**Stop** cancels waits/acquisition and retains the last commanded setpoint,
including a probe setpoint if stopped mid-cycle. VISA runs in a disposable
process; cancelling during a scope transaction may leave the scope stopped or
armed in single-sequence mode. A completed optimization ends at its final center.

## Optional recording inputs

**TEC temperature sweep** in Logging Mode optionally sets an ascending ITC4005
temperature sequence using **Minimum**, **Maximum**, **Increment** (°C), and
**Time per step** (minutes). For example, 20–30 °C with a 1 °C increment and
60 minutes per step records 11 setpoints over 660 minutes, including a full
60-minute dwell at 30 °C. Sweep duration replaces **Maximum Time**; the ordinary
recording interval still controls captures, with an extra capture at each step
boundary when needed. The maximum is included even if the increment does not
divide the range exactly. **Repeat count (N)** defaults to **1** and runs the
complete range N times. Each repetition includes the full dwell at maximum
before restarting at minimum. The displayed total duration and recording stop
time include all repetitions; saved files also identify the repetition number.

Setpoints are applied between captures in the isolated VISA process using
`SOUR2:TEMP`, with readback verification before waveform acquisition. A long
capture or retry can delay a transition; the next target follows elapsed session
time. The TEC must already be enabled in temperature-control mode. Controller
temperature limits are respected. Failed setpoint application stops logging.
The final commanded temperature is retained after completion or Stop. This is
a timed setpoint sweep, not a wait-for-stabilization mode. Every saved file
includes the commanded target, step number and schedule, alongside measured
TEC temperature. All setpoint writes are disabled when the sweep is unchecked.

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
  class. Set the instrument to °C. The observed response contains five bytes:
  one prefix byte (`06`) and a four-byte little-endian float. The float offset
  defaults to 1. The reader waits only for the prefix plus the four float bytes,
  with bounded reads that also handle fragmented responses.
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
