Protocol, device driver and tuning/step-test algorithms imported from the user-provided TecController.zip (2026-11-19). The standalone CustomTkinter GUI is replaced by the application Qt tab.

Local driver adaptations add serial transaction tracing and validation of the
channel-specific `LIMITED` voltage ceiling (integer 0–90 percent of supply voltage).
Register units verified against the manufacturer's Communication Protocol v1.3.0,
section 3.3.1: https://img.staticdj.com/3534f43b684aba758e37f8365a41ea83.pdf

The PC-side Ziegler-Nichols search is adapted for slow, output-limited thermal
loops: configurable timing and sensitivity, extrema-based cycle detection,
period/amplitude validation, per-channel ceiling monitoring, diagnostic traces,
and explicit cleanup errors. Shared CONTMODE and optional PWMDUTY are queried
using the manufacturer's v1.3.0 protocol (sections 3.3.6 and 3.3.12), available at
https://cnres.appracle.com/3534f43b684aba758e37f8365a41ea83.pdf .
