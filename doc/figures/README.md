# Figures

Photographs of the device used by the documentation. The hardware section
references these filenames; if a file is missing, a framed placeholder box is drawn
instead (the document still compiles).

| Filename                | What it shows                                          |
|-------------------------|--------------------------------------------------------|
| `device_assembled.jpg`  | The assembled ESP32 + AD8232 unit inside its case.     |
| `electrodes.jpg`        | The snap ECG electrodes used with the AD8232.          |
| `case_worn.jpg`         | The enclosure worn on the torso.                       |

Notes:

- **Wiring** is documented with an in-document TikZ schematic
  (`fig:schematic` in `sections/02_hardware.tex`), so no PCB/wiring photo is needed.
- **Electrode placement** on the body is conveyed by the conceptual torso diagram
  (`fig:case`); no on-body placement photo is required.

To add or replace a photo, drop a JPG/PNG with the matching filename here and
rebuild. Recommended: landscape, at least 1600 px on the long edge.
