import argparse
import os
import sys
import serial.tools.list_ports

# PJRC / Teensy USB vendor ID (see https://www.pjrc.com/teensy/usb_ids.html)
PJRC_VID = 0x16C0

# https://www.pjrc.com/teensy/usb_ids.html (hex keys, lowercase)
PJRC_PID_MEANING = {
    "0478": "Halfkay bootloader — no serial device; upload sketch or exit bootloader",
    "0483": "USB Serial (Teensyduino) — should appear as /dev/ttyACM*",
    "0485": "USB Serial (alternate PID)",
    "0487": "USB Serial (RAW I/O)",
    "0489": "USB Dual Serial",
    "0490": "USB Triple Serial",
}


def _linux_pjrc_pid_counts() -> dict[str, int]:
    """Count PJRC (16c0) USB devices by PID from sysfs (Linux only)."""
    base = "/sys/bus/usb/devices"
    if sys.platform != "linux" or not os.path.isdir(base):
        return {}
    counts: dict[str, int] = {}
    for name in os.listdir(base):
        d = os.path.join(base, name)
        vpath = os.path.join(d, "idVendor")
        ppath = os.path.join(d, "idProduct")
        if not (os.path.isfile(vpath) and os.path.isfile(ppath)):
            continue
        try:
            with open(vpath, encoding="ascii", errors="replace") as f:
                vid = f.read().strip().lower()
            with open(ppath, encoding="ascii", errors="replace") as f:
                pid = f.read().strip().lower()
        except OSError:
            continue
        if vid != "16c0":
            continue
        counts[pid] = counts.get(pid, 0) + 1
    return counts


def _is_teensy(port) -> bool:
    desc = port.description or ""
    mfr = port.manufacturer or ""
    # Linux often reports "USB Serial" / "Teensyduino" instead of "Teensy" / "PJRC"
    if "Teensy" in desc or "PJRC" in mfr or "Teensyduino" in mfr:
        return True
    if port.vid == PJRC_VID:
        return True
    hwid = (port.hwid or "").upper()
    if "16C0" in hwid:
        return True
    return False


def discover_teensys(*, list_all: bool = False) -> None:
    print("--- Teensy Discovery Tool ---")
    ports = list(serial.tools.list_ports.comports())
    found = False

    for port in ports:
        is_t = _is_teensy(port)
        if not list_all and not is_t:
            continue
        if is_t:
            found = True
        print(f"Device: {port.device}")
        if list_all and not is_t:
            print("  (serial port, not classified as Teensy)")
        print(f"  Description: {port.description!r}")
        print(f"  Manufacturer: {port.manufacturer!r}")
        if port.vid is not None and port.pid is not None:
            print(f"  VID:PID: {port.vid:04x}:{port.pid:04x}")
        print(f"  HWID: {port.hwid!r}")
        print(f"  Serial Number: {port.serial_number!r}")
        print("-" * 30)

    if not found:
        print("No Teensy serial ports found. Check USB cable and power.")
        pjrc = _linux_pjrc_pid_counts()
        if pjrc:
            total = sum(pjrc.values())
            print(
                f"USB scan: {total} PJRC device(s) (VID 16c0) plugged in, "
                "but none expose a serial port yet:"
            )
            for pid in sorted(pjrc.keys()):
                n = pjrc[pid]
                hint = PJRC_PID_MEANING.get(pid, "see PJRC USB ID list")
                print(f"  {n}× PID {pid} — {hint}")
            if pjrc.get("0478") and not any(
                p in pjrc for p in ("0483", "0485", "0487", "0489", "0490")
            ):
                print(
                    "All visible boards are in the bootloader. Program them "
                    "(Arduino/Teensyduino or teensy_loader_cli); after your "
                    "sketch runs with USB Serial, re-run this tool."
                )
        elif not list_all and ports:
            print("Tip: run with --all to list every serial port.")
        else:
            print("No PJRC USB devices seen under /sys/bus/usb/devices either.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Teensy serial devices")
    parser.add_argument(
        "--all",
        action="store_true",
        help="List all serial ports with metadata (debugging)",
    )
    args = parser.parse_args()
    discover_teensys(list_all=args.all)
