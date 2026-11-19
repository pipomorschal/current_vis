"""ASCII protocol framing for SenseFuture TEC controllers (TEC207L).

Frame format (from the "SenseFuture Communication Protocol Specification"):

* Channel command query : ``TC1:TG=?@``      -> reply ``OKTC1:TG=2500000@\\r\\n``
* Channel command write : ``TC1:TG=2500000@`` -> reply ``OKTC1:TG=2500000@\\r\\n``
* General command query : ``FPWM=?@``         -> reply ``OKFPWM=2@\\r\\n``
* General command write : ``FPWM=2@``         -> reply ``OKFPWM=2@\\r\\n``

Channel 2 uses the ``TC2:`` prefix. The ``@`` character terminates every
frame sent to the device; replies are terminated by ``@\\r\\n``.
"""

from __future__ import annotations

TERMINATOR = "@"


class ProtocolError(Exception):
    """Raised when a frame cannot be built or a reply is invalid."""


def _check_channel(channel: int) -> int:
    if channel not in (1, 2):
        raise ProtocolError(f"channel must be 1 or 2, got {channel!r}")
    return channel


def _check_command(command: str) -> str:
    command = command.strip().upper()
    if not command or any(c in command for c in "=?@\r\n "):
        raise ProtocolError(f"invalid command name: {command!r}")
    return command


def build_query(channel: int, command: str) -> bytes:
    """Build a channel-command query frame, e.g. ``TC1:TG=?@``."""
    _check_channel(channel)
    return f"TC{channel}:{_check_command(command)}=?{TERMINATOR}".encode("ascii")


def build_set(channel: int, command: str, value: int) -> bytes:
    """Build a channel-command write frame, e.g. ``TC1:TG=2500000@``."""
    _check_channel(channel)
    return f"TC{channel}:{_check_command(command)}={int(value)}{TERMINATOR}".encode("ascii")


def build_general_query(command: str) -> bytes:
    """Build a general-parameter query frame, e.g. ``FPWM=?@``."""
    return f"{_check_command(command)}=?{TERMINATOR}".encode("ascii")


def build_general_set(command: str, value: int) -> bytes:
    """Build a general-parameter write frame, e.g. ``FPWM=2@``."""
    return f"{_check_command(command)}={int(value)}{TERMINATOR}".encode("ascii")


def expected_key(channel: int | None, command: str) -> str:
    """Key expected in the reply body, e.g. ``TC1:TG`` or ``FPWM``."""
    _check_command(command)
    if channel is None:
        return command
    _check_channel(channel)
    return f"TC{channel}:{command}"


def parse_reply(raw: bytes, expected: str | None = None) -> int | str:
    """Parse a device reply and return its value (int if numeric, else str).

    Observed on a real TEC207L (FW 5.0.7): replies look like
    ``OKTC1:TG=2500000@\\r\\n\\x00`` -- some registers use an ``OK:`` prefix
    (e.g. ``OK:FPV=5007@``), the model register returns a string (``OKTEC=207L@``)
    and every reply is followed by a trailing NUL byte. All of that is handled
    here. Raises ProtocolError on timeout (empty reply -- the device stays
    silent for unknown commands), device error replies (``ERR...``) or a key
    mismatch with ``expected``.
    """
    text = raw.decode("ascii", errors="replace").replace("\x00", "").strip()
    if not text:
        raise ProtocolError("empty reply (timeout)")
    upper = text.upper()
    if upper.startswith("ERR") or upper.startswith("NG"):
        raise ProtocolError(f"device reported an error: {text!r}")
    if not upper.startswith("OK"):
        raise ProtocolError(f"unexpected reply: {text!r}")
    body = text[2:]
    if body.startswith(":"):  # "OK:KEY=value@" variant seen on real firmware
        body = body[1:]
    body = body.strip()
    if body.endswith(TERMINATOR):
        body = body[: -len(TERMINATOR)]
    key, sep, value = body.rpartition("=")
    if not sep or not key:
        raise ProtocolError(f"malformed reply body: {body!r}")
    if expected is not None and key.strip().upper() != expected.upper():
        raise ProtocolError(f"reply key {key!r} does not match expected {expected!r}")
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return int(float(value))
    except ValueError:
        return value  # non-numeric values, e.g. model "207L"
