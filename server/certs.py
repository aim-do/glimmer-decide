"""Self-signed HTTPS certificate so phones on the same Wi-Fi can use the camera.

Browsers give a page the camera only over HTTPS (or on localhost). `python -m server --https`
serves the app with a certificate made here for this machine's network addresses. The phone
warns about it once; after "Proceed" the page counts as secure and the camera works.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parent.parent / ".certs"   # git-ignored
VALID_DAYS = 397          # iOS and browsers refuse TLS server certificates valid for longer
RENEW_BEFORE_DAYS = 14


def lan_ips() -> list[str]:
    """IPv4 addresses other devices may reach this machine at, the default-route one first."""
    primary = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))      # a UDP connect only picks the route; nothing is sent
            primary = s.getsockname()[0]
    except OSError:
        pass
    others: set[str] = set()
    try:
        for *_, addr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            others.add(str(addr[0]))
    except OSError:
        pass
    ordered = ([primary] if primary else []) + sorted(others - {primary})
    return [ip for ip in ordered if _routable(ip)]


def _routable(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    return not (a.is_loopback or a.is_link_local or a.is_unspecified)


def ensure_cert(ips: list[str], cert_dir: Path | None = None) -> tuple[Path, Path]:
    """(cert, key) PEM paths for `ips` + localhost. Reuses the existing pair while it covers every
    address and is not about to expire, so the phone's one-time warning stays accepted."""
    from cryptography import x509

    cert_dir = cert_dir or CERT_DIR
    cert_path, key_path = cert_dir / "cert.pem", cert_dir / "key.pem"
    wanted = {"127.0.0.1", *ips}
    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            covered = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
            fresh = cert.not_valid_after_utc - _now() > dt.timedelta(days=RENEW_BEFORE_DAYS)
            if wanted <= covered and fresh:
                return cert_path, key_path
        except (ValueError, x509.ExtensionNotFound):
            pass                                   # unreadable or foreign file: make a new pair
    _write_self_signed(sorted(wanted, key=ipaddress.ip_address), cert_path, key_path)
    return cert_path, key_path


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _write_self_signed(ips: list[str], cert_path: Path, key_path: Path) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Glimmer Decide (self-signed, local)")])
    now = _now()
    san = [x509.DNSName("localhost"), *(x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips)]
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # owner-only on POSIX
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
