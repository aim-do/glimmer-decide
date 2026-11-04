"""python -m server --https: self-signed certificate for phones and the launcher wiring."""
import os
import ssl
from pathlib import Path

os.environ["MOCK"] = "1"

import uvicorn  # noqa: E402
from cryptography import x509  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID  # noqa: E402

from server import __main__ as launcher  # noqa: E402
from server import certs  # noqa: E402


def load(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def san_ips(cert: x509.Certificate) -> set[str]:
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}


def test_cert_covers_lan_ip_and_localhost_and_suits_phones(tmp_path):
    cert_path, key_path = certs.ensure_cert(["192.168.1.50"], tmp_path)
    c = load(cert_path)
    assert san_ips(c) == {"127.0.0.1", "192.168.1.50"}
    san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == ["localhost"]
    eku = c.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku          # iOS refuses server certs without it
    assert (c.not_valid_after_utc - c.not_valid_before_utc).days <= 398
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert_path, key_path)   # what uvicorn does


def test_cert_is_reused_until_the_address_changes(tmp_path):
    cert_path, _ = certs.ensure_cert(["192.168.1.50"], tmp_path)
    serial = load(cert_path).serial_number
    certs.ensure_cert(["192.168.1.50"], tmp_path)
    assert load(cert_path).serial_number == serial        # phone's accepted warning stays valid
    certs.ensure_cert(["10.0.0.7"], tmp_path)              # computer moved to another network
    assert "10.0.0.7" in san_ips(load(cert_path)) and load(cert_path).serial_number != serial


def test_broken_cert_file_is_replaced(tmp_path):
    (tmp_path / "cert.pem").write_text("not a certificate")
    (tmp_path / "key.pem").write_text("not a key")
    cert_path, _ = certs.ensure_cert([], tmp_path)
    assert san_ips(load(cert_path)) == {"127.0.0.1"}


def test_lan_ips_skip_loopback():
    assert all(not ip.startswith(("127.", "169.254.")) for ip in certs.lan_ips())


def test_client_resets_are_not_logged_as_errors():
    from server.app import _quiet_client_resets

    class Loop:
        seen: list = []

        def default_exception_handler(self, context):
            self.seen.append(context)

    loop = Loop()
    _quiet_client_resets(loop, {"exception": ConnectionResetError(10054, "reset by peer")})
    _quiet_client_resets(loop, {"exception": ValueError("a real bug")})
    assert [type(c["exception"]) for c in loop.seen] == [ValueError]


def run_launcher(monkeypatch, tmp_path, argv):
    calls = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.update(kw))
    monkeypatch.setattr(launcher, "ensure_env_file", lambda root=None: False)
    monkeypatch.setattr(certs, "lan_ips", lambda: ["192.168.1.50"])
    monkeypatch.setattr(certs, "CERT_DIR", tmp_path)
    launcher.main([*argv, "--mock", "--no-browser"])
    return calls


def test_https_flag_listens_on_the_network_with_tls(monkeypatch, tmp_path, capsys):
    calls = run_launcher(monkeypatch, tmp_path, ["--https"])
    assert calls["host"] == "0.0.0.0"
    assert "192.168.1.50" in san_ips(load(Path(calls["ssl_certfile"])))
    assert Path(calls["ssl_keyfile"]).exists()
    assert "https://192.168.1.50:8000" in capsys.readouterr().out


def test_default_stays_on_localhost_without_tls(monkeypatch, tmp_path):
    calls = run_launcher(monkeypatch, tmp_path, [])
    assert calls["host"] == "127.0.0.1" and "ssl_certfile" not in calls
