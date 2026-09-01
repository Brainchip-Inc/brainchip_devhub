"""Unit tests for brainchip_utils.hardware_utils (no hardware required)."""
import akida

from brainchip_utils.hardware_utils import get_akida_device


class _FakeDevice:
    def __init__(self, ip_version):
        self.ip_version = ip_version


def test_matching_device_after_the_first_is_found(monkeypatch):
    """A device matching target_version must be found wherever it sits in the list."""
    first, second = _FakeDevice(akida.IpVersion.v2), _FakeDevice(akida.IpVersion.v1)
    monkeypatch.setattr(akida, "devices", lambda: [first, second])
    assert get_akida_device(target_version=akida.IpVersion.v1) is second


def test_returns_none_when_no_device_matches(monkeypatch):
    monkeypatch.setattr(akida, "devices", lambda: [_FakeDevice(akida.IpVersion.v2)])
    assert get_akida_device(target_version=akida.IpVersion.v1) is None


def test_returns_first_device_when_no_target_requested(monkeypatch):
    first = _FakeDevice(akida.IpVersion.v1)
    monkeypatch.setattr(akida, "devices", lambda: [first, _FakeDevice(akida.IpVersion.v2)])
    assert get_akida_device() is first
