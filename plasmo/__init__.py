"""Plasmo: de una imagen a una pieza lista para imprimir en 3D, todo local."""

__version__ = "0.1.4"


def _use_system_certificates() -> None:
    """Usa los certificados del sistema para HTTPS.

    Detrás de un proxy corporativo que intercepta TLS, la lista propia de
    Python no conoce su certificado y las descargas de modelos fallan.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


_use_system_certificates()
