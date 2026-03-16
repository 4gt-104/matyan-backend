from __future__ import annotations

from matyan_backend.config import SETTINGS


def kafka_security_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if SETTINGS.kafka_security_protocol:
        kwargs["security_protocol"] = SETTINGS.kafka_security_protocol
    if SETTINGS.kafka_sasl_mechanism:
        kwargs["sasl_mechanism"] = SETTINGS.kafka_sasl_mechanism
    if SETTINGS.kafka_sasl_username and SETTINGS.kafka_sasl_password:
        kwargs["sasl_plain_username"] = SETTINGS.kafka_sasl_username
        kwargs["sasl_plain_password"] = SETTINGS.kafka_sasl_password
    return kwargs
