from __future__ import annotations


class LomasError(Exception):
    """Base for everything this system raises deliberately."""


class ConfigError(LomasError):
    pass


class RegistryError(LomasError):
    pass


class ContractError(LomasError):
    """An event payload or handler did not match its declared contract."""
