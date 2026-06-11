"""API Configuration – APISettings extends AppSettings with all fields unified."""

from smartloop.config.app_settings import AppSettings


class APISettings(AppSettings):
    """Backward-compatible alias; all settings live in AppSettings."""


api_settings = APISettings()
