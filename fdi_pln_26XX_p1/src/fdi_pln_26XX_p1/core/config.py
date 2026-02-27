import os
from pathlib import Path
from dynaconf import Dynaconf

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

settings = Dynaconf(
    envvar_prefix="FDI",
    settings_files=[
        os.path.join(BASE_DIR, "settings.toml"),
        os.path.join(BASE_DIR, ".secrets.toml"),
    ],
    nested_separator="__",
)
