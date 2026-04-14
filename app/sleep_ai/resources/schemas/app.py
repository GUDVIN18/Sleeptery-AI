from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List
from enum import Enum


class AppVersion(str, Enum):
    DEV: str = "dev"
    PROD: str = "prod"