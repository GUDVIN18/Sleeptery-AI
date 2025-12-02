from typing import Optional, Callable, Coroutine
import aiohttp
import orjson
from fastapi import Depends, Request
from loguru import logger as log
from .resources.exceptions import WrongSecretError
from include.config import config


class VerifyQuery:
    def __init__(
            self,
            decode_token: bool = True,
            verify_token: bool = True,
            verify_secret: bool = False
    ):
        self._decode_token = decode_token
        self._verify_token = verify_token
        self._verify_secret = verify_secret

    async def __call__(
            self,
            request: Request,
    ):
        secret = request.headers.get('Secret', None)
        if secret != config.DOCKER_SECRET:
            if self._verify_secret:
                raise WrongSecretError
        else:
            request.state.secret = secret

        return request.state.user


# Security presets
secret_access = VerifyQuery(verify_secret=True, verify_token=False, decode_token=False)