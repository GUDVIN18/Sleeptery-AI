from http import HTTPStatus
from typing import Any, Type, Optional, Callable
import pydantic
from fastapi import HTTPException
from pydantic import BaseModel, Field
from fastapi import status


class BaseError(BaseModel):
    """Class describing APPHTTPException json response content"""
    detail: Any
    user_message: str
    type: Optional[str] = Field("BaseError", description="Exception class name")


class AppHTTPException(HTTPException):
    http_code: int = 400
    detail: Any = "Error occurred"
    user_message: str = "Unexpected error"
    type: str = "AppHTTPException"

    def __init__(
            self,
            detail: Any = None,
            user_message: str = None,
            status_code: Optional[int] = None,
            exc_type: Optional[str] = None
    ):
        self.user_message = user_message or self.user_message
        self.type = exc_type or self.__class__.__name__
        super().__init__(status_code=status_code or self.http_code, detail=detail or self.detail)

    custom_related_api_build: Optional[
        Callable[
            [int, dict],
            'AppHTTPException'
        ]
    ] = None
    """
        Позволяет прокинуть кастомный билд инстанса, который в дальнейшем соберется в `QueryPolicy`.
        
        Пример:
        @classmethod
        def custom_related_api_build(cls, status: int, data: dict) -> ...:
            return cls(
                detail=...,
                user_message=...,
                status_code=status
            )
    """

    @property
    def response_model(self):
        model = pydantic.create_model(
            f"{self.__class__.__name__}",
            detail=(str, self.detail),
            user_message=(str, self.user_message),
            type=(str, self.__class__.__name__)
        )

        response = {
            self.http_code: {
                "description": HTTPStatus(self.http_code).phrase,
                "model": BaseError,
                "content": {
                    "application/json": {
                        "examples": {
                            model.__name__: {
                                "summary": model.__name__,
                                "value": model().model_dump()
                            }
                        }
                    }
                }
            }
        }
        return response


class WrongSecretError(AppHTTPException):
    http_code = status.HTTP_403_FORBIDDEN
    user_message = "Wrong secret key"

class DialogAiErrorGeneration(AppHTTPException):
    http_code = status.HTTP_504_GATEWAY_TIMEOUT
    user_message = "Error Generation from DialogAi"


class DialogAiErrorFormat(AppHTTPException):
    http_code = status.HTTP_404_NOT_FOUND
    user_message = "Error format JSON from DialogAi"


class DialogAiErrorConnect(AppHTTPException):
    http_code = status.HTTP_502_BAD_GATEWAY
    user_message = "Error connect to DialogAi"