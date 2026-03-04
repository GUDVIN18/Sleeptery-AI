from langchain.tools import tool
from pydantic import BaseModel, Field
from .schemas.dialog import ResponseDialogAi


class AddUserDairy(BaseModel):
    dairy: str = Field(..., description="Совет/Ритуал пользователя")


@tool(
    "Добавить ритуал/совет в дневник пользователя",
    args_schema=AddUserDairy,
)
def put_user_dairy(
    dairy: str
) -> ResponseDialogAi:
    """Добавить ритуал/совет по улучшению сна в дневник пользователя, при положительном ответе о добавлении"""
    print(f"ФУНКЦИЯ ВЫЗВАНА: {dairy=}")
    return ResponseDialogAi(answer="Ритуал успешно добавлен")


@tool(
    "Формат вывода ответа пользователю",
    args_schema=ResponseDialogAi,
)
def response_format(answer: str) -> ResponseDialogAi:
    """Формирует финальный ответ пользователю в строго заданном формате."""
    return ResponseDialogAi(answer=answer)