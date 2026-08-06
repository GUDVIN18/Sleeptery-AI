import requests
import time
import json

a = time.time()

json_user = '''{
  "user diary records": {
    "default_habits": {
      "amount": null,
      "type": "array of strings",
      "description": "Привычки пользователя из списка по умолчанию, зафиксированные в этот день."
    },
    "custom_habits": {
      "amount": null,
      "type": "array of strings",
      "description": "Пользовательские привычки, добавленные вручную (если есть)."
    },
    "sleep_rate": {
      "amount": 4,
      "type": "integer",
      "description": "Оценка сна по шкале от 1 до 5, выставленная пользователем."
    },
    "day_rate": {
      "amount": null,
      "type": "integer",
      "description": "Оценка прошедшего дня по шкале от 1 до 5 (если указана)."
    },
    "is_happy": {
      "amount": null,
      "type": "boolean",
      "description": "Флаг, указывающий, чувствовал ли себя пользователь счастливым в течение дня."
    },
    "date": {
      "amount": "2025-11-15",
      "type": "string",
      "description": "Дата записи дневника (в формате YYYY-MM-DD)."
    },
    "duration_qualitative_sleep_seconds": {
      "amount": 12146,
      "type": "integer",
      "description": "Продолжительность качественного сна за этот день (в секундах), если фиксировалась."
    },
    "day_satisfaction": {
      "amount": null,
      "type": "string",
      "description": "Субъективная оценка пользователя относительно прошедшего дня."
    }
  },
  "sleep daily stats": {
    "start_time": {
      "amount": "02:52:55",
      "type": "string",
      "description": "Время отхода ко сну."
    },
    "end_time": {
      "amount": "11:08:10",
      "type": "string",
      "description": "Время пробуждения."
    },
    "awakenings_count": {
      "amount": 0,
      "type": "integer",
      "description": "Количество пробуждений за ночь."
    },
    "light_sleep_rating": {
      "amount": "хорошо",
      "type": "string",
      "description": "Оценка легкого сна за ночь"
    },
    "deep_sleep_rating": {
      "amount": "хорошо",
      "type": "string",
      "description": "Оценка глубокого сна за ночь"
    },
    "rem_sleep_rating": {
      "amount": "так себе",
      "type": "string",
      "description": "Оценка REM-сна за ночь"
    },
    "duration_asleep_state_seconds": {
      "amount": 29687,
      "type": "integer",
      "description": "Общая продолжительность сна в секундах."
    },
    "duration_awake_state_seconds": {
      "amount": 0,
      "type": "integer",
      "description": "Время бодрствования в секундах."
    },
    "duration_light_sleep_state_seconds": {
      "amount": 17541,
      "type": "integer",
      "description": "Лёгкий сон в секундах."
    },
    "duration_qualitative_sleep_seconds": {
      "amount": 12146,
      "type": "integer",
      "description": "Качественный сон в секундах."
    },
    "duration_rem_sleep_state_seconds": {
      "amount": 4389,
      "type": "integer",
      "description": "REM-сон в секундах."
    },
    "duration_deep_sleep_state_seconds": {
      "amount": 7757,
      "type": "integer",
      "description": "Глубокий сон в секундах."
    },
    "bed_in_time": {
      "amount": 0,
      "type": "boolean",
      "description": "Лёг ли спать вовремя."
    },
    "wake_in_time": {
      "amount": 0,
      "type": "boolean",
      "description": "Проснулся вовремя."
    },
    "bedtime_at": {
      "amount": "00:10:00",
      "type": "string",
      "description": "Время отхода ко сну (локальное время, формат HH:MI:SS)."
    },
    "wakeup_at": {
      "amount": "08:10:00",
      "type": "string",
      "description": "Время пробуждения (локальное время, формат HH:MI:SS)."
    },
    "bedtime_deviation_seconds": {
      "amount": 9775,
      "type": "integer",
      "description": "Отклонение отхода ко сну от нормы."
    },
    "wakeup_deviation_seconds": {
      "amount": 10690,
      "type": "integer",
      "description": "Отклонение пробуждения от нормы."
    }
  },
  "sleep weekly stats": {
    "14.11.2025": {
      "start_time": {
        "amount": "02:14:08",
        "type": "string",
        "description": "Время отхода ко сну."
      },
      "end_time": {
        "amount": "12:00:00",
        "type": "string",
        "description": "Время пробуждения."
      },
      "awakenings_count": null,
      "light_sleep_rating": null,
      "deep_sleep_rating": null,
      "rem_sleep_rating": null,
      "duration_asleep_state_seconds": {
        "amount": 35129,
        "type": "integer",
        "description": "Общая продолжительность сна в секундах."
      },
      "duration_awake_state_seconds": {
        "amount": 0,
        "type": "integer",
        "description": "Время бодрствования в секундах."
      },
      "duration_light_sleep_state_seconds": {
        "amount": 21595,
        "type": "integer",
        "description": "Лёгкий сон в секундах."
      },
      "duration_qualitative_sleep_seconds": {
        "amount": 13534,
        "type": "integer",
        "description": "Качественный сон в секундах."
      },
      "duration_rem_sleep_state_seconds": {
        "amount": 6140,
        "type": "integer",
        "description": "REM-сон в секундах."
      },
      "duration_deep_sleep_state_seconds": {
        "amount": 7394,
        "type": "integer",
        "description": "Глубокий сон в секундах."
      },
      "bed_in_time": {
        "amount": 0,
        "type": "boolean",
        "description": "Лёг ли спать вовремя."
      },
      "wake_in_time": {
        "amount": 0,
        "type": "boolean",
        "description": "Проснулся вовремя."
      },
      "bedtime_at": {
        "amount": "00:10:00",
        "type": "string",
        "description": "Планируемое Время отхода ко сну"
      },
      "wakeup_at": {
        "amount": "08:10:00",
        "type": "string",
        "description": "Планируемое Время пробуждения."
      },
      "bedtime_deviation_seconds": {
        "amount": 7448,
        "type": "integer",
        "description": "Отклонение отхода ко сну от нормы."
      },
      "wakeup_deviation_seconds": {
        "amount": 13800,
        "type": "integer",
        "description": "Отклонение пробуждения от нормы."
      }
    },
    "15.11.2025": {
      "start_time": {
        "amount": "02:52:55",
        "type": "string",
        "description": "Время отхода ко сну."
      },
      "end_time": {
        "amount": "11:08:10",
        "type": "string",
        "description": "Время пробуждения."
      },
      "awakenings_count": null,
      "light_sleep_rating": null,
      "deep_sleep_rating": null,
      "rem_sleep_rating": null,
      "duration_asleep_state_seconds": {
        "amount": 29687,
        "type": "integer",
        "description": "Общая продолжительность сна в секундах."
      },
      "duration_awake_state_seconds": {
        "amount": 0,
        "type": "integer",
        "description": "Время бодрствования в секундах."
      },
      "duration_light_sleep_state_seconds": {
        "amount": 17541,
        "type": "integer",
        "description": "Лёгкий сон в секундах."
      },
      "duration_qualitative_sleep_seconds": {
        "amount": 12146,
        "type": "integer",
        "description": "Качественный сон в секундах."
      },
      "duration_rem_sleep_state_seconds": {
        "amount": 4389,
        "type": "integer",
        "description": "REM-сон в секундах."
      },
      "duration_deep_sleep_state_seconds": {
        "amount": 7757,
        "type": "integer",
        "description": "Глубокий сон в секундах."
      },
      "bed_in_time": {
        "amount": 0,
        "type": "boolean",
        "description": "Лёг ли спать вовремя."
      },
      "wake_in_time": {
        "amount": 0,
        "type": "boolean",
        "description": "Проснулся вовремя."
      },
      "bedtime_at": {
        "amount": "00:10:00",
        "type": "string",
        "description": "Планируемое Время отхода ко сну"
      },
      "wakeup_at": {
        "amount": "08:10:00",
        "type": "string",
        "description": "Планируемое Время пробуждения."
      },
      "bedtime_deviation_seconds": {
        "amount": 9775,
        "type": "integer",
        "description": "Отклонение отхода ко сну от нормы."
      },
      "wakeup_deviation_seconds": {
        "amount": 10690,
        "type": "integer",
        "description": "Отклонение пробуждения от нормы."
      }
    },
    "29.11.2025": {
      "start_time": {
        "amount": "00:10:39",
        "type": "string",
        "description": "Время отхода ко сну."
      },
      "end_time": {
        "amount": "07:35:18",
        "type": "string",
        "description": "Время пробуждения."
      },
      "awakenings_count": null,
      "light_sleep_rating": null,
      "deep_sleep_rating": null,
      "rem_sleep_rating": null,
      "duration_asleep_state_seconds": {
        "amount": 25590,
        "type": "integer",
        "description": "Общая продолжительность сна в секундах."
      },
      "duration_awake_state_seconds": {
        "amount": 1057,
        "type": "integer",
        "description": "Время бодрствования в секундах."
      },
      "duration_light_sleep_state_seconds": {
        "amount": 16598,
        "type": "integer",
        "description": "Лёгкий сон в секундах."
      },
      "duration_qualitative_sleep_seconds": {
        "amount": 8992,
        "type": "integer",
        "description": "Качественный сон в секундах."
      },
      "duration_rem_sleep_state_seconds": {
        "amount": 4371,
        "type": "integer",
        "description": "REM-сон в секундах."
      },
      "duration_deep_sleep_state_seconds": {
        "amount": 4621,
        "type": "integer",
        "description": "Глубокий сон в секундах."
      },
      "bed_in_time": {
        "amount": 1,
        "type": "boolean",
        "description": "Лёг ли спать вовремя."
      },
      "wake_in_time": {
        "amount": 0,
        "type": "boolean",
        "description": "Проснулся вовремя."
      },
      "bedtime_at": {
        "amount": "00:10:00",
        "type": "string",
        "description": "Планируемое Время отхода ко сну"
      },
      "wakeup_at": {
        "amount": "08:10:00",
        "type": "string",
        "description": "Планируемое Время пробуждения."
      },
      "bedtime_deviation_seconds": {
        "amount": 39,
        "type": "integer",
        "description": "Отклонение отхода ко сну от нормы."
      },
      "wakeup_deviation_seconds": {
        "amount": -2082,
        "type": "integer",
        "description": "Отклонение пробуждения от нормы."
      }
    }
  },
  "sleep monthly stats": null,
  "history sleep assessment": [
    {
      "sleep_date": {
        "amount": "2025-11-15",
        "type": "string",
        "description": "Дата сна"
      },
      "sleep_assessment": {
        "amount": "Ночь была почти без пробуждений — это суперсила! Но ты снова лег на 2,5 часа позже цели, и режим уплывает от идеала.",
        "type": "string",
        "description": "Оценка сна"
      },
      "recommendation": {
        "amount": "Твоя команда Sleeptery видит, что прошлые советы по якорению и вечернему ритуалу не сработали — режим всё ещё скачет. Давай попробуем «миссию цифрового заката»: сегодня вечером за 15 минут до цели в 00:10 выключи все гаджеты и просто посиди с закрытыми глазами. Это поможет твоему мозгу переключиться без стресса. 💤 И не забудь заполнить дневник — он наш ключ к твоим привычкам.",
        "type": "string",
        "description": "Совет по сну"
      },
      "reaction": {
        "amount": null,
        "type": "string",
        "description": "Реакция пользователя на совет. (True - нравиться, False - Не нравится, None - не отметил)"
      }
    },
    {
      "sleep_date": {
        "amount": "2025-11-14",
        "type": "string",
        "description": "Дата сна"
      },
      "sleep_assessment": {
        "amount": "Ночь была почти без пробуждений — это суперсила! Но ты снова лег на 2 часа позже цели, и режим уплывает от идеала.",
        "type": "string",
        "description": "Оценка сна"
      },
      "recommendation": {
        "amount": "Твоя команда Sleeptery видит, что прошлые советы по вечернему ритуалу не сработали — режим всё ещё скачет. Давай попробуем «миссию якорения»: сегодня вечером за 15 минут до цели в 00:10 просто ляг и представь, как твой внутренний будильник закрепляется на нужной волне. Это поможет телу войти в ритм без стресса. 💤 И не забудь заполнить дневник — он наш ключ к твоим привычкам.",
        "type": "string",
        "description": "Совет по сну"
      },
      "reaction": {
        "amount": null,
        "type": "string",
        "description": "Реакция пользователя на совет. (True - нравиться, False - Не нравится, None - не отметил)"
      }
    },
    {
      "sleep_date": {
        "amount": "2025-11-13",
        "type": "string",
        "description": "Дата сна"
      },
      "sleep_assessment": {
        "amount": "Ночь была почти без пробуждений — это суперсила! Но ты снова лег на 2 часа позже цели, и режим уплывает от идеала.",
        "type": "string",
        "description": "Оценка сна"
      },
      "recommendation": {
        "amount": "Твоя команда Sleeptery видит, что прошлые советы по якорению и цифровому закату не сработали — режим всё ещё скачет. Давай попробуем «миссию вечернего ритуала»: сегодня вечером за 15 минут до цели в 00:10 сделай что-то простое и расслабляющее, например, почитай бумажную книгу или послушай спокойную музыку. Это поможет твоему мозгу настроиться на сон без давления. 💤 И не забудь заполнить дневник — он наш ключ к твоим привычкам.",
        "type": "string",
        "description": "Совет по сну"
      },
      "reaction": {
        "amount": null,
        "type": "string",
        "description": "Реакция пользователя на совет. (True - нравиться, False - Не нравится, None - не отметил)"
      }
    }
  ]
}'''



response = requests.get(
    "https://analytics.sleeptery.xdev.team/dialog-ai/inner/history", 
    params={
        'user_id': 580,
        'sleep_date': '2025-12-28',
        'page_size': 100
    },
    headers={'Authorization': f'Bearer 2Og799dxj4mw8nW3pfnDnJt1h4R4fs1H'}
)

b = time.time()
print(b - a, response.text)