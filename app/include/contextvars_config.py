import contextvars

# https://medium.com/gradiant-talks/identifying-fastapi-requests-in-logs-bac3284a6aa
request_id_contextvar = contextvars.ContextVar("request_id", default="")
request_ip_contextvar = contextvars.ContextVar("request_ip", default="")
endpoint_name_contextvar = contextvars.ContextVar("endpoint_name", default="")
