import re

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
