import random
import string

def generate_random_password(length):
    try:
        l = int(length)
    except (ValueError, TypeError):
        raise ValueError("Length must be a valid integer.")

    if not (8 <= l <= 32):
        raise ValueError("Password length must be between 8 and 32 characters.")

    characters = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
    password = "".join(random.choices(characters, k=l))
    return password