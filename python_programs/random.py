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

def explain_random_password(result):
    length = result["length"]
    return (
        f"The program picked {length} random characters, one at a time, from a pool "
        f"containing every uppercase and lowercase letter, every digit (0–9), and common "
        f"symbols (like ! @ # $) — and glued them together. "
        f"Because each character is chosen independently and randomly from such a large "
        f"pool, the resulting {length}-character password is extremely hard to guess "
        f"or predict."
    )