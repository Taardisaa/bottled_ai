def is_int_string(value: str) -> bool:
    """Check whether a string parses as an integer.

    Args:
        value: String value to test.

    Returns:
        bool: True when parsing to int succeeds.
    """
    try:
        int(value)
        return True
    except ValueError:
        return False
