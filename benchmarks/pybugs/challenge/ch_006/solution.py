def fahrenheit_to_celsius(temps_dict):
    """Read temps_dict['fahrenheit'] and convert it to Celsius using
    the standard (F - 32) * 5 / 9 formula. Buggy: reads the wrong key
    ('celsius') and converts the value as if it were Fahrenheit."""
    f = temps_dict['celsius']
    return (f - 32) * 5 / 9
