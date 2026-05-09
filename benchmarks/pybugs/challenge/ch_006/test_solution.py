from solution import fahrenheit_to_celsius


def test_room():
    assert abs(fahrenheit_to_celsius({"celsius": 25, "fahrenheit": 77}) - 25.0) < 1e-6


def test_boiling():
    assert abs(fahrenheit_to_celsius({"celsius": 100, "fahrenheit": 212}) - 100.0) < 1e-6


def test_freezing():
    assert abs(fahrenheit_to_celsius({"celsius": 0, "fahrenheit": 32}) - 0.0) < 1e-6
