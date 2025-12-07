import random

def get_fake_price():
    return round(random.uniform(60000, 65000), 2)

def get_fake_signal():
    return random.choice(["buy", "hold", "sell"])
