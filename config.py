import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

DEVICES = {
    "air_purifier": {"states": ["on", "off", "auto"], "actions": ["on", "off", "set_mode"]},
    "light": {"states": ["on", "off", "colored", "dimmed"], "actions": ["on", "off", "set_value", "flash"]},
    "door": {"states": ["open", "closed", "locked"], "actions": ["lock", "unlock", "open", "close"]},
    "thermostat": {"states": ["high", "low", "idle"], "actions": ["on", "off", "set_value"]},
    "smart_plug": {"states": ["on", "off"], "actions": ["on", "off"]},
    "fan": {"states": ["on", "off", "low", "high"], "actions": ["on", "off", "set_speed"]},
    "garage_door": {"states": ["open", "closed"], "actions": ["open", "close"]},
    "window": {"states": ["open", "closed"], "actions": ["open", "close"]},
    "water_valve": {"states": ["open", "closed"], "actions": ["open", "close"]},
    "sprinkler": {"states": ["on", "idle"], "actions": ["on", "off"]},
    "dryer": {"states": ["on", "off"], "actions": ["on", "off"]},
    "washer": {"states": ["on", "off"], "actions": ["on", "off"]},
    "dishwasher": {"states": ["on", "off"], "actions": ["on", "off"]},
    "refrigerator": {"states": ["on", "off", "open"], "actions": ["on", "off"]},
    "coffee_pot": {"states": ["on", "idle"], "actions": ["on", "off"]},
    "kettle": {"states": ["on", "off"], "actions": ["on", "off"]},
    "vacuum": {"states": ["on", "docked"], "actions": ["on", "off"]},
    "television": {"states": ["on", "off"], "actions": ["on", "off"]},
    "alarm": {"states": ["armed", "disarmed", "triggered"], "actions": ["arm", "disarm", "panic"]},
    "camera": {"states": ["active", "idle"], "actions": ["on", "off"]},
    "smoke_detector": {"states": ["alert", "clear"], "actions": ["on", "off"]},
    "time_sensor": {"states": ["morning", "afternoon", "evening", "night", "sunset", "sunrise"], "actions": ["at_time"]},
    "doorbell": {"states": ["pressed", "idle"], "actions": ["press"]},
    "phone": {"states": ["notifying", "low_battery", "pressed", "idle"], "actions": ["notify", "press"]},
    "watch": {"states": ["notifying", "low_battery", "pressed", "idle"], "actions": ["notify", "press"]},
    "boiler": {"states": ["on", "off"], "actions": ["on", "off"]},
    "heater": {"states": ["on", "off", "high", "low"], "actions": ["on", "off", "set_value"]},
    "ac": {"states": ["on", "off", "high", "low"], "actions": ["on", "off", "set_value"]},
    "smart_switch": {"states": ["on", "off"], "actions": ["on", "off"]},
    "location": {"states": ["home", "away", "entering", "leaving"], "actions": ["at_location"]},
    "weather": {"states": ["sunny", "rainy", "cloudy", "snowy", "clear"], "actions": ["at_weather"]},
    "blinds": {"states": ["raised", "lowered"], "actions": ["raise", "lower"]},
    "home_assistant": {"states": ["listening", "idle"], "actions": ["voice_command"]}
}

ACTION_TO_STATE = {
    "on": "on", "off": "off", "turn_on": "on", "turn_off": "off",
    "plug_on": "on", "plug_off": "off", "start_brewing": "on", "stop_brewing": "idle",
    "start_cycle": "on", "stop_cycle": "off", "water_on": "on", "water_off": "idle",
    "fan_on": "on", "fan_off": "off", "start_cleaning": "on", "return_to_dock": "docked",
    "start_boiling": "on", "turn_on_tv": "on", "turn_off_tv": "off",
    "open": "open", "close": "closed", "open_door": "open", "close_door": "closed",
    "garage_open": "open", "garage_close": "closed", "open_window": "open", "close_window": "closed",
    "valve_open": "open", "valve_close": "closed",
    "raise": "raised", "lower": "lowered",
    "lock": "locked", "unlock": "open", "arm": "armed", "disarm": "disarmed",
    "panic_on": "triggered", "alarm_on": "alert", "alarm_off": "clear",
    "press": "pressed", "notify": "notifying",
    "heat_on": "high", "cool_on": "low", "set_temp": "high", "set_speed": "high",
    "voice_command": "listening", "at_location": "home", "at_weather": "rainy"
}

OPPOSITE_ACTION = {
    "on": "off", "off": "on",
    "open": "close", "close": "open",
    "lock": "unlock", "unlock": "lock",
    "arm": "disarm", "disarm": "arm",
    "heat_on": "cool_on", "cool_on": "heat_on",
    "raise": "lower", "lower": "raise"
}

DEVICE_ROLES = {
    "time_sensor": "sensor", "smoke_detector": "sensor", "water_valve": "sensor",
    "location": "sensor", "weather": "sensor", "doorbell": "sensor",
    "light": "actuator", "fan": "actuator", "smart_plug": "actuator", "coffee_pot": "actuator",
    "kettle": "actuator", "vacuum": "actuator", "television": "actuator", "sprinkler": "actuator",
    "smart_switch": "actuator", "blinds": "actuator", "heater": "actuator", "ac": "actuator",
    "garage_door": "security", "door": "security", "window": "security", "alarm": "security",
    "refrigerator": "appliance", "washer": "appliance", "dryer": "appliance", 
    "dishwasher": "appliance", "boiler": "appliance", "phone": "actuator",
    "home_assistant": "sensor", "watch": "actuator" 
}

ROLE_LIST = ["sensor", "actuator", "security", "appliance"]
DEVICE_LIST = sorted(list(DEVICES.keys()))
STATE_LIST = sorted({s for d in DEVICES.values() for s in d["states"]})
ACTION_LIST = sorted({a for d in DEVICES.values() for a in d["actions"]})

EDGE_LE = LabelEncoder()
EDGE_LE.fit(["chain", "direct", "none", "resource"])

def to_one_hot(val, vocab):
    if val not in vocab:
        return torch.zeros(len(vocab))
    idx = vocab.index(val)
    return F.one_hot(torch.tensor(idx), num_classes=len(vocab)).float()