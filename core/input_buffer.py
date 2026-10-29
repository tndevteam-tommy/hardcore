"""Timestamped input buffering queue for combos and responsive parkour."""
import time
import config


class Action:
    JUMP = "JUMP"
    GRAB = "GRAB"
    FORWARD = "FORWARD"
    BACK = "BACK"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    SPRINT = "SPRINT"


class InputBuffer:
    def __init__(self, buffer_time=config.INPUT_BUFFER_TIME):
        self.buffer_time = buffer_time
        self.queue = []  # list of [action, timestamp, consumed]
        self.held_keys = set()

    def set_key_down(self, action):
        now = time.time()
        self.held_keys.add(action)
        self.queue.append([action, now, False])

    def set_key_up(self, action):
        self.held_keys.discard(action)

    def is_down(self, action):
        return action in self.held_keys

    def update(self):
        now = time.time()
        self.queue = [item for item in self.queue if (now - item[1]) <= self.buffer_time and not item[2]]

    def has_action(self, action, consume=True):
        now = time.time()
        for item in reversed(self.queue):
            if item[0] == action and not item[2] and (now - item[1]) <= self.buffer_time:
                if consume:
                    item[2] = True
                return True
        return False

    def has_combo(self, actions, consume=True):
        """
        Checks if a multi-key combo is active.
        A combo is considered active if all keys in the combo are either currently held
        or present in the recent input buffer.
        """
        now = time.time()
        matched_items = []
        for act in actions:
            found = False
            # Check currently held
            if act in self.held_keys:
                found = True
            else:
                # Check recent buffered tap
                for item in reversed(self.queue):
                    if item[0] == act and not item[2] and (now - item[1]) <= self.buffer_time:
                        matched_items.append(item)
                        found = True
                        break
            if not found:
                return False

        if consume:
            for item in matched_items:
                item[2] = True
        return True

    def clear(self):
        self.queue.clear()
