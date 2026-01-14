import time
import threading
from pynput.mouse import Button, Controller
from pynput.keyboard import Listener, KeyCode, Key, Controller
import pyautogui
import pydirectinput

delay = 1
start_stop_key = KeyCode(char='0')
exit_key = KeyCode(char='i')

class AutoClicker(threading.Thread):
    def __init__(self, delay):
        super(AutoClicker, self).__init__()
        self.delay = delay
        self.running = False
        self.program_running = True

    def start_clicking(self):
        self.running = True

    def stop_clicking(self):
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_running = False

    def run(self):
        while self.program_running:
            while self.running:
                pydirectinput.keyDown('e')
                time.sleep(0.15)
                pydirectinput.keyUp('e')
                time.sleep(self.delay)
                pydirectinput.keyDown('v')
                time.sleep(0.15)
                pydirectinput.keyUp('v')
                time.sleep(self.delay)
                pydirectinput.keyDown('v')
                time.sleep(0.15)
                pydirectinput.keyUp('v')
                time.sleep(self.delay)
            time.sleep(0.1)

keyboard = Controller()
click_thread = AutoClicker(delay)
click_thread.start()

def on_press(key):
    if key == start_stop_key:
        if click_thread.running:
            click_thread.stop_clicking()
        else:
            click_thread.start_clicking()
    elif key == exit_key:
        click_thread.exit()
        listener.stop()

with Listener(on_press=on_press) as listener:
    listener.join()
