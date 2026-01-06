import sys
import time
import machine
import neopixel

# Config
NUM_STRIPS = 3
LEDS_PER_STRIP = 8
TOTAL_LEDS = NUM_STRIPS * LEDS_PER_STRIP
PIN = 2

np = neopixel.NeoPixel(machine.Pin(PIN), TOTAL_LEDS)

def set_strip(strip, count, r, g, b):
    # strip = 1,2,3  → indexes 0-7, 8-15, 16-23
    base = (strip - 1) * LEDS_PER_STRIP

    for i in range(LEDS_PER_STRIP):
        reversed_i = (LEDS_PER_STRIP - 1) - i  # dus 7→0, 6→1, etc.
        led_index = base + reversed_i

        if i < count:   # deze led moet aan
            np[led_index] = (r, g, b)
        else:
            np[led_index] = (0, 0, 0)

    np.write()

def parse_command(line):
    parts = line.strip().split()
    if len(parts) != 6:
        print("ERR syntax")
        return

    cmd, strip, count, r, g, b = parts
    if cmd != "SET":
        print("ERR cmd")
        return

    strip = int(strip)
    count = int(count)
    r = int(r)
    g = int(g)
    b = int(b)

    if not (1 <= strip <= 3):
        print("ERR strip")
        return

    set_strip(strip, count, r, g, b)
    print("OK")

# MAIN LOOP
while True:
    line = sys.stdin.readline()
    if line:
        parse_command(line)
    time.sleep(0.01)
