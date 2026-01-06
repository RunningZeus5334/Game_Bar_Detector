import serial
import serial.tools.list_ports
import time
import platform

# Met deze code stuur je commando's naar de Raspberry Pi Pico
# die de NeoPixel strips aanstuurt zoals gedefinieerd in Pico_Code.py

def find_pico():
    """Zoek automatisch de Pico poort op Windows en Linux"""
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Raspberry Pi Pico heeft meestal deze USB VID:PID
        if "2E8A" in port.hwid.upper():  # Raspberry Pi Pico VID
            return port.device
    
    # Fallback: eerste beschikbare poort
    if ports:
        print(f"Pico niet herkend, gebruik {ports[0].device}")
        return ports[0].device
    
    # Als er helemaal niks is
    raise Exception("Geen seriële poorten gevonden!")

# Detecteer automatisch de juiste poort
port = find_pico()
print(f"Verbinden met {port}...")

ser = serial.Serial(port, 115200)
time.sleep(2)   # De Pico herstart bij serial-connect

def send(cmd):
    ser.write((cmd + "\n").encode())
    reply = ser.readline().decode().strip()
    print("Pico:", reply)

# Voorbeeldlijstje kleurtjes en hoeveel leds aan
sequence = [
    ("SET 1 8 255 0 0"),    # strip 1, drie rode leds
    ("SET 2 4 0 0 255"),    # strip 2, vijf groene
    ("SET 3 3 0 255 0"),    # strip 3 volledig blauw
    ("SET 1 7 255 0 0"),    # strip 1 geel
    ("SET 2 3 0 0 255"),    # strip 2 paars
    ("SET 3 2 0 255 0"),    # strip 3 uit
    ("SET 1 6 255 0 0"),    # strip 1 geel
    ("SET 2 2 0 0 255"),    # strip 2 paars
    ("SET 3 1 0 255 0"),    # strip 3 uit
    ("SET 1 5 255 0 0"),    # strip 1 geel
    ("SET 2 1 0 0 255"),    # strip 2 paars
    ("SET 3 0 0 255 0"),    # strip 3 uit
]

while True:
    for cmd in sequence:
        send(cmd)
        time.sleep(0.25)  # Ritme erin, Pico krijgt ademruimte
