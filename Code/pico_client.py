import serial
import serial.tools.list_ports
import time


def find_pico(port_hint=None):
    ports = serial.tools.list_ports.comports()
    if port_hint:
        return port_hint
    for port in ports:
        # common VID for Raspberry Pi Pico (may vary)
        hwid = (port.hwid or '').upper()
        if '2E8A' in hwid or 'C03E' in hwid:
            return port.device
    if ports:
        return ports[0].device
    raise Exception('No serial ports found')


class PicoClient:
    def __init__(self, port=None, baud=115200, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.open()

    def open(self):
        try:
            p = find_pico(self.port)
            self.ser = serial.Serial(p, self.baud, timeout=self.timeout)
            # small delay for device ready
            time.sleep(2)
        except Exception as e:
            raise RuntimeError(f'Could not open serial port: {e}')

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def send_set(self, strip_index, led_count, r, g, b):
        # Format: SET <strip> <count> <r> <g> <b>\n
        if not self.ser or not self.ser.is_open:
            raise RuntimeError('Serial port not open')
        cmd = f"SET {strip_index} {led_count} {int(r)} {int(g)} {int(b)}\n"
        try:
            self.ser.write(cmd.encode())
            # read one line reply if available (non-blocking due to timeout)
            try:
                reply = self.ser.readline().decode().strip()
            except Exception:
                reply = ''
            return reply
        except Exception as e:
            raise RuntimeError(f'Failed to send serial command: {e}')
