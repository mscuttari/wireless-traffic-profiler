import pyshark
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python %s interface_name" % sys.argv[0])
        sys.exit(1)

    cap = pyshark.LiveCapture(interface = sys.argv[1])

    packets = []

    # Main loop
    for packet in cap.sniff_continuously():
        packets.append(packet)
