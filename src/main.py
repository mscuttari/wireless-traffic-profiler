import pyshark
import re
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python %s interface_name mac_address" % sys.argv[0])
        sys.exit(1)

    interface = sys.argv[1]     # Wireless interface to listen on
    mac = sys.argv[2].lower()   # MAC address of the machine whose traffic has to be profiled

    # Check if the MAC address is valid
    if not re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac):
        print("Invalid MAC address")
        sys.exit(1)

    # Wireshark filter for MAC address
    mac = mac.replace("-", ":")
    filter = "wlan.sa == " + str(mac) + " || wlan.da == " + mac

    # Start capturing
    cap = pyshark.LiveCapture(interface = interface, display_filter = filter)

    for packet in cap.sniff_continuously():
        print("Sender: " + str(packet.wlan.sa))
