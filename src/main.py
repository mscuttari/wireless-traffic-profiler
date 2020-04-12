import pyshark
import re
import sys
from classifier import Classifier

WINDOW_SIZE_DEFAULT = 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python %s interface_name mac_address [window_size]" % sys.argv[0])
        sys.exit(1)

    interface = sys.argv[1]     # Wireless interface to listen on
    mac = sys.argv[2].lower()   # MAC address of the machine whose traffic has to be profiled
    window_size = WINDOW_SIZE_DEFAULT if len(sys.argv) == 3 else sys.argv[3]    # Window size in seconds

    # Check if the MAC address is valid
    if not re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac):
        print("Invalid MAC address")
        sys.exit(1)

    # Capture filter (capture only packets involving the specified MAC address + all the probe requests to track time)
    mac = mac.replace("-", ":")
    filter = "wlan.da == " + mac + " || wlan.sa == " + mac + " || wlan.fc.type_subtype == 0x0008"

    cap = pyshark.LiveCapture(interface=interface, only_summaries=True, display_filter=filter)
    #cap = pyshark.FileCapture("/home/mscuttari/Scrivania/wireless_internet_project/src/captures/youtube.pcap", only_summaries=True, display_filter=filter)

    classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)

    for packet in cap.sniff_continuously():
    #for packet in cap:
        try:
            if packet.destination == mac:
                classifier.add(packet)
        except:
            pass

        classifier.update_current_time(float(packet.time))
        print(classifier.get_features())

