import pyshark
import re
import sys
import datetime
from datetime import timedelta

from classifier import Classifier

WINDOW_SIZE_DEFAULT = 5

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python %s interface_name mac_address model [window_size]" % sys.argv[0])
        sys.exit(1)

    interface = sys.argv[1]     # Wireless interface to listen on
    mac = sys.argv[2].lower()   # MAC address of the machine whose traffic has to be profiled
    model = sys.argv[3]
    window_size = WINDOW_SIZE_DEFAULT if len(sys.argv) == 4 else int(sys.argv[4])   # Window size in seconds

    # Check if the MAC address is valid
    if not re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac):
        print("Invalid MAC address")
        sys.exit(1)

    # Capture only the data packets involving the specified MAC address + all the probe requests to track time
    mac = mac.replace("-", ":")
    filter = "((wlan.da == " + mac + " || wlan.sa == " + mac + ") && wlan.fc.type_subtype == 0x0028) || wlan.fc.type_subtype == 0x0008"

    # Create the classifier and load the pre-trained model
    classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)
    classifier.load_trained_model(model)
    print("Loaded pre-trained model " + model)

    # Start the live capture
    cap = pyshark.LiveCapture(interface=interface, only_summaries=True, display_filter=filter)

    start = datetime.datetime.now()

    for packet in cap.sniff_continuously():
        try:
            if packet.destination == mac:
                classifier.add(packet)
        except:
            pass

        now = datetime.datetime.now()

        if ((now - start) / timedelta(microseconds=1000)) / 1000 > window_size / 2:
            start = now
            classifier.print_features()
            classifier.predict()

        classifier.update_current_time(float(packet.time))



