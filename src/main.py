import argparse
import datetime
import pyshark
import re
import sys

from classifier import Classifier
from datetime import timedelta

if __name__ == "__main__":
    # Configuration parameters
    parser = argparse.ArgumentParser(description="Wireless encrypted traffic classifier")
    parser.add_argument("interface", help="name of the wireless interface to listen on", metavar="<interface_name>")
    parser.add_argument("mac", help="MAC address of the device whose traffic has to be profiled", metavar="<mac_address>")
    parser.add_argument("model", help="pre-trained model file", metavar="<model_path>")
    parser.add_argument("--debug", help="enable debug mode", action="store_true")
    args = parser.parse_args()

    # Check if the MAC address is valid
    args.mac = args.mac.lower()

    if not re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", args.mac):
        print("Invalid MAC address")
        sys.exit(1)

    # Capture only the data packets involving the specified MAC address + all the probe requests to better track time
    args.mac = args.mac.replace("-", ":")
    filter = "((wlan.da == " + args.mac + " || wlan.sa == " + args.mac + ") && wlan.fc.type_subtype == 0x0028) || wlan.fc.type_subtype == 0x0008"

    # Create the classifier and load the pre-trained model
    classifier = Classifier(incremental_computation_threshold=50, debug=args.debug)
    classifier.load_trained_model(args.model)

    # Print configuration parameters
    print("Interface: %s" % args.interface)
    print("Profiled MAC address: %s" % args.mac)
    print("Pre-trained model: %s" % args.model, end="\n\n")

    # Start the live capture
    cap = pyshark.LiveCapture(interface=args.interface, only_summaries=True, display_filter=filter)
    start = datetime.datetime.now()

    try:
        for packet in cap.sniff_continuously():
            if packet.destination == args.mac:
                classifier.add(packet)

            now = datetime.datetime.now()
            classifier.update_current_time(float(packet.time))

            if ((now - start) / timedelta(seconds=1)) > 1:
                start = now
                classifier.predict()

    except KeyboardInterrupt:
        print("\nStopped")
