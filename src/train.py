import pyshark
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

from classifier import Classifier

window_size = 0
mac_address = None

x = []
y = []
classifier = None

def convert():
    print("Capture file: ", end='')
    file = input()

    if mac_address is None:
        print("MAC address: ", end='')
        mac = input()
    else:
        print("MAC address (leave empty to use %s): " % mac_address, end='')
        mac = input()

        if not mac:
            mac = mac_address

    if not re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac):
        print("Invalid MAC address")
        return

    mac = mac.replace("-", ":")

    print("Traffic direction (D = download, U = upload): ", end='')
    direction = input().upper()

    if direction != 'D' and direction != 'U':
        print("Invalid traffic direction")
        return

    print("Traffic class: ", end='')
    traffic_class = input()

    mac = mac.replace("-", ":")
    filter = "wlan.da == " + mac + " || wlan.sa == " + mac + " || wlan.fc.type_subtype == 0x0008"

    cap = pyshark.FileCapture(file, only_summaries=True, display_filter=filter)
    local_classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)

    counter = 0

    f = open(str("csv/" + traffic_class + ".csv"), 'w')

    for packet in cap:
        counter += 1

        try:
            if direction == 'D' and packet.destination == mac:
                local_classifier.add(packet)
        except:
            pass

        local_classifier.update_current_time(float(packet.time))

        features = local_classifier.features

        # Write to csv
        values = features.copy()
        values.append(traffic_class)
        writer = csv.writer(f)
        writer.writerow(values)

        x.append(features)
        y.append(traffic_class)

    cap.close()
    print("%d packets processed" % counter)

    f.close()


def train():
    if len(y) <= 1:
        print("At least two classes required")
        return

    accuracy = classifier.train(x, y)
    print("Model trained. Accuracy = " + str(accuracy))


def save():
    print("Path: ", end='')
    path = input()
    classifier.save_trained_model(path)


def menu():
    while True:
        print("Select an option:")
        print(" 1. Load capture file")
        print(" 2. Train")
        print(" 3. Save trained model")
        print(" 4. Exit")
        print()

        choice = int(input())
        print()

        if choice == 1:
            convert()
            print()

        elif choice == 2:
            train()
            print()

        elif choice == 3:
            save()
            print()

        elif choice == 4:
            sys.exit(0)

        else:
            print("Invalid option\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python %s window_size [mac_address]" % sys.argv[0])
        sys.exit(1)

    window_size = int(sys.argv[1])
    mac_address = sys.argv[2] if len(sys.argv) == 3 else None
    classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)

    menu()
