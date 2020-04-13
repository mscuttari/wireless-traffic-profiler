import pyshark
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

from classifier import Classifier


class Trainer:
    def __init__(self, window_size: int, mac: str):
        self.__window_size = window_size
        self.__mac = mac
        self.__x = []
        self.__y = []
        self.__classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)
        self.__trained = False

    def load_capture(self):
        print("Capture file: ", end='')
        file = input()

        if self.__mac is None:
            print("MAC address: ", end='')
            mac = input()
        else:
            print("MAC address (leave empty to use %s): " % self.__mac, end='')
            mac = input()
            if not mac: mac = self.__mac

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

        filter = "wlan.da == " + mac + " || wlan.sa == " + mac + " || wlan.fc.type_subtype == 0x0008"
        cap = pyshark.FileCapture(file, only_summaries=True, display_filter=filter)
        classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)

        counter = 0

        for packet in cap:
            counter += 1

            try:
                if direction == 'D' and packet.destination == mac:
                    classifier.add(packet)
                elif direction == 'U' and packet.source == mac:
                    classifier.add(packet)
            except:
                pass

            classifier.update_current_time(float(packet.time))
            features = classifier.features

            self.__x.append(features)
            self.__y.append(traffic_class)

        cap.close()
        print("%d packets processed" % counter)

    def train(self):
        if len(self.__y) <= 1:
            print("At least two classes required")
            return

        accuracy = self.__classifier.train(self.__x, self.__y)
        self.__trained = True
        print("Model trained. Accuracy = " + str(accuracy))

    def save(self):
        if not self.__trained:
            print("Model not trained yet")
            return

        print("Path: ", end='')
        path = input()
        self.__classifier.save_trained_model(path)
        print("Model saved")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python %s window_size [mac_address]" % sys.argv[0])
        sys.exit(1)

    window_size = int(sys.argv[1])
    mac = sys.argv[2] if len(sys.argv) == 3 else None

    trainer = Trainer(window_size=window_size, mac=mac)

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
            trainer.load_capture()
            print()

        elif choice == 2:
            trainer.train()
            print()

        elif choice == 3:
            trainer.save()
            print()

        elif choice == 4:
            sys.exit(0)

        else:
            print("Invalid option\n")
