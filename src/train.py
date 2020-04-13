import datetime
import pyshark
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

from classifier import Classifier
from datetime import timedelta


class Trainer:
    def __init__(self, window_size: int, mac: str):
        self.__window_size = window_size
        self.__mac = mac
        self.__x = []
        self.__y = []
        self.__classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)
        self.__trained = False

    def load_capture(self):
        """
        Parse a capture file and extract the features to be used to train the model
        """

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

        filter = "((wlan.da == " + mac + " || wlan.sa == " + mac + ") && wlan.fc.type_subtype == 0x0028) || wlan.fc.type_subtype == 0x0008"
        cap = pyshark.FileCapture(file, only_summaries=True, display_filter=filter)
        classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)

        packet_counter = 0
        samples_counter = 0
        start = 0

        for packet in cap:
            packet_counter += 1

            try:
                if direction == 'D' and packet.destination == mac:
                    classifier.add(packet)
                elif direction == 'U' and packet.source == mac:
                    classifier.add(packet)
            except:
                pass

            now = float(packet.time)
            classifier.update_current_time(now)

            if now - start > window_size / 3:
                samples_counter += 1
                self.__x.append(classifier.features)
                self.__y.append(traffic_class)

        cap.close()

        print("%d packets processed" % packet_counter)
        print("%d samples added" % samples_counter)

    def train(self):
        """
        Train the classifier.
        """

        accuracy = self.__classifier.train(self.__x, self.__y)
        self.__trained = True
        print("Classifier trained. Accuracy = " + str(accuracy))

    def save(self):
        """
        Save the trained model to file.
        """

        if not self.__trained:
            print("Classifier not trained yet")
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
