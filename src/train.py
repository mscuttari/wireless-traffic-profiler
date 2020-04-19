#!/usr/bin/env python3
# Trainer for the classifier

__author__ = "Michele Scuttari"
__copyright__ = "Copyright 2020 Michele Scuttari"
__license__ = "GPL"
__version__ = "1.0"

import argparse
import datetime
import json
import pyshark
import re
import sys

from classifier import Classifier
from datetime import timedelta


class Trainer:
    def __init__(self, window_size: int):
        self.__window_size = window_size
        self.__x = []
        self.__y = []
        self.__classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)
        self.__trained = False

    def menu_load_capture(self):
        file = input("Capture file: ")
        mac = input("MAC address: ")
        direction = input("Traffic direction (D = download, U = upload): ").upper()
        clazz = input("Traffic class: ")
        print()

        self.load_capture(file=file, mac=mac, clazz=clazz)

    def load_capture(self, file, mac, clazz):
        """
        Parse a capture file and extract the features to be used to train the model
        """

        if not re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac):
            print("Invalid MAC address")
            return

        print("----------------------------------------------------------")
        print("Capture file: %s" % file)
        print("MAC address: %s" % mac)
        print("Class: %s" % clazz)

        mac = mac.replace("-", ":")
        filter = "((wlan.da == " + mac + " || wlan.sa == " + mac + ") && wlan.fc.type_subtype == 0x0028) || wlan.fc.type_subtype == 0x0008"
        cap = pyshark.FileCapture(file, only_summaries=True, display_filter=filter)
        classifier = Classifier(window_size=window_size, incremental_computation_threshold=50)

        packet_counter = 0
        samples_counter = 0
        start = 0

        for packet in cap:
            packet_counter += 1

            try:
                if packet.destination == mac:
                    classifier.add(packet)
            except:
                pass

            now = float(packet.time)
            classifier.update_current_time(now)

            if now - start > window_size / 3:
                samples_counter += 1
                self.__x.append(classifier.features)
                self.__y.append(clazz)

        cap.close()

        print("\n%d packets processed" % packet_counter)
        print("%d samples added" % samples_counter)
        print("----------------------------------------------------------\n")

    def menu_train(self):
        """
        Train the classifier.
        """

        print("Training started")
        start = datetime.datetime.now()
        self.__classifier.train(self.__x, self.__y)
        self.__trained = True
        end = datetime.datetime.now()
        elapsed = (end - start) / timedelta(seconds=1)
        print("Classifier trained. Took %d seconds" % elapsed)

    def menu_save(self):
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
    # Configuration parameters
    parser = argparse.ArgumentParser(description="Train script for the wireless encrypted traffic classifier")
    parser.add_argument("--config", help="configuration file containing the window size and the captures to be loaded", metavar="<file_path>")
    args = parser.parse_args()

    window_size = None
    trainer = None

    if args.config is not None:
        with open(args.config) as config_file:
            config = json.load(config_file)

            if config['window_size']: window_size = config['window_size']
            trainer = Trainer(window_size=window_size)

            for capture in config['captures']:
                trainer.load_capture(capture['file'], capture['mac'], capture['direction'], capture['class'])

    if window_size is None:
        window_size = int(input("Window size (in seconds): "))

    if trainer is None:
        trainer = Trainer(window_size=window_size)

    # Menu loop
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
            trainer.menu_load_capture()
            print()

        elif choice == 2:
            trainer.menu_train()
            print()

        elif choice == 3:
            trainer.menu_save()
            print()

        elif choice == 4:
            sys.exit(0)

        else:
            print("Invalid option\n")
