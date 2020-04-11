import datetime
import statistics
from datetime import timedelta
from scipy.stats import kurtosis


class Classifier:

    __packets = []
    __mean = 0
    __variance = 0
    __kurtosis = 0
    __rate = 0

    def __init__(self, window_size: int):
        self.__window_size = window_size

    def add(self, packet):
        # Save only QoS packet
        if int(packet.wlan.fc_type) == 2:
            self.__packets.append(packet)

    def update_current_time(self, current_time: datetime):
        """
        Inform the classifier that the time is passing. The old packets exceeding the time window are removed.
        The removal is done is such a way that the resulting window exceed by just one packet the desired window time.
        It's done in this way in order to avoid an over-cut of the window.
        """

        while len(self.__packets) > 2 and \
                (current_time - self.__packets[1].sniff_time) / timedelta(seconds=1) > self.__window_size:
            self.__packets.pop(0)

    def update_features(self):
        """
        Update the features to be used by the SVM to classify the packet sequence.
        The features are all set to zero if there are less than three packets
        """
        inter_arrival_times = []

        for i in range(1, len(self.__packets)):
            time = (self.__packets[i].sniff_time - self.__packets[i - 1].sniff_time) / timedelta(microseconds=1) / 1000
            inter_arrival_times.append(time)

        if len(inter_arrival_times) <= 1:
            self.__mean = 0
            self.__variance = 0
            self.__kurtosis = 0
            self.__rate = 0
        else:
            self.__mean = statistics.mean(inter_arrival_times)
            self.__variance = statistics.variance(inter_arrival_times, self.__mean)
            self.__kurtosis = kurtosis(inter_arrival_times)
            self.__rate = len(self.__packets) / \
                          ((self.__packets[-1].sniff_time - self.__packets[0].sniff_time) / timedelta(seconds=1))

    def get_features(self):
        return "[ Mean: " + str(self.__mean) + " ms, " + \
               "Variance: " + str(self.__variance) + " ms, " + \
               "Kurtosis: " + str(self.__kurtosis) + ", " + \
               "Rate: " + str(self.__rate) + " packets/s ]"
