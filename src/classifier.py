import statistics
from scipy.stats import kurtosis
from sklearn import svm


class Classifier:

    def __init__(self, window_size: int, incremental_computation_threshold: int):
        """
        :param window_size: packets window size in seconds
        :param incremental_computation_threshold: number of packets after which the statistical information must be
        processed incrementally (this allows to compute real time big amount of data and still keep precision by
        getting back to precise formulas when the chunks are small)
        """

        self.__window_size = window_size
        self.__incremental_computation_threshold = incremental_computation_threshold

        self.__packets = []
        self.__inter_arrival_times = []

        self.__mean = 0
        self.__m2 = 0
        self.__m3 = 0
        self.__m4 = 0

    def add(self, packet):
        packet.time = float(packet.time)

        if len(self.__packets) != 0:
            x_t = (packet.time - self.__packets[-1].time) * 1000
            self.__inter_arrival_times.append(x_t)

        self.__packets.append(packet)

        # Update the statistics

        if len(self.__inter_arrival_times) <= self.__incremental_computation_threshold:
            self.__mean = statistics.mean(self.__inter_arrival_times) if len(self.__inter_arrival_times) != 0 else 0

            self.__m2 = 0
            self.__m3 = 0
            self.__m4 = 0

            for time in self.__inter_arrival_times:
                self.__m2 += pow(time - self.__mean, 2)
                self.__m3 += pow(time - self.__mean, 2)
                self.__m4 += pow(time - self.__mean, 4)

        else:
            x_t = self.__inter_arrival_times[-1]
            t = len(self.__inter_arrival_times)

            mean = self.__mean + (x_t - self.__mean) / t
            m2 = self.__m2 + (x_t - self.__mean) * (x_t - mean)

            m3 = self.__m3 - 3 * (x_t - self.__mean) * self.__m2 / t + \
                 (t - 1) * (t - 2) * pow(x_t - self.__mean, 3) / pow(t, 2)

            m4 = self.__m4 - 4 * (x_t - self.__mean) * self.__m3 / t + \
                 6 * pow((x_t - self.__mean) / t, 2) * self.__m2 + \
                 (t - 1) * (pow(t, 2) - 3 * t + 3) * pow(x_t - self.__mean, 4) / pow(t, 3)

            self.__mean = mean
            self.__m2 = m2
            self.__m3 = m3
            self.__m4 = m4

    def update_current_time(self, current_time: float):
        """
        Inform the classifier that the time is passing.
        The old packets exceeding the time window are removed.
        """

        while len(self.__packets) > 2 and \
                (current_time - self.__packets[1].time) > self.__window_size:

            self.__packets.pop(0)
            x_t = self.__inter_arrival_times.pop(0)

            # Update the statistics

            if len(self.__inter_arrival_times) <= self.__incremental_computation_threshold:
                self.__mean = statistics.mean(self.__inter_arrival_times) if len(self.__inter_arrival_times) != 0 else 0

                self.__m2 = 0
                self.__m3 = 0
                self.__m4 = 0

                for time in self.__inter_arrival_times:
                    self.__m2 += pow(time - self.__mean, 2)
                    self.__m3 += pow(time - self.__mean, 2)
                    self.__m4 += pow(time - self.__mean, 4)

            else:
                t = len(self.__inter_arrival_times) + 1

                mean = (t * self.__mean - x_t) / (t - 1)
                m2 = self.__m2 - (x_t - mean) * (x_t - self.__mean)
                m3 = self.__m3 + 3 * (x_t - mean) * m2 / t - (t - 1) * (t - 2) * pow(x_t - mean, 3) / pow(t, 2)

                m4 = self.__m4 + 4 * (x_t - mean) * m3 / t - 6 * pow((x_t - mean) / t, 2) * m2 - \
                     (t - 1) * (pow(t, 2) - 3 * t + 3) * pow(x_t - mean, 4) / pow(t, 3)

                self.__mean = mean
                self.__m2 = m2
                self.__m3 = m3
                self.__m4 = m4

    @property
    def __variance(self):
        n = len(self.__inter_arrival_times)

        if n <= 1:
            return 0

        return self.__m2 / (n - 1)

    @property
    def __kurtosis(self):
        n = len(self.__inter_arrival_times)

        if n == 0:
            return -3

        if n <= self.__incremental_computation_threshold or n <= 2:
            return kurtosis(self.__inter_arrival_times, bias=False)

        return (n * (n + 1) * self.__m4) / ((n - 1) * (n - 2) * (n - 3) * pow(self.__variance, 2)) - \
                              (3 * pow(n - 1, 2)) / ((n - 2) * (n - 3))

    @property
    def __rate(self):
        if len(self.__packets) < 2:
            return 0

        return len(self.__packets) / (self.__packets[-1].time - self.__packets[0].time)

    def get_features(self):
        time = "null" if len(self.__packets) == 0 else str(self.__packets[-1].time)

        return "[ Time: " + time + ", " + \
               "Packets: " + str(len(self.__packets)) + ", " + \
               "Mean: " + str(self.__mean) + " ms, " + \
               "Variance: " + str(self.__variance) + " ms, " + \
               "Kurtosis: " + str(self.__kurtosis) + ", " + \
               "Rate: " + str(self.__rate) + " packets/s ]"


