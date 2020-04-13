import statistics

from joblib import dump, load
from scipy.stats import kurtosis
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


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

        self.__packets = []              # Arrived packets
        self.__sizes = []                # Position i contains the size of packet i of __packets
        self.__inter_arrival_times = []  # Position i contains the inter arrival time between packet i and i + 1 of __packets

        # Packet size statistics
        self.__size_mean = 0
        self.__size_m2 = 0
        self.__size_m3 = 0
        self.__size_m4 = 0

        # Inter arrival time statistics
        self.__time_mean = 0
        self.__time_m2 = 0
        self.__time_m3 = 0
        self.__time_m4 = 0

        self.__svc = OneVsRestClassifier(SVC(max_iter=20000))
        self.__scaler = StandardScaler()

        self.__state = '-'

    def add(self, packet):
        packet.time = float(packet.time)

        if len(self.__packets) != 0:
            self.__inter_arrival_times.append((packet.time - self.__packets[-1].time) * 1000)

        self.__sizes.append(int(packet.length))
        self.__packets.append(packet)

        # Update the statistics

        if len(self.__packets) <= self.__incremental_computation_threshold:
            self.__size_mean = statistics.mean(self.__sizes) if len(self.__sizes) != 0 else 0
            self.__size_m2 = 0
            self.__size_m3 = 0
            self.__size_m4 = 0

            for size in self.__sizes:
                self.__size_m2 += pow(size - self.__size_mean, 2)
                self.__size_m3 += pow(size - self.__size_mean, 2)
                self.__size_m4 += pow(size - self.__size_mean, 4)

            self.__time_mean = statistics.mean(self.__inter_arrival_times) if len(self.__inter_arrival_times) != 0 else 0
            self.__time_m2 = 0
            self.__time_m3 = 0
            self.__time_m4 = 0

            for time in self.__inter_arrival_times:
                self.__time_m2 += pow(time - self.__time_mean, 2)
                self.__time_m3 += pow(time - self.__time_mean, 2)
                self.__time_m4 += pow(time - self.__time_mean, 4)

        else:
            # Packet size statistics
            x_t_size = self.__sizes[-1]
            t = len(self.__sizes)

            mean = self.__size_mean + (x_t_size - self.__size_mean) / t
            m2 = self.__size_m2 + (x_t_size - self.__size_mean) * (x_t_size - mean)

            m3 = self.__size_m3 - 3 * (x_t_size - self.__size_mean) * self.__size_m2 / t + \
                 (t - 1) * (t - 2) * pow(x_t_size - self.__size_mean, 3) / pow(t, 2)

            m4 = self.__size_m4 - 4 * (x_t_size - self.__size_mean) * self.__size_m3 / t + \
                 6 * pow((x_t_size - self.__size_mean) / t, 2) * self.__size_m2 + \
                 (t - 1) * (pow(t, 2) - 3 * t + 3) * pow(x_t_size - self.__size_mean, 4) / pow(t, 3)

            self.__size_mean = mean
            self.__size_m2 = m2
            self.__size_m3 = m3
            self.__size_m4 = m4

            # Inter arrival time statistics
            x_t_time = self.__inter_arrival_times[-1]
            t = len(self.__inter_arrival_times)

            mean = self.__time_mean + (x_t_time - self.__time_mean) / t
            m2 = self.__time_m2 + (x_t_time - self.__time_mean) * (x_t_time - mean)

            m3 = self.__time_m3 - 3 * (x_t_time - self.__time_mean) * self.__time_m2 / t + \
                 (t - 1) * (t - 2) * pow(x_t_time - self.__time_mean, 3) / pow(t, 2)

            m4 = self.__time_m4 - 4 * (x_t_time - self.__time_mean) * self.__time_m3 / t + \
                 6 * pow((x_t_time - self.__time_mean) / t, 2) * self.__time_m2 + \
                 (t - 1) * (pow(t, 2) - 3 * t + 3) * pow(x_t_time - self.__time_mean, 4) / pow(t, 3)

            self.__time_mean = mean
            self.__time_m2 = m2
            self.__time_m3 = m3
            self.__time_m4 = m4

    def update_current_time(self, current_time: float):
        """
        Inform the classifier that the time is passing.
        The old packets exceeding the time window are removed.

        :param current_time: last and most recently known capture time
        """

        while len(self.__packets) > 2 and (current_time - self.__packets[1].time) > self.__window_size:

            self.__packets.pop(0)
            x_t_size = self.__sizes.pop(0)
            x_t_time = self.__inter_arrival_times.pop(0)

            # Update the statistics

            if len(self.__packets) <= self.__incremental_computation_threshold:
                # Packet size statistics
                self.__size_mean = statistics.mean(self.__sizes) if len(self.__sizes) != 0 else 0

                self.__size_m2 = 0
                self.__size_m3 = 0
                self.__size_m4 = 0

                for size in self.__sizes:
                    self.__size_m2 += pow(size - self.__size_mean, 2)
                    self.__size_m3 += pow(size - self.__size_mean, 2)
                    self.__size_m4 += pow(size - self.__size_mean, 4)

                # Inter arrival time statistics
                self.__time_mean = statistics.mean(self.__inter_arrival_times) if len(self.__inter_arrival_times) != 0 else 0

                self.__time_m2 = 0
                self.__time_m3 = 0
                self.__time_m4 = 0

                for time in self.__inter_arrival_times:
                    self.__time_m2 += pow(time - self.__time_mean, 2)
                    self.__time_m3 += pow(time - self.__time_mean, 2)
                    self.__time_m4 += pow(time - self.__time_mean, 4)

            else:
                # Packet size statistics
                t = len(self.__sizes) + 1

                mean = (t * self.__size_mean - x_t_size) / (t - 1)
                m2 = self.__size_m2 - (x_t_size - mean) * (x_t_size - self.__size_mean)
                m3 = self.__size_m3 + 3 * (x_t_size - mean) * m2 / t - (t - 1) * (t - 2) * pow(x_t_size - mean, 3) / pow(t, 2)

                m4 = self.__size_m4 + 4 * (x_t_size - mean) * m3 / t - 6 * pow((x_t_size - mean) / t, 2) * m2 - \
                     (t - 1) * (pow(t, 2) - 3 * t + 3) * pow(x_t_size - mean, 4) / pow(t, 3)

                self.__size_mean = mean
                self.__size_m2 = m2
                self.__size_m3 = m3
                self.__size_m4 = m4

                # Inter arrival time statistics
                t = len(self.__inter_arrival_times) + 1

                mean = (t * self.__time_mean - x_t_time) / (t - 1)
                m2 = self.__time_m2 - (x_t_time - mean) * (x_t_time - self.__time_mean)
                m3 = self.__time_m3 + 3 * (x_t_time - mean) * m2 / t - (t - 1) * (t - 2) * pow(x_t_time - mean, 3) / pow(t, 2)

                m4 = self.__time_m4 + 4 * (x_t_time - mean) * m3 / t - 6 * pow((x_t_time - mean) / t, 2) * m2 - \
                     (t - 1) * (pow(t, 2) - 3 * t + 3) * pow(x_t_time - mean, 4) / pow(t, 3)

                self.__time_mean = mean
                self.__time_m2 = m2
                self.__time_m3 = m3
                self.__time_m4 = m4

    @property
    def __size_variance(self):
        n = len(self.__sizes)

        if n <= 1:
            return 0

        return self.__size_m2 / (n - 1)

    @property
    def __size_kurtosis(self):
        n = len(self.__sizes)

        if n == 0:
            return -3

        if n <= 3:
            return kurtosis(self.__sizes, bias=False)

        var = self.__size_variance

        if var == 0:
            return -3

        return (n * (n + 1) * self.__size_m4) / ((n - 1) * (n - 2) * (n - 3) * pow(var, 2)) - \
               (3 * pow(n - 1, 2)) / ((n - 2) * (n - 3))

    @property
    def __time_variance(self):
        n = len(self.__inter_arrival_times)

        if n <= 1:
            return 0

        return self.__time_m2 / (n - 1)

    @property
    def __time_kurtosis(self):
        n = len(self.__inter_arrival_times)

        if n == 0:
            return -3

        if n <= 3:
            return kurtosis(self.__inter_arrival_times, bias=False)

        var = self.__time_variance

        if var == 0:
            return -3

        return (n * (n + 1) * self.__time_m4) / ((n - 1) * (n - 2) * (n - 3) * pow(var, 2)) - \
               (3 * pow(n - 1, 2)) / ((n - 2) * (n - 3))

    @property
    def __rate(self):
        if len(self.__packets) < 2:
            return 0

        return len(self.__packets) / (self.__packets[-1].time - self.__packets[0].time)

    @property
    def features(self):
        return [self.__size_mean, self.__size_variance,
                self.__time_mean, self.__time_variance,
                self.__rate]

    def train(self, x, y):
        """
        Train the SVM.

        :param x: training data (array of data, where each entry is an array of features)
        :param y: labels corresponding to the provided data
        :return accuracy
        """

        # Divide data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

        # Train and evaluate accuracy
        x_train = self.__scaler.fit_transform(x_train)
        x_test = self.__scaler.transform(x_test)

        self.__svc.fit(x_train, y_train)
        y_pred = self.__svc.predict(x_test)

        return accuracy_score(y_test, y_pred)

    def save_trained_model(self, path: str):
        """
        Save the trained model to disk.

        :param path: where the model has to be saved
        """
        dump([self.__svc, self.__scaler], path)

    def load_trained_model(self, model_path: str):
        """
        Load pre-trained model from disk.

        :param model_path: pre-trained model path
        """
        self.__svc, self.__scaler = load(model_path)
        print("Mean: " + str(self.__scaler.mean_))
        print("Var: " + str(self.__scaler.var_))

    def print_features(self):
        time = 0 if len(self.__packets) == 0 else self.__packets[-1].time
        features = self.features

        print("[ Time: %.2f, Packets: %d, Size mean: %.2f bytes, Size var: %.2f bytes, Time mean: %.2f ms, Time var: %.2f ms, Rate: %.2f packets/s ]" % \
              (time, len(self.__packets), features[0], features[1], features[2], features[3], features[4]))

    def predict(self):
        data = [self.features]
        data = self.__scaler.transform(data)
        action = self.__svc.predict(data)[0]

        if action != self.__state:
            print("Action: " + str(action))
            self.__state = action
