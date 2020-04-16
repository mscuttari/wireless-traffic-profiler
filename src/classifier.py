import statistics

from collections import Counter
from joblib import dump, load
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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
        self.__s_mean = 0
        self.__s_m2 = 0

        # Inter arrival time statistics
        self.__t_mean = 0
        self.__t_m2 = 0

        self.__svc = OneVsRestClassifier(SVC(max_iter=30000), n_jobs=-1)
        self.__scaler = StandardScaler()

        self.__state_history = []
        self.__state = None

    def add(self, packet):
        packet.time = float(packet.time)

        if len(self.__packets) != 0:
            self.__inter_arrival_times.append((packet.time - self.__packets[-1].time) * 1000)

        self.__sizes.append(int(packet.length))
        self.__packets.append(packet)

        # Update the statistics

        if len(self.__packets) <= self.__incremental_computation_threshold:
            self.__s_mean = statistics.mean(self.__sizes) if len(self.__sizes) != 0 else 0
            self.__s_m2 = 0

            for size in self.__sizes:
                self.__s_m2 += pow(size - self.__s_mean, 2)

            self.__t_mean = statistics.mean(self.__inter_arrival_times) if len(self.__inter_arrival_times) != 0 else 0
            self.__t_m2 = 0

            for time in self.__inter_arrival_times:
                self.__t_m2 += pow(time - self.__t_mean, 2)

        else:
            # Packet size statistics
            x_t_size = self.__sizes[-1]
            t = len(self.__sizes)

            mean = self.__s_mean + (x_t_size - self.__s_mean) / t
            m2 = self.__s_m2 + (x_t_size - self.__s_mean) * (x_t_size - mean)

            self.__s_mean = mean
            self.__s_m2 = m2

            # Inter arrival time statistics
            x_t_time = self.__inter_arrival_times[-1]
            t = len(self.__inter_arrival_times)

            mean = self.__t_mean + (x_t_time - self.__t_mean) / t
            m2 = self.__t_m2 + (x_t_time - self.__t_mean) * (x_t_time - mean)

            self.__t_mean = mean
            self.__t_m2 = m2

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
                self.__s_mean = statistics.mean(self.__sizes) if len(self.__sizes) != 0 else 0

                self.__s_m2 = 0

                for size in self.__sizes:
                    self.__s_m2 += pow(size - self.__s_mean, 2)

                # Inter arrival time statistics
                self.__t_mean = statistics.mean(self.__inter_arrival_times) if len(self.__inter_arrival_times) != 0 else 0

                self.__t_m2 = 0

                for time in self.__inter_arrival_times:
                    self.__t_m2 += pow(time - self.__t_mean, 2)

            else:
                # Packet size statistics
                t = len(self.__sizes) + 1

                mean = (t * self.__s_mean - x_t_size) / (t - 1)
                m2 = self.__s_m2 - (x_t_size - mean) * (x_t_size - self.__s_mean)

                self.__s_mean = mean
                self.__s_m2 = m2

                # Inter arrival time statistics
                t = len(self.__inter_arrival_times) + 1

                mean = (t * self.__t_mean - x_t_time) / (t - 1)
                m2 = self.__t_m2 - (x_t_time - mean) * (x_t_time - self.__t_mean)

                self.__t_mean = mean
                self.__t_m2 = m2

    @property
    def __size_mean(self):
        return self.__s_mean

    @property
    def __size_variance(self):
        n = len(self.__sizes)

        if n <= 1:
            return 0

        return min(self.__s_m2 / (n - 1), n * pow(2304 / 2, 2) / (n - 1))

    @property
    def __time_mean(self):
        return min(self.__t_mean, self.__window_size * 1000)

    @property
    def __time_variance(self):
        n = len(self.__inter_arrival_times)

        if n <= 1:
            return 0

        variance = self.__t_m2 / (n - 1)
        return min(variance, n * pow(self.__window_size * 1000 / 2, 2) / (n - 1))

    @property
    def __rate(self):
        return len(self.__packets) / self.__window_size

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

        # Print train accuracy
        print("Confusion matrix:\n" + str(confusion_matrix(y_test, y_pred)) + "\n")

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        print("Features importance: " + str(tree.feature_importances_) + "\n")

        print("Test accuracy: " + str(accuracy_score(y_test, y_pred)) + "\n")

    def save_trained_model(self, path: str):
        """
        Save the trained model to file.
        The saved model includes the scaler used in the training process, in order to allow for
        normalization of live features using the same normalization parameters as in training phase.

        :param path: where the model has to be saved
        """
        dump([self.__svc, self.__scaler], path)

    def load_trained_model(self, model_path: str):
        """
        Load pre-trained model and scaler from file.

        :param model_path: pre-trained model path
        """
        self.__svc, self.__scaler = load(model_path)

    def predict(self):
        features = self.features
        data = [features]
        state = self.__svc.predict(self.__scaler.transform(data))[0]

        if len(self.__state_history) == self.__window_size:
            self.__state_history.pop(0)

        self.__state_history.append(state)
        state = Counter(self.__state_history).most_common(n=1)[0][0]
        print("Features: %s, States: %s" % (features, Counter(self.__state_history).most_common()))

        if self.__state != state:
            self.__state = state
            print("Features: %s, State: %s" % (features, state))
            #print("\r%s" % state, end='')
