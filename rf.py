import pandas as pd
import numpy as np
from datetime import datetime
import math
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict


class Rf():
    def __init__(self, sessions):
        self.data = self.prepare_data(sessions)
        self.create_model()

    def rf_predict(self, rawdata):
        rawdata = rawdata.replace("'", '"')
        rawdata = rawdata.replace("None", '0')

        data = self.prepare_data(rawdata)

        # creating input matrix with features in following order:
        # session time, discount, number of products viewed, day, month, percent of sessions ended buy
        features = np.c_[[session['time'] for session in data],
                         [x['discount'] for x in data],
                         [len(x['products']) for x in data],
                         [x['date'].day for x in data],
                         [x['date'].month for x in data],
                         [x['date'].weekday() for x in data],
                         self.get_user_buy_history(data), self.getPopularity(data)]

        predictions = self.rf.predict(features)
        predictions = predictions.round()
        predictions = predictions.astype(int)
        return str(predictions)

    def create_model(self):
        labels = np.array([session['label'] for session in self.data])

        # creating input matrix with features in following order:
        # session time, discount, number of products viewed, day, month, percent of sessions ended buy
        features = np.c_[[session['time'] for session in self.data],
                         [x['discount'] for x in self.data],
                         [len(x['products']) for x in self.data],
                         [x['date'].day for x in self.data],
                         [x['date'].month for x in self.data],
                         [x['date'].weekday() for x in self.data],
                         self.get_user_buy_history(self.data), self.getPopularity(self.data)]

        train_features, self.test_features, train_labels, self.test_labels = train_test_split(
            features, labels, test_size=0.25)

        self.rf = RandomForestClassifier(n_estimators=633, random_state=27, min_samples_split=2,
                                        min_samples_leaf=4, max_features='auto', max_depth=50, bootstrap=True)
        self.rf.fit(train_features, train_labels)

    def accuracy(self):
        predictions = self.rf.predict(self.test_features)

        predictions = predictions.round()
        predictions = predictions.astype(int)

        num_false = (predictions == self.test_labels).sum()
        rf_acc = round(100 * num_false / len(predictions), 2)

        p = 0
        for x in self.test_labels:
            if x == 0:
                p += 1

        score = round(100 * p/len(self.test_labels), 2)
        diff = rf_acc - score

        return str(rf_acc), str(score), str(round(diff, 2))

    def prepare_data(self, sessions):
        features = pd.read_json(sessions, lines=True)
        # remove NaNs from data, replace with 0
        features['user_id'].fillna(0, inplace=True)
        features['user_id'] = features['user_id'].astype(int)
        features['product_id'].fillna(0, inplace=True)
        features['product_id'] = features['product_id'].astype(int)
        features['purchase_id'].fillna(0, inplace=True)
        features['purchase_id'] = features['purchase_id'].astype(int)
        self.dataForPopularity = features
        data = []
        id = 0
        timestamp_start = ''
        timestamp_end = ''

        for index, row in features.iterrows():
            if id != row[0]:
                if id != 0:
                    time = self.calculate_time(timestamp_start, timestamp_end)
                    data[-1]['time'] = time

                data.append({'id': row[0], 'user': row[2], 'products': [
                            row[3]], 'discount': row[5], 'date': row[1]})

                if row[6] == 0:
                    data[-1]['label'] = 0
                else:
                    data[-1]['label'] = 1

                id = row[0]
                timestamp_start = row[1]
                timestamp_end = row[1]

            elif id == row[0]:
                timestamp_end = row[1]
                data[-1]['products'].append(row[3])
                if data[-1]['user'] == 0:
                    data[-1]['user'] = row[2]
                if data[-1]['label'] == 0:
                    if row[6] != 0:
                        data[-1]['label'] = 1

        time = self.calculate_time(timestamp_start, timestamp_end)
        data[-1]['time'] = time

        data = self.handle_user_id_missing_data(data)
        return data

    def handle_user_id_missing_data(self, data):
        i = 0
        for i in range(len(data)):
            if data[i]['user'] == 0 and data[i - 1]['user'] == data[i + 1]['user']:
                data[i]['user'] = data[i - 1]['user']

        to_remove = []
        for x in data:
            if x['user'] == 0:
                to_remove.append(x)
                # wyrzucam 12 sesji z danych
        for x in to_remove:
            data.remove(x)

        return data

    def get_user_buy_history(self, data):
        users = {}
        for x in data:
            user_id = x['user']
            if user_id not in users:
                users[user_id] = {'all': 1, 'buy': 0}
                if x['label'] == 1:
                    users[user_id]['buy'] += 1
            else:
                users[user_id]['all'] += 1
                if x['label'] == 1:
                    users[user_id]['buy'] += 1

        percentages = {}

        for user, value in users.items():
            percentages[user] = round(100 * value['buy']/value['all'], 2)

        percentage = []
        for x in data:
            percentage.append(percentages[x['user']])

        return percentage

    # liczy czas między timestampami
    def calculate_time(self, start, end):
        td = end - start
        return round(td.total_seconds()/60, 2)  # minutes

    def getPopularity(self, data):
        lsorted = self.dataForPopularity.sort_values('timestamp')
        productHistory = defaultdict(list)
        for index, row in lsorted.iterrows():
            if row['event_type'] == "BUY_PRODUCT":
                productHistory[row['product_id']].append(row)
        popularity = []
        for record in data:
            date = record['date']
            counterForRecord = 0
            for product in record['products']:
                counter = 0
                for x in productHistory[product]:
                    if (date - x['timestamp']).days < 0:
                        break
                    if (date - x['timestamp']).days < 31:
                        counter += 1
                if counter > counterForRecord:
                    counterForRecord = counter
            popularity.append(counterForRecord)
        return popularity
