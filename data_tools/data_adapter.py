# -*- coding: utf-8 -*-

class DataAdapter(object):
    def __init__(self, features, answer):
        self._features = self._transform_features(features)
        self._answer = self._transform_answer(answer)

    def _transform_features(self, data):
        return map(lambda x: [x], data)

    def _transform_answer(self, data):
        return list(data)

    def get_data(self):
        return {'x': self._features, 'y': self._answer}
