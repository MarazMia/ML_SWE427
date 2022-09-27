# x_term = all the values of Xi-Xavg
# y_term = all the values of Yi-Yavg
# x_term_squared_sum = sum of all x_term[i]**2
# xy_term_sum = sum of x_term[i]*y_term[i]

import matplotlib.pyplot as plt
import random as rand
import math


class linear_regression():
    def __init__(self, x, y):

        if len(x) != len(y):
            raise Exception(
                'length of the two lists are not equal to each other!')

        self.n = len(x)

        self.x = x
        self.y = y

        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.Y_test_actual = []

        self.selected_test_indices = set()

        self.M = 0
        self.C = 0

    def train_test_split(self):

        desired_split_number = math.ceil(self.n*0.33)

        mn = min(self.x)
        mx = max(self.x)

        for i in range(self.n):
            if self.x[i] == mn:
                self.X_test.append(mn)
                self.Y_test_actual.append(self.y[i])
                break
        for i in range(self.n):
            if self.x[i] == mx:
                self.X_test.append(mx)
                self.Y_test_actual.append(self.y[i])
                break

        while len(self.selected_test_indices) < desired_split_number:
            self.selected_test_indices.add(rand.randint(0, self.n-1))

        for idx in range(self.n):
            if idx in self.selected_test_indices:
                self.X_test.append(self.x[idx])
                self.Y_test_actual.append(self.y[idx])
            else:
                self.X_train.append(self.x[idx])
                self.Y_train.append(self.y[idx])

    def train(self):

        self.train_test_split()

        # print(self.X_train, self.Y_train)

        x_term = self.individual_value_substract_avg(self.X_train)
        y_term = self.individual_value_substract_avg(self.Y_train)

        x_term_squared_sum, xy_term_sum = 0, 0
        for i in range(len(x_term)):
            x_term_squared_sum += x_term[i]**2
            xy_term_sum += x_term[i]*y_term[i]

        self.M = xy_term_sum/x_term_squared_sum
        self.C = self.Avg(self.Y_train)-self.M*self.Avg(self.X_train)
        print('M value =', self.M, '   C value', self.C)

        for it in self.X_test:
            self.Y_test.append(self.predict(it))

        print('MEAN SQUARED ERROR ', self.MSE())

    def fit(self):

        plt.scatter(self.x, self.y, label='actual values')
        plt.plot(self.X_test, self.Y_test, color='green', marker='o',
                 markerfacecolor='red', markersize=8, label='regression line')
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt.legend()
        plt.show()

    def predict(self, x_val):

        return self.M*x_val+self.C

    def MSE(self):

        y_test_len = len(self.Y_test)
        MSerr = 0
        for i in range(y_test_len):
            MSerr += (self.y[i]-self.Y_test[i])**2

        return MSerr/y_test_len

    def Avg(self, a):

        if len(a) == 0:
            return 0
        return sum(a)/len(a)

    def individual_value_substract_avg(self, a):

        ans = []
        avg = self.Avg(a)
        for i in a:
            ans.append(i-avg)
        return ans


def main():
    x = [105, 110, 115, 120, 125, 129, 130, 135,
         140, 145, 150, 155, 160, 165, 168, 170]
    y = [14, 14.5, 16, 17, 17.5, 18, 20, 20.5,
         21.8, 22.6, 25, 25.9, 26.5, 27, 27.3, 29]
    obj = linear_regression(x, y)
    obj.train()
    obj.fit()


if __name__ == '__main__':
    main()
