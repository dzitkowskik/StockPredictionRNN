from pylab import *
from scipy import stats
import numpy as np


class ClassificationPerformance:
    names = []
    errors = []

    def __init__(self):
        self.count = 0

    def add(self, name, error_rates):
        self.names.append(name)
        self.errors.append(error_rates)
        self.count += 1

    def compare(self):
        for i in range(self.count):
            for j in range(self.count):
                if i < j:
                    tst, pvalue = stats.ttest_ind(self.errors[i], self.errors[j])
                    if pvalue < 0.05:
                        print("{0} is significantly better than {1}".format(self.names[i], self.names[j]))
                        print("{0} avg err = {1}, {2} avg err = {3}".format(
                                self.names[i], np.average(self.errors[i]),
                                self.names[j], np.average(self.errors[j])
                        ))
                    else:
                        print("{0} and {1} are not significantly different".format(self.names[i], self.names[j]))

    def make_plots(self):
        # Boxplot to compare classifier error distributions
        figure()
        boxplot(self.errors)
        xlabel(' vs '.join(self.names))
        ylabel('K-fold cross-validation error [%]')
        xticks(range(self.count), self.names)
        show()