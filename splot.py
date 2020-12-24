import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



class SpiralPlot():

    def __init__(self, df, date_col, type_col, metric_col, plimit=10):
        df = df.copy()
        self.point_limit = plimit
        self.labels = df[type_col].unique()
        self.min_value = df[metric_col].min()
        self.max_value = df[metric_col].max()
        self.start_date = datetime.strptime(df[date_col].min(), '%Y-%m-%d')
        self.end_date = datetime.strptime(df[date_col].max(), '%Y-%m-%d')
        self.len = (self.end_date - self.start_date).days + 1
        index = []
        for date in df[date_col].to_numpy():
            index.append((datetime.strptime(date, '%Y-%m-%d') - self.start_date).days)
        df['index'] = index
        df = df.set_index('index')
        self.counts = []
        self.metrics = []
        for label in self.labels:
            sub_df = df.loc[df[type_col] == label]
            metrics_df = sub_df.groupby(by=['index'])[metric_col].apply(list)
            metrics = np.zeros(self.len, dtype=object)
            for i in range(metrics_df.shape[0]):
                metrics[metrics_df.index.values[i]] = metrics_df.values[i]
            self.metrics.append(metrics)
            counts_df = sub_df.groupby(by=['index']).count()
            counts = np.zeros(self.len, dtype=np.int)
            counts[counts_df.index.values] = counts_df[type_col].to_numpy()
            self.counts.append(counts)
        self.precision = 100
        self.step_scale = 30
        self.max_angle = self.len / 365 * 2.0 * np.pi
        self.x = np.linspace(0, self.max_angle, num=self.len, endpoint=True)
        self.xx = np.linspace(0, self.max_angle, num=self.len*self.precision, endpoint=True)
        self.base = self.step_scale * self.xx
        self.sbase = self.step_scale * self.x

    def get_lines(self):
        lines = [self.base]
        for i in range(len(self.labels)):
            f = interp1d(self.x, self.counts[i],  kind='cubic')
            y = f(self.xx)
            y[y < 0] = 0
            line = y + lines[i]
            lines.append(line)
            print(lines)
        return lines

    def get_points(self):
        level = self.sbase
        points = []
        for i in range(len(self.labels)):
            counts = self.counts[i]
            metrics = self.metrics[i]
            points_y = []
            points_x = []
            for date_ind in range(self.len):
                if metrics[date_ind] == 0:
                    continue
                m = metrics[date_ind][:self.point_limit]
                date_points = (m-self.min_value) / (self.max_value-self.min_value) * counts[date_ind] + level[date_ind]
                points_y = np.concatenate((points_y, date_points))
                points_x = np.concatenate((points_x, np.full(len(date_points), self.x[date_ind])))
            points.append({'x': points_x, 'y': points_y})
            level += self.counts[i]
        return points

    def plot(self):
        ax = plt.subplot(111, projection='polar')
        ax.yaxis.grid(False)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        ax.set_xticks(np.arange(0, 2.0*np.pi, np.pi/6.0))
        months = [x[:3] for x in calendar.month_name if x]
        ax.set_xticklabels(months)
        start_year = self.start_date.year
        end_year = self.end_date.year
        years = [year for year in range(start_year, end_year + 1)]
        ax.set_yticks(
            np.linspace(
                0.8*self.step_scale*2*np.pi,
                (len(years)-0.2)*self.step_scale*2*np.pi,
                num=len(years), endpoint=True
            )
        )
        ax.set_yticklabels(years)
        ax.set_rlabel_position(0)

        lines = self.get_lines()
        for i in range(1, len(lines)):
            ax.fill_between(self.xx, lines[i-1], lines[i], alpha=0.5)

        points = self.get_points()
        for p in points:
            ax.plot(p['x'], p['y'], '.', markersize=1)

        plt.show()
