from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import accuracy_score, mean_squared_error

class MetricPlotter:
    """
    評価指標を切り替えて学習曲線をプロットするクラス

    Attributes:
        estimator: データを学習するためのモデル。
        title: グラフのタイトル。
        metric: 使用する評価指標。['r2', 'rmse', 'auc', 'accuracy']

    Methods:
        plot_learning_curve(X, y): 学習曲線をプロットする。
    """

    def __init__(self, estimator, title, metric='r2'):
        """
        MetricPlotterを初期化

        Parameters:
            estimator: モデルのオブジェクト
            title (str): グラフのタイトル
            metric (str, optional): 使用する評価指標、デフォルトは 'r2'
        """
        self.estimator = estimator
        self.title = title
        self.metric = metric
        self.scorer = self._get_scorer(metric)

    def _get_scorer(self, metric):
        """
        評価指標に基づいたスコアリング方法を返す

        Parameters:
            metric (str): 評価指標の名称

        Returns:
            callable: スコアリング関数
        """
        if metric == 'r2':
            return None
        elif metric == 'rmse':
            return make_scorer(lambda y_true, y_pred:
                np.sqrt(mean_squared_error(y_true, y_pred)),
                greater_is_better=False)
        elif metric == 'auc':
            return make_scorer(roc_auc_score, needs_proba=True)
        elif metric == 'accuracy':
            return make_scorer(accuracy_score)
        else:
            raise ValueError('Unknown metric: {}'.format(metric))
    
    def _get_ylabel(self):
        """評価指標に基づいてy軸のラベルを返す"""
        metric_labels = {
            'r2': 'R^2',
            'rmse': 'RMSE',
            'auc': 'AUC',
            'accuracy': 'Accuracy'
        }
        return metric_labels.get(self.metric, '')

    def plot_learning_curve(self, X, y, ylim=None, cv=None, n_jobs=1,
                            train_sizes=np.linspace(.1, 1.0, 10),
                            figsize=(8, 5)):
        """
        データと評価指標に基づいて学習曲線をプロットする

        Parameter:
            X: 学習データ
            y: ターゲットデータ
            ylim (tuple, optional): プロットされるyの最小値と最大値を定義
            cv: クロスバリデーションの分割戦略
            n_jobs (int, optional): 並行して実行するジョブ数、デフォルトは1
            train_sizes (array-like, optional): 学習データの相対的または絶対的な数
            デフォルトは np.linspace(.1, 1.0, 10)
            figsize: グラフのサイズ
        """
        
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.set_title(self.title)

        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel('Training samples')
        ax.set_ylabel(self._get_ylabel())
        
        
        train_sizes, train_scores, test_scores = learning_curve(self.estimator,
                                                                X, y, cv=cv,
                                                                scoring=self.scorer,
                                                                n_jobs=n_jobs,
                                                                train_sizes=train_sizes,
                                                                shuffle=True)
        if self.metric == 'rmse':
            train_scores = -train_scores
            test_scores = -test_scores

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='r')
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color='g')
        ax.plot(train_sizes, train_scores_mean, 'o-', color='r',
                label='Training score')
        ax.plot(train_sizes, test_scores_mean, 'o-', color='g',
                label='Validation score')

        ax.legend(loc='best')

        def _custom_format(row):
            if row.name == 'Training samples':
                return row.map('{:.0f}'.format)
            else:
                return row.map('{:.3f}'.format)

        # DataFrameでスコアを表示
        df_results = pd.DataFrame({
            'Training samples': train_sizes,
            'Training score': train_scores_mean,
            'Validation score': test_scores_mean
        })
        display(df_results.T.apply(_custom_format, axis=1))
        plt.show()
        
        return ax
