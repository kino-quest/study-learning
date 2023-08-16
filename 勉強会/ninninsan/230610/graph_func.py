import matplotlib.pyplot as plt
import numpy as np

class Mix_in:
    '''
    Mix_in(__init__のないclassで機能追加関数を定義)
    tick_paramsとticklabel_formatを合成してデフォルト値を設定
    '''
    def default_ticks(self, ax, style='plain', rotation=0,
                      labelsize=12, axis=['x', 'y']):
        ax.tick_params(axis=axis[0], rotation=rotation,
                       labelsize=labelsize)
        ax.tick_params(axis=axis[1], labelsize=labelsize)
        ax.ticklabel_format(style=style, axis=axis[1])

class Descriptor:
    '''
    デスクリプタを使って追加の属性変更を実施
    本来の使い方とは異なるが、ここでは割り込み処理での機能追加に使用
    super().__init__()で初期化される変数にデスクリプタのインスタンスを
    忍び込ませており、追加コードを省略して全体の設定が実現できている
    '''
    def __get__(self, obj, objstyle=None):
        plt.rcParams['font.size'] = 12
        plt.rcParams['lines.marker'] = 'o'
        print('Hi, I am descriptor!')
        return

class Customized_bars(Mix_in):
    '''
    描画データをDataFrameのindexとcolumnsをインデックス参照して
    積上げ棒グラフ、集合棒グラフを描画するベースのclass
    ここではMixinを継承
    
        args
            idx   : index
            cols  : columns
            data  : DataFrame
            ax    : axes
        methods
            stacked_bar(self, ax): 積上げ棒グラフ(縦)
            stacked_barh(self, ax): 積上げ棒グラフ(横)
            clustered_bar1(self, ax): 集合棒グラフ1(縦)
            clustered_bar2(self, ax): 集合棒グラフ2(縦)
            clustered_barh(self, ax): 集合棒グラフ(横)
        usage
            fig, (ax1, ax2) = plt.subplots(1, 2)
            df = pd.DataFrame(data, args)
            idx = df.index[3: 15]
            cols1 = df.columns[[3, 4, 5]]
            cg = Customized_bars(idx=idx, cols=cols, data=df)
            cg.stacked_bar(ax=ax1)
            cg.clustered_bar1(ax=ax2)
            cg.clustered_bar2(ax=ax2)
    '''
    att = Descriptor()

    def __init__(self, idx, cols, data):
        self.idx = idx
        self.cols = cols
        self.data = data
        self.att = Customized_bars.att

    def stacked_bar(self, ax):
        bottom = np.zeros_like(self.idx)
        for c in self.cols:
            ax.bar(self.idx, self.data.loc[self.idx, c],
                   bottom=bottom, label=c)
            bottom += self.data.loc[self.idx, c]
        super().default_ticks(ax=ax)
        ax.legend()

    def stacked_barh(self, ax):
        left = np.zeros_like(self.idx)
        for c in self.cols:
            ax.barh(self.idx, self.data.loc[self.idx, c],
                    left=left, label=c)
            left += self.data.loc[self.idx, c]
        super().default_ticks(ax=ax, axis=['y', 'x'], rotation=0)
        ax.legend()

    def clustered_bar1(self, ax):
        x = np.arange(len(self.idx))
        width = 0.8 / len(self.cols)
        for i, c in enumerate(self.cols):
            pos = x -0.4 + width * i
            ax.bar(x=pos, height=self.data.loc[self.idx, c].T,
                   width=width, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(self.idx)
        super().default_ticks(ax=ax)
        ax.legend()

    def clustered_bar2(self, ax):
        x = np.arange(len(self.cols))
        width = 0.8 / len(self.idx)
        for i, r in enumerate(self.idx):
            pos = x -0.4 + width * i
            ax.bar(x=pos, height=self.data.loc[r, self.cols],
                   width=width, label=r)
        ax.set_xticks(x)
        ax.set_xticklabels(self.cols)
        super().default_ticks(ax=ax)
        ax.legend()

    def clustered_barh1(self, ax):
        y = np.arange(len(self.idx))
        height = 0.8 / len(self.cols)
        for i, c in enumerate(self.cols):
           pos = y -0.4 + height * i
           ax.barh(y=pos, width=self.data.loc[self.idx, c].T,
                   height=height, label=c)
        ax.set_yticks(y)
        ax.set_yticklabels(self.idx)
        super().default_ticks(ax=ax, axis=['y', 'x'], rotation=0)
        ax.legend()

    def clustered_barh2(self, ax):
        y = np.arange(len(self.cols))
        height = 0.8 / len(self.idx)
        for i, r in enumerate(self.idx):
           pos = y -0.4 + height * i
           ax.barh(y=pos, width=self.data.loc[r, self.cols],
                   height=height, label=r)
        ax.set_yticks(y)
        ax.set_yticklabels(self.cols)
        super().default_ticks(ax=ax, axis=['y', 'x'], rotation=0)
        ax.legend()


class Dual_axis(Customized_bars):
    '''
    Customized_barsの縦型グラフに2軸の折れ線グラフを追加する

    '''
    def __init__(self, idx, cols, cols2, data):
        super().__init__(idx, cols, data)
        self.idx = idx
        self.cols = cols
        self.cols2 = cols2
        self.data = data

    def stacked_bar_with_lines(self, ax):
        super().stacked_bar(ax=ax)
        axt = ax.twinx()
        for c in self.cols2:
            axt.plot(self.idx, self.data.loc[self.idx, c],
                     label=c)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axt.get_legend_handles_labels()
        super().default_ticks(ax=ax)
        ax.legend(h1 + h2, l1 + l2)

    def clustered_bar1_with_lines(self, ax):
        super().clustered_bar1(ax=ax)
        axt = ax.twinx()
        for c in self.cols2:
            axt.plot(self.idx, self.data.loc[self.idx, c],
                     label=c)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axt.get_legend_handles_labels()
        super().default_ticks(ax=ax)
        ax.legend(h1 + h2, l1 + l2)
