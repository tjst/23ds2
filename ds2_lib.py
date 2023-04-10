import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
## ver 2023.4.9
## 名称
### 表示k次元データ... (k=1,2,3) k次元データ集合の表示 あるいは k次元データ表示
### 表示k変数関数...   (k=1,2) k変数関数のグラフ表示  k変数関数表示
### 生成k次元データ

# 実数集合Dの表示
def 表示１次元データ(データリスト,ラベルリスト=[],枠=[600,300],毛=True,ヒストグラム=False,密度=False,ビンサイズ=1.0):
    """ このコードは、1次元データ（複数）を表示するためのものです。
        データリスト：[D1,D2,...], Diは (N_i,1)型のnp.array
        ラベルリスト：各データのラベル. デフォルトでは [1,2,...]
        毛 (=True): 線分で表示するにはTrue
        ヒストグラム (=False) : ヒストグラムで表示するときは True
        密度(=False) : 密度関数で表示するときは True
        ビンサイズ=1.0 : ヒストグラムのビンの大きさ
        枠=[600,200]: 描画領域の横幅と高さ
    """
    import plotly.figure_factory as ff
    if len(ラベルリスト)==0:
        ラベルリスト=range(len(データリスト))
    fig= ff.create_distplot(データリスト,ラベルリスト,show_rug=毛, show_hist=ヒストグラム,show_curve=密度,bin_size=ビンサイズ)
    width,height=枠
    fig.update_layout(width=width,height=height)
    return fig

def 表示２次元データ(データリスト,色リスト=[],枠=[600,600],透明度=1.0):
    """ 二次元データを表示 
        データリスト:２次元データリスト [[1,2],[2,3],...] 
        色リスト (=[]): 各データの色を指定するリスト(空の時は同じ色)
        枠 (=[600,600]) : 表示枠の横と縦
        透明度 ( =1.0 ): 点の透明度 
    
    """
    width,height=枠
    D=np.array(データリスト)
    if len(色リスト)==0:
        色リスト=np.zeros(len(D))
    fig=px.scatter(x=D[:,0],y=D[:,1],opacity=透明度,color=色リスト)
    fig.update_layout(width=width,height=height)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_coloraxes(showscale=False)
    return fig

def 表示２次元データヒストグラム(データリスト,xビン数=10,yビン数=10,周辺分布=False,枠=[600,600]):
    """ ２次元データのヒストグラム表示
        データリスト: ２次元データリスト [[1,2],[2,3],...]
        xビン数=10: x方向の分割数
        yビン数=10: y方向の分割数
        周辺分布=True の時、周辺分布のヒストグラムも表示
        枠=[600,600]: 表示枠の横と縦 
    """
    D = np.array(データリスト)
    if 周辺分布:
        fig = px.density_heatmap(x=D.T[0], y=D.T[1],nbinsx=xビン数,nbinsy=yビン数,marginal_x="histogram", marginal_y="histogram")
    else:
        fig = px.density_heatmap(x=D.T[0], y=D.T[1],nbinsx=xビン数,nbinsy=yビン数)
    width,height=枠
    fig.update_layout(width=width,height=height)
    return fig

def 表示２次元データ密度等高線(データリスト,枠=[600,600],周辺分布=False,濃淡=False,ラベル=False):
    """ ２次元データの密度関数を等高線で表示
        データリスト: ２次元データリスト [[1,2],[2,3],...]
        周辺分布=True の時、周辺分布のヒストグラムも表示
        濃淡=True の時、密度の多少を濃淡で示す
        ラベル=True の時、等高線における密度を表示する
        枠=[600,600]: 表示枠の横と縦 
    """
    D=np.array(データリスト)
    if 周辺分布:
        fig = px.density_contour(x=D.T[0], y=D.T[1],marginal_x="histogram", marginal_y="histogram")
    else:
        fig = px.density_contour(x=D.T[0], y=D.T[1])
    if 濃淡:
        fig.update_traces(contours_coloring="fill")
    if ラベル:
        fig.update_traces(contours_showlabels = True)
    width,height=枠
    fig.update_layout(width=width,height=height)
    return fig

def 表示３次元データ(D,Dcolor=[],点サイズ=2,透明度=0.6,形状="circle",枠=[500,500]):
    """
    (N,3) 型のデータ D を 空間内にプロットする
    Dcolor : 各データの色指定. ( Dcolor=[] の時は青で描写)
    点サイズ : 各点のサイズ( =2 )
    形状 : 各点の形状 ( 'circle', 'circle-open', 'cross', 'diamond',
        'diamond-open', 'square', 'square-open', 'x'])
    透明度: ０.0 〜 1.0 ( デフォルト 0.6)
    枠 = [500,500]: 描画する領域の横・縦

    """
    fig=px.scatter()
    D=np.array(D)
    if len(Dcolor)==0:
        Dcolor=["blue" for i in range(len(D))]
    fig.add_trace(go.Scatter3d(x=D[:,0],y=D[:,1],z=D[:,2],mode="markers",\
                               marker=dict(size=点サイズ, symbol=形状, color=Dcolor),opacity=透明度))
    width,height=枠
    fig.update_layout(width=width,height=height)    
    return fig        
                  
def 表示多次元データ２次元射影(D,色リスト=[],透明度=0.7,枠=[1000,1000],seaborn=False):
    """
    D: n次元データ (shape (N,n))
    ２次元への射影 ( n(n-1)/2 個)　と1次元への射影をプロット表示
    色リスト: 各点の色の指定 ( デフォルト [] では 青色)
    透明度: ０.0 〜 1.0 ( デフォルト 0.7)
    枠 = [1000,1000]: 描画する領域の横・縦
    seaborn : ライブラリ seaborn を用いるとき True (デフォルトでは False) 
    """
    N=len(D)
    width,height=枠
    if len(色リスト)==0:
        色リスト=np.zeros(len(D))
    if seaborn:
        import seaborn as sbn
        Df=pd.DataFrame(D)
        Df["color"]=色リスト
        sbn.pairplot(Df,kind="kde",hue="color")        
    else:
        fig=px.scatter_matrix(pd.DataFrame(D),opacity=透明度,color=色リスト,dimensions=list(range(len(D[0]))))
        fig.update_layout(width=width,height=height)
        fig.update_coloraxes(showscale=False)
        return fig
    
# 関数のグラフ表示
def 表示１変数関数(関数リスト,区間=[0,1],分割数=50,枠=[600,600]):
    """ 
       １変数関数達のグラフ表示 
       関数リスト: 関数のリスト
       区間=[0,1]: x が動く範囲
       分割数=50: 区間の等分点の数
       枠=[600,600]: 描画領域の横と縦
    """ 
    width,height=枠
    fig = go.Figure()
    tmin,tmax=区間
    ts=np.linspace(tmin,tmax,分割数)
    for i in range(len(関数リスト)):
        fn=関数リスト[i]
        fig.add_trace(go.Scatter(x=ts, y=[fn(t) for t in ts], name=fn.__name__))
    if len(関数リスト)==1:
        fig.update_layout(title=関数リスト[0].__name__)
    fig.update_layout(width=width,height=height)
    return fig

def 表示２変数関数(関数,x区間=[-1,1],y区間=[-1,1],分割数=50,透明度=0.5,枠=[600,600]):
    """ ２変数関数のグラフ 
        x区間=[0,1]: x の動く範囲
        y区間=[0,1]: y の動く範囲
        分割数=50: 分割数だけ等分して描画
        透明度=0.5: 透明度
        枠 = [600,600]: 描画する領域の横・縦
    """
    width,height=枠
    xmin,xmax=x区間
    ymin,ymax=y区間
    a0=np.linspace(xmin,xmax,分割数)
    b0=np.linspace(ymin,ymax,分割数)
    xx,yy=np.meshgrid(a0,b0)
    F=go.Figure()
    F.add_surface(x=xx,y=yy,z=関数(xx,yy),opacity=透明度)  
    F.update_layout(width=width,height=height)
    return F

def 表示２変数関数等高線(関数,範囲=[[0,1],[0,1]],分割数=20,値域=[0,1],間隔=1.0,枠=[600,600]):

    """ ２変数関数 関数(x,y) の等高線表示 
        範囲 = [ 変数xの区間, 変数yの区間] (= [[0,1], [0,1]]) 
        分割数=20: 描画時の分割数
        値域 = [fmin, fmax] ( = [0., 1.] )  fmin <= 関数値 <= fmax の範囲を描く
        間隔 =1. 等高線の値の間隔  
        枠 = [600,600]: 描画する領域の横・縦
    """
    xrange,yrange=範囲
    start,end=値域
    width,height=枠
    xmin,xmax=xrange
    ymin,ymax=yrange
    T=分割数
    a0=np.linspace(xmin,xmax,T)
    b0=np.linspace(ymin,ymax,T)
    x,y=np.meshgrid(a0,b0)
    F=go.Figure()
    F.add_contour(x=a0,y=b0,z=f(x,y),colorscale="oranges",
        contours=dict(
            start=start, #2 等高線の高さの最小値
            end=end,    #3 等高線の高さの最大値
            size=間隔, #4 描く等高線の高さの間隔
            coloring='lines',
            showlabels=True
        ))
    F.update_yaxes(scaleanchor="x", scaleratio=1)
    F.update_layout(width=600,height=600)
    return F

# ランダムデータの作成
def 生成１次元データ(平均=0,分散=1,個数=100):
    """ 
        ランダムな実数を生成
        分布は、与えられた平均(=0) と分散(=1) の正規分布
        個数=100
    """
    return np.random.normal(loc=平均,scale=分散,size=個数)

def 生成２次元データ1(重心=[0,0],分散=[[0.2, 0], [0, 0.2]],個数=100):
    """ 正規分布に従う二次元のランダムデータを作成
        重心 ( =[0,0] ) 
        分散 =[[0.2,0],[0,0.2]]: 共分散行列
        個数 ( = 100) : データの個数
    """
    return np.random.multivariate_normal(重心, 分散,個数)

def 生成２次元データ(重心リスト,分散=[[0.2, 0], [0, 0.2]],個数=100):
    """ 重心リストにある点を重心とし、共分散が分散である個数だけのランダムな点を合わせたもの """
    return np.vstack([生成２次元データ1(重心,分散=分散,個数=個数) for 重心 in 重心リスト])


def readcsv(url):
    """ url にあるcsvファイルを読み込む"""
    import pandas as pd
    return pd.read_csv(url).to_numpy()
                    
