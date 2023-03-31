import numpy as np
import plotly.express as px

## 名称
## dkplot (k=1,2,3) k次元データ集合の表示 あるいは k次元データ表示
## fkplot  (k=1,2) k変数関数のグラフ表示  k変数関数表示

# 実数集合Dの表示
def d1plot(D,width=600,height=300):
    """ 
        実数の集合Dを平面に表示
        x軸にDの値、y軸に0をセット
        width=600: 幅
        height=300: 高さ
    """
    import plotly.express as px
    import numpy as np
    fig=px.scatter(x=D,y=np.zeros(len(D)))
    fig.update_layout(width=width,height=height)
    return fig  
def 表示１次元データ(D,枠=[600,200]):
    """ 
        実数の集合Dを平面に表示
        x軸にDの値、y軸に0をセット
        枠=[600,200]: 描画領域の横幅と高さ
    """
    width,height=枠
    return d1plot(D,width=width,height=height)

    
# １変数関数のグラフ表示, flist は表示する関数のリスト、labels はそれぞれのラベル, xrange はxの動く範囲
def f1plot(flist,labels,xrange=[0,1],K=50,width=600,height=600):
    """ 
       １変数関数のグラフ表示 
       flist: 関数のリスト
       labels: 各関数の名称
       xrange=[0,1]: x が動く範囲
       K=50: xrangeを K個に等分して描画
       width=600: 幅 
       height=600: 高さ 
    """
    import plotly.graph_objects as go
    import numpy as np
    fig = go.Figure()
    tmin,tmax=xrange
    ts=np.linspace(tmin,tmax,K)
    for i in range(len(flist)):
        fig.add_trace(go.Scatter(x=ts, y=[flist[i](t) for t in ts], name=labels[i]))
    if len(flist)==1:
        fig.update_layout(title=labels[0])
    fig.update_layout(width=width,height=height)
    return fig
def 表示１変数関数(関数リスト,関数名リスト,区間=[0,1],分割数=50,枠=[600,600]):
    """ 
       １変数関数達のグラフ表示 
       関数リスト: 関数のリスト
       関数名リスト: 各関数の名称
       区間=[0,1]: x が動く範囲
       分割数=50: 区間の等分点の数
       枠=[600,600]: 描画領域の横と縦
    """ 
    width,height=枠
    return f1plot(関数リスト,関数名リスト,xrange=区間,K=分割数,width=width,height=height)    

# ２変数関数fのグラフを表示
def f2plot(f,xrange=[-1,1],yrange=[-1,1],T=50,opacity=0.5,width=600,height=600):
    """ ２変数関数のグラフ 
        xrange=[0,1]: x の動く範囲
        yrange=[0,1]: y の動く範囲
        T=50: T等分して描画
        opacithy=0.5: 透明度
    """
    import numpy as np
    import plotly.graph_objects as go
    xmin,xmax=xrange
    ymin,ymax=yrange
    a0=np.linspace(xmin,xmax,T)
    b0=np.linspace(ymin,ymax,T)
    xx,yy=np.meshgrid(a0,b0)
    F=go.Figure()
    F.add_surface(x=xx,y=yy,z=f(xx,yy),opacity=opacity)  
    F.update_layout(width=width,height=height)
    return F
def 表示２変数関数(関数,x区間=[-1,1],y区間=[-1,1],分割数=50,透明度=0.5,枠=[600,600]):
    """ ２変数関数のグラフ 
        x区間=[0,1]: x の動く範囲
        y区間=[0,1]: y の動く範囲
        分割数=50: 分割数だけ等分して描画
        透明度=0.5: 透明度
        枠型 = [600,600]: 描画する領域の横・縦
    """
    width,height=枠
    return f2plot(関数,xrange=x区間,yrange=y区間,T=分割数,opacity=透明度,width=width,height=height)

def toukousen(f,xrange=[0,1],yrange=[0,1],T=20,start=0,end=1,size=1):
    """ ２変数関数 f(x,y) の等高線表示 
        xrange=[0,1]: x の動く範囲
        yrange=[0,1]: y の動く範囲
        T=20: T等分して描画
        start=0,end=1: start <= f <= end の範囲を描く
        size=1 等高線の本数
    """
    import numpy as np
    import plotly.graph_objects as go
    xmin,xmax=xrange
    ymin,ymax=yrange
    a0=np.linspace(xmin,xmax,T)
    b0=np.linspace(ymin,ymax,T)
    x,y=np.meshgrid(a0,b0)
    F=go.Figure()
    F.add_contour(x=a0,y=b0,z=f(x,y),colorscale="oranges",
        contours=dict(
            start=start, #2 等高線の高さの最小値
            end=end,    #3 等高線の高さの最大値
            size=size, #4 描く間隔
            coloring='lines',
            showlabels=True
        ))
    F.update_yaxes(scaleanchor="x", scaleratio=1)
    F.update_layout(width=600,height=600)
    return F
def 表示２変数関数等高線(関数,範囲=[[0,1],[0,1]],分割数=20,値域=[0,1],個数=1):
    xrange,yrange=範囲
    start,end=値域
    return toukousen(関数,xrange=xrange,yrange=yrange,T=分割数,start=start,end=end,size=個数)

def make2data(center=[0,0],cov=[[0.2, 0], [0, 0.2]],N=100): 
    """ 正規分布に従う二次元のランダム,データを作成
        center=[0,0]:重心 
        cov=[[0.2,0],[0,0.2]]: 共分散行列
        N=100: データの個数
    """
    import numpy as np
    return np.random.multivariate_normal(center, cov, N)
    D21=np.vstack([np.random.multivariate_normal([2*i,2*j], cov, N) for i in range(2) for j in range(2)])
    return D21

def 生成２次元データ(重心=[0,0],分散=[[0.2, 0], [0, 0.2]],個数=100):
    """ 二次元ベクトルを個数だけランダムに生成. 
        重心が"重心", 共分散行列が "分散"となる正規分布に従って"""
    return make2data(center=重心,cov=分散,N=個数)

def 生成２次元データA(重心リスト,分散=[[0.2, 0], [0, 0.2]],個数=100):
    """ 重心リストにある点を重心とし、共分散が分散である個数だけのランダムな点を合わせたもの """
    return np.vstack([データ生成２次元(重心,分散=分散,個数=個数) for 重心 in 重心リスト])

def make1data(heikin=0,bunsan=1,N=100):
    """ 
        正規分布に従うランダムな１次元データを作成
        heikin=0: 平均値
        bunsan=1: 分散
        N=100: 個数
    """
    return np.random.normal(loc=heikin,scale=bunsan,size=N)

def 生成１次元データ(平均=0,分散=1,個数=100):
    """ 
        ランダムな実数を生成
        分布は、与えられた平均(=0) と分散(=1) の正規分布
        個数=100
    """
    return np.random.normal(loc=平均,scale=分散,size=個数)

def 表示１次元データ2(データリスト,ラベルリスト=[],枠=[600,500],毛=False,ヒストグラム=False,密度=False,ビンサイズ=1.0):
    """ １次元データリストを複数表示
        データにはラベルがついている
        毛 (=False): 線分で表示するにはTrue
        ヒストグラム (=False) : ヒストグラムで表示するときは True
        密度(=False) : 密度関数で表示するときは True
        ビンサイズ=1.0 : ヒストグラムのビンの大きさ
    """
    import plotly.figure_factory as ff
    fig= ff.create_distplot(データリスト,ラベルリスト,show_rug=毛, show_hist=ヒストグラム,show_curve=密度,bin_size=ビンサイズ)
    width,height=枠
    fig.update_layout(width=width,height=height)
    return fig

def plot2d(D,color=[],width=600,height=600,透明度=1.0):
    import plotly.express as px
    import numpy as np
    if len(color)==0:
        color=np.zeros(len(D))
    fig=px.scatter(x=D[:,0],y=D[:,1],opacity=透明度,color=color)
    fig.update_layout(width=width,height=height)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
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
    return plot2d(D=D,color=色リスト,width=width,height=height,透明度=透明度)


def 表示２次元データヒストグラム(データリスト,xビン数=10,yビン数=10,周辺分布=False,枠=[600,600]):
    """ ２次元データのヒストグラム表示
        データリスト: ２次元データリスト [[1,2],[2,3],...]
        xビン数=10: x方向の分割数
        yビン数=10: y方向の分割数
        周辺分布=True の時、周辺分布のヒストグラムも表示
        枠=[600,600]: 表示枠の横と縦 
    """
    import plotly.express as px
    import numpy as np
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
    import numpy as np
    import plotly.express as px
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

def readcsv(url):
    """ url にあるcsvファイルを読み込む"""
    import pandas as pd
    return pd.read_csv(url).to_numpy()

def plot3d1(D,color=[],透明度=0.8,点サイズ=1,形状="circle",width=500,height=500):
    import plotly.graph_objects as go
    import plotly.express as px
    fig=px.scatter()
    fig.add_trace(go.Scatter3d(x=D.T[0],y=D.T[1],z=D.T[2],mode="markers",\
                               marker=dict(size=点サイズ, symbol=形状, color=color,opacity=透明度)))
    fig.update_layout(width=width,height=height)
    fig.show()
                    
def 表示３次元データ(D,Dcolor=[],点サイズ=2,透明度=0.6,形状="circle",枠=[500,500]):
    import plotly.graph_objects as go
    import plotly.express as px
    fig=px.scatter()
    D=np.array(D)
    if len(Dcolor)==0:
        Dcolor=["blue" for i in range(len(D))]
    fig.add_trace(go.Scatter3d(x=D[:,0],y=D[:,1],z=D[:,2],mode="markers",\
                               marker=dict(size=点サイズ, symbol=形状, color=Dcolor),opacity=透明度))
    width,height=枠
    fig.update_layout(width=width,height=height)    
    return fig        
                  

def plot3d(D,color=[],透明度=0.8,点サイズ=0.5,width=500,height=500):
    import numpy as np
    import plotly.express as px
    if len(color)==0:
        color=np.ones(len(D))
    D1=np.vstack([D.T,点サイズ*0.1*np.ones(len(D))]).T
    D2=np.vstack([D1,[0,0,0,0.6]])
    Dcolor=np.hstack([color,"white"])

    fig=px.scatter_3d(D2,x=0,y=1,z=2,color=Dcolor,opacity=透明度,size=3)
    fig.update_layout(width=600,height=600)
    print(" 'white' をクリックして大きな球を消去してください")
    fig.show()    


def DAyame():
    import plotly.express as px
    temp = px.data.iris().to_numpy()
    return temp[:,[0,1,2,3,5]].astype("float32")

def sc3d(a,b,c):
    import plotly.express as px
    D45=np.vstack([D44.T,0.1*np.ones(150)]).T
    D45A=np.vstack([D45,[0,0,0,0,0.6]])
    D4color_name_1=np.hstack([D4color_name,"white"])
    fig=px.scatter_3d(D45A,x=a,y=b,z=c,color=D4color_name_1,opacity=0.6,size=4)
    fig.update_layout(width=600,height=600)
    fig.show()    

def proj3d():
    from ipywidgets import interact
    import ipywidgets as widgets
    interact(sc3d  
        ,a=widgets.RadioButtons(description="第1変数",value=0,options=[0,1,2,3])
        ,b=widgets.RadioButtons(description="第２変数",value=1,options=[0,1,2,3])
        ,c=widgets.RadioButtons(description="第３変数",value=2,options=[0,1,2,3])
            )    
def バブル表示(D,透明度=0.5):
    fig=px.scatter(x=D.T[0],y=D.T[1],size=100*np.abs(D.T[2]),opacity=透明度)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(width=500,height=500)
    fig.show()

def normalgraph(D):    
    fig1=px.line(D,x=range(12),y=[0,1])
    fig1.update_layout(width=500,height=500)
    fig1.show() 

def climograph(D,label=[]):
    import plotly.express as px
    if len(label)==0:
        label=np.arange(len(D))
    fig=px.line(D,x=0,y=1,text=label)
    fig.update_layout(width=500,height=500)
    fig.show()

def 関数表示(D,color=[]):
    import numpy as np
    import plotly.express as px
    if(len(color)==0):
        color=np.zeros(len(D[0]))
    fig = px.parallel_coordinates(D, color=color)
    fig.show()

def 二次元散布図(D,color=[],透明度=0.7,width=1000,height=1000,seaborn=False):
    import pandas as pd
    N=len(D)
    if len(color)==0:
        color=np.zeros(len(D))
    if seaborn:
        import seaborn as sbn
        Df=pd.DataFrame(D)
        Df["color"]=color
        sbn.pairplot(Df,kind="kde",hue="color")        
    else:
        fig=px.scatter_matrix(pd.DataFrame(D),opacity=透明度,color=color)
        fig.update_layout(width=width,height=height)
        fig.show()

## 統計関数
def heikin(D):  
    """ リストDの平均値"""  
    if len(D)>0:
        return sum(D)/len(D)
    else:
        return None
    
def bunsan(D):
    """ 分散 """
    if len(D)>0:
        m=heikin(D)
        return sum((D-m)**2)/len(D)
    else:
        return 0
    
def median(D):
    """ 中央値 """
    D=sorted(D)
    k=len(D)//2
    if len(D)==0:
        return None
    elif len(D)%2==1:
        return D[k]
    else:
        return (D[k-1]+D[k])/2.
    
def l1bunsan(D):
    """ 中央値との差の絶対値の和 """
    if len(D)>0:
        m=median(D)
        return sum(abs(D-m))
    else:
        return 0