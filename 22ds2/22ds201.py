def showgraph(flist,labels,xrange=[0,1],K=50):
    import plotly.graph_objects as go
    import numpy as np
    fig = go.Figure()
    tmin,tmax=xrange
    ts=np.linspace(tmin,tmax,K)
    for i in range(len(flist)):
        fig.add_trace(go.Scatter(x=ts, y=[flist[i](t) for t in ts], name=labels[i]))
    if len(flist)==1:
        fig.update_layout(title=labels[0])
    fig.show() 
def showgraph2(f,xrange=[-1,1],yrange=[-1,1],K=50,opacity=0.5):
    import numpy as np
    import plotly.graph_objects as go
    xmin,xmax=xrange
    ymin,ymax=yrange
    a0=np.linspace(xmin,xmax,K)
    b0=np.linspace(ymin,ymax,K)
    xx,yy=np.meshgrid(a0,b0)
    F=go.Figure()
    F.add_surface(x=xx,y=yy,z=f(xx,yy),opacity=opacity)   
    F.show()
def toukousen(f,xrange=[0,1],yrange=[0,1],T=20,start=0,end=0,size=1):
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
            start=start, # 2 等高線の高さの最小値
            end=end,    # 3 等高線の高さの最大値
            size=size, # 4 描く間隔
            coloring='lines',
            showlabels=True
        ))
    F.update_yaxes(scaleanchor="x", scaleratio=1)
    F.show()
def heikin(D):    
    return sum(D)/len(D)
def bunsan(D):
    if len(D)>0:
        m=heikin(D)
        return sum((D-m)**2)
    else:
        return 0
def mode(D):
    D=sorted(D)
    k=len(D)//2
    if len(D)%2==1:
        return D[k]
    else:
        return (D[k-1]+D[k])/2.
def l1bunsan(D):
    if len(D)>0:
        m=mode(D)
        return sum(abs(D-m))
    else:
        return 0