import torch
import visdom

vis = visdom.Visdom(env='test1')
x = torch.arange(1, 30, 0.01)
y = torch.sin(x)
vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})

# append 追加数据
for ii in range(0, 10):
    # y = x
    x = torch.Tensor([ii])
    y = x
    vis.line(X=x, Y=y, win='polynomial', update='append')
