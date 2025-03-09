import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io
import torch.nn.init
from scipy.stats import qmc 
import matplotlib.pyplot as plt
from matplotlib import cm 
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../data/cylinder_nektar_wake.mat')


device = torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
print(device)

data =  scipy.io.loadmat(data_path)
N_train = 5000

X_star=data['X_star'] #Nx2
t_star=data['t'] #Tx1
P_star=data['p_star'] #NxT
U_star=data['U_star'] #Nx2xT

N=X_star.shape[0]
T=t_star.shape[0]

XX = np.tile(X_star[:,0:1], (1,T)) #N x T
YY = np.tile(X_star[:, 1:2], (1,T)) #N x T
TT = np.tile(t_star,(1,N)).T #N x T

UU = U_star[:,0,:] 
VV = U_star[:,1,:]
PP = P_star

x = XX.flatten()[:,None] # NT x 1 (x coordinate)
y = YY.flatten()[:,None] # NT x 1 (y coordinate)
t = TT.flatten()[:,None] # NT x 1 (time)

u = UU.flatten()[:,None] # NT x 1 (x direction velocity)
v = VV.flatten()[:,None] # NT x 1 (y direction velocity)
p = PP.flatten()[:,None] # NT x 1 (pressure)

idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]


class Net(nn.Module) :
    def __init__(self, layerlist, actftn=torch.tanh, initializer=nn.init.xavier_uniform_) :
        super(Net,self).__init__()
        self.actftn = actftn
        self.layers = nn.ModuleList()

        for i in range(len(layerlist)-1):
            self.layers.append(nn.Linear(layerlist[i],layerlist[i+1]))

        self.lambda1 = nn.Parameter(torch.tensor(0.0))
        self.lambda2 = nn.Parameter(torch.tensor(0.0))

    def forward(self,x,y,t) :
        inputs = torch.cat([x,y,t],axis =1 )
        out = inputs
        for layer in self.layers[:-1]:
            out = self.actftn(layer(out))

        output =self.layers[-1](out) 
        return output

net = Net([3,64,64,64,64,64,2],torch.tanh,nn.init.xavier_uniform_) 
net = net.to(device) 
mse_cos_function = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(net.parameters())

def fg(x,y,t,lamb, net):
    output = net(x,y,t) 
    psi = output[:, 0:1]
    p = output[:, 1:2]  
    lambda1 = net.lambda1
    lambda2 = net.lambda2
    
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

    
    f = u_t + lambda1*(u*u_x+v*u_y) +p_x+ -lambda2*(u_xx + u_yy)
    g = v_t + lambda1*(u*v_x+v*v_y) +p_y+ -lambda2*(v_xx + v_yy)
    residual = torch.cat([f, g], dim=1)

    return residual, u ,v, p

losses = []
iterations = 40000
previous_validation_loss=99999999.0
lamb=np.array([0,0])    
pt_lamb = Variable(torch.from_numpy(lamb).float(),requires_grad=True).to(device) 
pt_x = Variable(torch.from_numpy(x_train).float(),requires_grad=True).to(device)
pt_y= Variable(torch.from_numpy(y_train).float(),requires_grad=True).to(device) 
pt_t = Variable(torch.from_numpy(t_train).float(),requires_grad=True).to(device)
pt_u = Variable(torch.from_numpy(u_train).float(),requires_grad=False).to(device)
pt_v = Variable(torch.from_numpy(v_train).float(),requires_grad=False).to(device)
    
for epoch in range(iterations) :
    optimizer.zero_grad()
    all_zeros = np.zeros((N_train,1))
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    

    res,u_pred,v_pred,p_pred = fg(pt_x,pt_y,pt_t,pt_lamb,net)
    mse_u = mse_cos_function(u_pred,pt_u)
    mse_v = mse_cos_function(v_pred,pt_v)
    mse_data = (mse_u +mse_v )/2
 

    mse_res = mse_cos_function(res,pt_all_zeros)
    
    loss =mse_data +  mse_res
    
    loss.backward()
    
    optimizer.step()
    
    with torch.autograd.no_grad():
        print(epoch, "training loss", loss.data)
        print("Lambda values:", net.lambda1.item(), net.lambda2.item())
        loss_value = loss.item()
        losses.append(loss_value)
torch.save(net.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters_checkpoints/Model_parameters.pth"))


plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.title("Training Loss (Log Scale)")
plt.show()



data = scipy.io.loadmat(data_path)
X_star = data['X_star']  # Nx2
t_star = data['t']  # Tx1
P_star = data['p_star']  # NxT

t0_idx = np.argmin(np.abs(t_star))
print(f"t value at index {t0_idx}: {t_star[t0_idx][0]}")

p_exact_t0 = P_star[:, t0_idx]

N = X_star.shape[0]
x = X_star[:, 0]
y = X_star[:, 1]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_pressure(x_points, y_points, t_value, model):
    x_tensor = torch.tensor(x_points.reshape(-1, 1), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_points.reshape(-1, 1), dtype=torch.float32).to(device)
    t_tensor = torch.tensor(np.ones_like(x_points.reshape(-1, 1)) * t_value, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(x_tensor, y_tensor, t_tensor)
        p_pred = outputs[:, 1:2].cpu().numpy()
    
    return p_pred.reshape(x_points.shape)

t0_value = t_star[t0_idx][0]
p_pred_t0 = predict_pressure(x, y, t0_value, net)


unique_x = np.unique(x)
unique_y = np.unique(y)
X_grid, Y_grid = np.meshgrid(unique_x, unique_y)

P_pred_grid = np.zeros_like(X_grid)
P_exact_grid = np.zeros_like(X_grid)

for i in range(N):
    x_idx = np.where(unique_x == x[i])[0][0]
    y_idx = np.where(unique_y == y[i])[0][0]
    P_pred_grid[y_idx, x_idx] = p_pred_t0[i]
    P_exact_grid[y_idx, x_idx] = p_exact_t0[i]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.pcolormesh(X_grid, Y_grid, P_pred_grid, cmap='jet', shading='auto')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Predicted pressure (t=0)')
fig.colorbar(im1, ax=ax1)


im2 = ax2.pcolormesh(X_grid, Y_grid, P_exact_grid, cmap='jet', shading='auto')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Exact pressure (t=0)')
fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures/pressure_comparison_t0.png'), dpi=300, bbox_inches='tight')
plt.show()