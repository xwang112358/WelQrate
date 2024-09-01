
from tqdm import tqdm
import numpy as np
import torch
from torch_scatter import scatter_add


def train_class(model, loader, optimizer, scheduler, device, loss_fn):
    
    model.train()
    loss_list = []

    for i, batch in enumerate(tqdm(loader, miniters=100)):
        batch.to(device)
        assert batch.edge_index.max() < batch.x.size(0), f"Edge index {batch.edge_index.max()} exceeds number of nodes"
        y_pred = model(batch)
        
        loss= loss_fn(y_pred.view(-1), batch.y.view(-1).float())
            
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss = np.mean(loss_list)
    return loss

# def train_class(model, loader, optimizer, scheduler, device, loss_fn):
#     model.train()
#     loss_list = []

#     for i, batch in enumerate(tqdm(loader, miniters=100)):
#         batch = batch.to(device)

#         # Check for isolated nodes
#         num_nodes = batch.num_nodes
#         connection_counts = torch.zeros(num_nodes, dtype=torch.int32, device=device)
#         ones = torch.ones_like(batch.edge_index[0], dtype=torch.int32)

#         connection_counts = scatter_add(ones, batch.edge_index[0], dim=0, dim_size=num_nodes)
#         connection_counts += scatter_add(ones, batch.edge_index[1], dim=0, dim_size=num_nodes)

#         # Add self-loops to isolated nodes
#         isolated_nodes = torch.where(connection_counts == 0)[0]
#         if len(isolated_nodes) > 0:
#             self_loops = torch.stack([isolated_nodes, isolated_nodes], dim=0)
#             batch.edge_index = torch.cat([batch.edge_index, self_loops], dim=1)

#             # Update node features if necessary (e.g., for attention mechanisms)
#             if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
#                 # Create self-loop edge attributes (you may need to adjust this based on your edge attributes)
#                 self_loop_attr = torch.zeros(len(isolated_nodes), batch.edge_attr.size(1), device=device)
#                 batch.edge_attr = torch.cat([batch.edge_attr, self_loop_attr], dim=0)


#         y_pred = model(batch)
#         loss = loss_fn(y_pred.view(-1), batch.y.view(-1).float())
        
#         loss_list.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#     loss = np.mean(loss_list) 
#     return loss


def train_reg(model, loader, optimizer, scheduler, device, loss_fn):
    
    model.train()
    loss_list = []

    for i, batch in enumerate(tqdm(loader, miniters=100)):
        batch.to(device)
        # assert batch.edge_index.max() < batch.x.size(0), f"Edge index {batch.edge_index.max()} exceeds number of nodes"
        y_pred = model(batch)
        
        loss = loss_fn(y_pred.view(-1), batch.activity_value.view(-1))
            
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss = np.mean(loss_list)
    return loss