import lasio as ls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from matplotlib.cm import get_cmap
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score 
from sklearn.cluster import KMeans

from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import copy
import random

# set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # Fixed typo
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,z):
        h = F.relu(self.fc1(z))
        logits = self.fc2(h)
        return logits
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_classes)

    def forward(self,x):
        mu,logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
    def loss_function(self, logits, target, mu, logvar, beta=1.0):
        class_loss = F.cross_entropy(logits, target, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return class_loss + beta * kld_loss

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#assigning facies colors
facies_color = [
     "#E74C3C",   #red
     "#F27B44",  # orange
     "#F1C40F",  # yellow
     "#A2A2A2",  # grey
     ]


def create_log_file(raw_input, processed_input, zone_min, zone_max):
    
    #columns to input
    column_input = ['DEPTH', 'GR', 'LLD', 'NPHI', 'RHOB', 'PERM', 'SWE', 'VSH', 'PHIE', 'PRT'] 
    
    #raw logs
    raw_logs = ls.read(raw_input)
    raw_logs_df = raw_logs.df()
    raw_logs_df = raw_logs_df.reset_index()
    
    #processed logs
    processed_logs = ls.read(processed_input)
    processed_logs_df = processed_logs.df()
    processed_logs_df = processed_logs_df.reset_index()
    
    #mergig the dataset of processed and raw logs into one dataframe
    #well_df = pd.merge(raw_logs_df, processed_logs_df, on='DEPTH', how='outer')
    
    df_merged = pd.merge(raw_logs_df, processed_logs_df, how='outer', on='DEPTH')
    df_merged['DEPTH']= np.round(df_merged['DEPTH'], decimals=4)

    # Group by 'Depth' and merge the rows within each group
    well_df = df_merged.groupby('DEPTH', as_index=False).apply(lambda group: group.ffill(axis=0).bfill(axis=0).iloc[0])
    #well_df = processed_logs_df
    well_df['PRT'] -= 1   # to ensure count starts from 0 {0,1,2} classes
    
    rename_pairs = {'LLDC':'LLD', 'LLSC':'LLS'}
    
    well_df.rename(columns=rename_pairs, inplace=True)
    
    #subset via zone depths
    df_zone = well_df[(well_df['DEPTH']>=zone_min) & (well_df['DEPTH']<=zone_max)]
    df_zone = df_zone[column_input]
    
    # removing all rows with nan values
    df_zone = df_zone.dropna()
    
    return df_zone

def create_log_file_sp(raw_input, processed_input, zone_min, zone_max):  # Fixed signature
    
    #columns to input
    column_input = ['DEPTH','SWE','VSH','PHIE','PRT'] 
    
    #raw logs
    raw_logs = ls.read(raw_input)
    raw_logs_df = raw_logs.df()
    raw_logs_df = raw_logs_df.reset_index()
    
    #processed logs
    processed_logs = ls.read(processed_input)
    processed_logs_df = processed_logs.df()
    processed_logs_df = processed_logs_df.reset_index()
    
    #merging the dataset of processed and raw logs into one dataframe
    well_df = pd.merge(raw_logs_df, processed_logs_df, on='DEPTH', how='outer')
    
    #subset via zone depths
    df_zone = well_df[(well_df['DEPTH']>=zone_min) & (well_df['DEPTH']<=zone_max)]
    df_zone = df_zone[column_input]
    
    # removing all rows with nan values
    df_zone_sp = df_zone.dropna()
    
    return df_zone_sp

def create_plot(dataframe, well_log_plot, depth_curve, logarithmic_log=[], facies_curves=[]):
    # Count the tracks we need
    num_tracks = len(well_log_plot)
    
    # Setting up figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks * 2, 10))
    
    # Looping through each log and create a track with the data
    for i, curve in enumerate(well_log_plot):
        if curve in facies_curves:
            # Map unique facies values to colors
            unique_facies = np.unique(dataframe[curve].dropna())
            # Get the Set1 colormap with enough discrete colors
            #cmap_obj = get_cmap('Set1', len(unique_facies))  # e.g., Set1 for 9 distinct colors
            #cmap_facies = ListedColormap(cmap_obj.colors[:len(unique_facies)], name='Set1_facies')
            cmap_facies = ListedColormap(facies_color[:len(unique_facies)], 'indexed')
            
            cluster = np.repeat(np.expand_dims(dataframe[curve].values, 1), 100, 1)
            im = ax[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',
                              vmin=dataframe[curve].min(), vmax=dataframe[curve].max(),
                              extent=[0, 20, depth_curve.max(), depth_curve.min()])
            # Adding a color bar
            cbar = plt.colorbar(im, ax=ax[i], orientation='vertical')
            cbar.set_ticks(range(1, len(unique_facies)+1))
            cbar.set_ticklabels(unique_facies)
            
        else:
            ax[i].plot(dataframe[curve], depth_curve)
        
        # Setting up titles and cosmetics
        ax[i].set_title(curve, fontsize=14)
        ax[i].grid(which='major', color='lightgrey', linestyle='-')
        
        # Displaying shallow to deepest
        ax[i].set_ylim(depth_curve.max(), depth_curve.min())
        
        # Only set y label for the first track, hide it for others
        if i == 0:
            ax[i].set_ylabel('Depth (m)', fontsize='18')
        else:
            plt.setp(ax[i].get_yticklabels(), visible=False)
        
        # Check if there's any logarithmic curve
        if curve in logarithmic_log:
            ax[i].set_xscale('log')
            ax[i].grid(which='minor', color='lightgrey', linestyle='-')
    
    plt.tight_layout()
    plt.show()
    
    return cmap_facies

#metrics
def calculate_metrics(y_actual, y_predicted):
    #confusion matrix
    cm = confusion_matrix(y_actual, y_predicted)
    print('confusion matrix')
    print(cm)
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Coarse Sand', 'Fine Sand', 'Shale'], yticklabels=['Coarse Sand', 'Fine Sand', 'Shale'])
    plt.xlabel('Predicted Facies')
    plt.ylabel('True Facies')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 4. Accuracy
    accuracy = accuracy_score(y_actual, y_predicted)
    print(f"Accuracy: {accuracy:.4f}")
    
    # 5. Precision (for binary or multi-class classification)
    # For multi-class, you can set the `average` parameter to 'macro', 'micro', or 'weighted' as needed.
    precision = precision_score(y_actual, y_predicted, average='weighted')  # 'micro', 'macro', or 'weighted'
    print(f"Precision: {precision:.4f}")
    
    # 6. F1 Score (for binary or multi-class classification)
    f1 = f1_score(y_actual, y_predicted, average='weighted')  # 'micro', 'macro', or 'weighted'
    print(f"F1 Score: {f1:.4f}")

# CONFUSION MATRIX
def best_map(y_true, y_pred):
    D = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-D)
    mapping = dict(zip(col_ind, row_ind))
    return np.array([mapping[y] for y in y_pred])

#loading D3 well
raw_input = 'file/to/raw.las'
processed_input = 'file/to/processed.las'
zone_min, zone_max = 2000, 3000  # Example depth range for the zone
well = create_log_file_sp(raw_input, processed_input, zone_min, zone_max)  # Fixed call

X = well[['GR','NPHI', 'LLD', 'VSH', 'PHIE', 'SWE']]  # Fixed missing comma
y = well['PRT']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_array = np.array(y)

# Creating the dataset
dataset = TabularDataset(X_scaled, y_array)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model parameters
input_dim = len(X_scaled[0])
hidden_dim = 128
latent_dim = 32
num_classes = len(np.unique(y_array))

# Initialize the VAE model
model = VAE(input_dim, hidden_dim, latent_dim, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training the model
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits, mu, logvar = model(x_batch)
        
        # Compute loss
        loss = model.loss_function(logits, y_batch, mu, logvar)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Evaluation
model.eval()
all_mu = []

with torch.no_grad():
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits, mu, logvar = model(x_batch)
        all_mu.append(mu.cpu().numpy())

mu_array = np.vstack(all_mu)

#KMeans clustering
kmeans = KMeans(n_clusters=num_classes, random_state=seed)
y_clusters = kmeans.fit_predict(mu_array)

# Map the clusters to the original labels
plt.figure(figsize=(10, 6))
plt.scatter(mu_array[:,0], mu_array[:,1], c=y_clusters, cmap='Set1', alpha=0.5)
plt.title('Clustering of Latent Space (Predicted Labels)')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar(ticks=range(num_classes), label='Predicted Facies')
plt.show()

#crossplot with original labels
plt.figure(figsize=(10, 6))
plt.scatter(X['PHIE'], X['VSH'], c=y_array, cmap='Set1', alpha=0.5)  # Fixed typo
plt.title('Crossplot of PHIE vs VSH (Original Labels)')
plt.xlabel('PHIE')
plt.ylabel('VSH')
plt.colorbar(ticks=range(num_classes), label='Original Facies')
plt.show()

# Calculate metrics
y_predicted = best_map(y_array, y_clusters)
calculate_metrics(y_array, y_predicted)

well['Predicted_Facies'] = y_predicted

#plotting the well logs
# Define parameters
well_log_plot = ['GR', 'LLD', 'NPHI', 'VSH','PHIE', 'SWE','Predicted_Facies']
logarithmic_log = []
facies_curve = ['Predicted_Facies']

# Create the first plot to generate the colormap
cmap_facies = create_plot(well, well_log_plot, well['DEPTH'], logarithmic_log, facies_curve)