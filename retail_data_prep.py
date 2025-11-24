## load and clean the events csv

import os
import kagglehub
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
from torch_geometric.data import Data


def download_events( limit = None):
# download dataset
    temp_sample_size = limit

    path = kagglehub.dataset_download("retailrocket/ecommerce-dataset")
    events_path = os.path.join(path, "events.csv")
    events = pd.read_csv(events_path)
    events = events[['visitorid', 'itemid', 'timestamp', 'event']].dropna()
    events = events.sort_values('timestamp').reset_index(drop=True)
    print("Total events:", len(events))

    if temp_sample_size is not None:
      events = events.sample(n=temp_sample_size, random_state=42).sort_values('timestamp').reset_index(drop=True)
    else:
      events = events.sort_values('timestamp').reset_index(drop=True)

    return events


def apply_activity_threshold(events, min_user_interactions = 5, min_item_interactions = 10):
    '''
    Filter out users and items that have less than min_user_interactions and min_item_interactions
    '''
    user_counts = events['visitorid'].value_counts()
    item_counts = events['itemid'].value_counts()

    MIN_USER_INTERACTIONS = min_user_interactions
    MIN_ITEM_INTERACTIONS = min_item_interactions

    active_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    active_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index

    events_filtered = events[
        events['visitorid'].isin(active_users) &
        events['itemid'].isin(active_items)
    ].copy()

    return events_filtered.reset_index(drop=True)




def unix_to_datetime(timestamp):
    try:
        # If timestamp is in milliseconds, convert to seconds
        if timestamp > 1e12:
            timestamp /= 1000
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object
    except (OSError, OverflowError, ValueError) as e:
        print(f"Invalid timestamp: {e}")
        return None




def preprocess_events(min_user_interactions = 5, min_item_interactions = 10, limit = None):
    events = download_events(limit)
    events = apply_activity_threshold(events, min_user_interactions, min_item_interactions)
    #events = reindex_nodes(events) don't need to do this until splitting 
    events['datetime'] = [unix_to_datetime(timestamp) for timestamp in events['timestamp']]
    events['month'] = events['datetime'].dt.month
    events['date'] = events['datetime'].dt.date
    events = events.sort_values('datetime').reset_index(drop=True)
    return events


def create_graph_features(events, use_edge_weights = True):
    

    event_weight_map = {
        'view': 1.0,
        'addtocart': 2.0,
        'transaction': 3.0
    }

    unique_users = len(events['visitorid'].unique())
    unique_items = len(events['itemid'].unique())
    num_nodes = unique_items + unique_users

    if use_edge_weights:
        events['weight'] = events['event'].map(event_weight_map).fillna(1.0)
    else:
        events['weight'] = 1.0

    edge_src = torch.tensor(events['user_idx'].values, dtype=torch.long)
    edge_dst = torch.tensor(events['item_idx'].values, dtype=torch.long)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_weight = torch.tensor(events['weight'].values, dtype=torch.float)

    deg = torch.zeros(num_nodes, dtype=torch.float)
    #deg.index_add_(0, edge_src, torch.ones_like(edge_src, dtype=torch.float))
    #deg.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=torch.float))

    graph_summary_data = Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=deg.view(-1, 1),
        num_nodes=num_nodes
    )

    return graph_summary_data
