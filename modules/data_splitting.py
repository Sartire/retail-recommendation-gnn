import numpy as np
import pandas as pd

from datetime import datetime
import torch


'''
Manage test/train/validation splits and negative sampling
'''

def get_preferred_negatives(events_subset):
    '''
    Find cases where people who did make a purcahse at some point viewed an item and did not buy it.
    '''
    transaction_events = events_subset[events_subset['event'] == 'transaction']
    transactional_visitors = transaction_events['user_idx'].unique()

    view_events_df = events_subset[events_subset['event'] == 'view'][['user_idx', 'item_idx', 'timestamp']]
    view_events_df = view_events_df.drop_duplicates(subset = ['user_idx', 'item_idx'], keep = 'last')
    #print(view_events_df.head(1))
    view_events_by_transactional_visitors = view_events_df[view_events_df['user_idx'].isin(transactional_visitors)]
    addtocart_events_df = events_subset[events_subset['event'] == 'addtocart'][['user_idx', 'item_idx']].drop_duplicates()
    merged_df = view_events_by_transactional_visitors.merge(
        addtocart_events_df.assign(is_addtocart=True),
        on=['user_idx', 'item_idx'],
        how='left',
        indicator=True
    )

    view_but_no_addtocart_events = merged_df[merged_df['_merge'] == 'left_only'][['user_idx', 'item_idx','timestamp']]
    # adjust the timestamp of the last view event forward
    view_but_no_addtocart_events['timestamp'] = view_but_no_addtocart_events['timestamp'] + 10
    return view_but_no_addtocart_events

def random_negative_edges(fake_timestamps, positive_sample, events_subset):

    user_set = events_subset['user_idx'].unique()
    item_set = events_subset['item_idx'].unique()

    srcs = []
    dsts = []
    added_count = 0
    for timestamp in fake_timestamps:
      # get the valid user and item lists
      valid_events = events_subset[events_subset['timestamp'] < timestamp]
      user_set = valid_events['user_idx'].unique()
      item_set = valid_events['item_idx'].unique()
      added_count += 1
      while len(srcs) < added_count:
        # try random set members until we can add one, checking against the known positives
          u = np.random.choice(user_set)
          i = np.random.choice(item_set)
          if positive_sample.query("user_idx == @u and item_idx == @i").empty:
              srcs.append(u)
              dsts.append(i)

    return pd.DataFrame({'user_idx': srcs, 'item_idx': dsts, 'timestamp':fake_timestamps})


def generate_negative_sample(positive_sample, events_subset, neg_to_pos_ratio = 1):

    target_count = int(np.floor(positive_sample.shape[0] * neg_to_pos_ratio))
    negative_starter = get_preferred_negatives(events_subset)
    #print(negative_starter.shape)


    if target_count <= negative_starter.shape[0]:
      negative_sample = negative_starter.sample(target_count)
      return negative_sample

    num_additional_negatives = target_count - negative_starter.shape[0]
    print(f'Generating {num_additional_negatives} negative edges')

    min_time = positive_sample['timestamp'].min()
    max_time = positive_sample['timestamp'].max()
    diff_time = max_time - min_time

    fake_timestamps = np.random.uniform(0, 1, num_additional_negatives)
    fake_timestamps = np.floor(fake_timestamps * diff_time + min_time)
    additional_negatives = random_negative_edges(fake_timestamps, positive_sample, events_subset)

    negative_sample = pd.concat([negative_starter, additional_negatives], axis=0)
    return negative_sample

def reindex_nodes(events):

    unique_users = events['visitorid'].unique()
    unique_items = events['itemid'].unique()

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {it: idx + len(unique_users) for idx, it in enumerate(unique_items)}

    events['user_idx'] = events['visitorid'].map(user_to_idx)
    events['item_idx'] = events['itemid'].map(item_to_idx)

    return events

def get_split_subset(events: pd.DataFrame,
                     subset_col: str,
                     split_values: list,
                     pos_limit = None,
                     neg_ratio = 1):
   
   '''
   Create the events df and the pos/neg sample for a split given by the set of split_values
   '''
   
   split_events = reindex_nodes(events[events[subset_col].isin(split_values)])
   
   
   pos_sample = split_events.query("event == 'addtocart' or event == 'transaction'")[['user_idx', 'item_idx', 'timestamp']]
   
   if pos_limit is not None and pos_limit < pos_sample.shape[0]:
     pos_sample = pos_sample.sample(pos_limit)

   neg_sample = generate_negative_sample(pos_sample, split_events, neg_ratio)
   
   return split_events, pos_sample, neg_sample

