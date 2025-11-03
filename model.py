# model.py
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv, GATConv

# NOTE:
# - Pass num_relations=5 now that you have: mentioned(0), replied_to(1),
#   co_mention(2), co_reply(3), same_conversation(4).
# - If you load from a checkpoint, the loader in test/train already reads that
#   number from the saved config; this file just needs to accept it.

# -----------------------------
# Main multi-feature RGCN model
# -----------------------------
class BotRGCN(nn.Module):
    def __init__(
        self,
        des_size=768,
        tweet_size=768,
        num_relations=None,
        num_rels=None,                  # alias for older checkpoints
        num_prop_size=6,
        cat_prop_size=11,
        embedding_dimension=128,
        dropout=0.3,
    ):
        super().__init__()
        if num_relations is None and num_rels is not None:
            num_relations = num_rels
        # Default to 5 (your current relation set). Safe even if some relations are unused.
        # Also supports original's 2-relation setup for backward compatibility
        self.num_relations = int(num_relations) if num_relations is not None else 5

        self.dropout = dropout

        # Store dimensions for debugging/inspection
        self.des_size = des_size
        self.tweet_size = tweet_size
        self.num_prop_size = num_prop_size
        self.cat_prop_size = cat_prop_size
        self.embedding_dimension = embedding_dimension

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


# -----------------------------
# Variants that use a subset of features
# (kept functionally identical to your originals, but fixed num_relations handling)
# -----------------------------
class BotRGCN1(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        x = d
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


# =================== BOTRGCN_Original Architecture Replica ===================

class BotRGCN_Original(nn.Module):
    """
    Exact replica of BOTRGCN_Original model architecture for compatibility testing.
    Uses hardcoded parameters matching the original implementation.
    """
    def __init__(self, cat_prop_size=3, embedding_dimension=32, num_relations=2):
        super(BotRGCN_Original, self).__init__()
        self.dropout = 0.1

        # Original architecture: 4 feature types with embedding_dimension/4 each
        self.linear_relu_des = nn.Sequential(
            nn.Linear(768, int(embedding_dimension/4)),  # des_size=768 (fixed)
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(768, int(embedding_dimension/4)),  # tweet_size=768 (fixed)
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(6, int(embedding_dimension/4)),    # num_prop_size=6 (fixed)
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension/4)),  # cat_prop_size=3 or 11
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        # Original uses 2 relations (following/followers) but supports configurable
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=num_relations)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        # Exactly match original forward pass
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


# =================== Enhanced Model Variants ===================

class BotRGCN2(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        t = self.linear_relu_tweet(tweet)
        x = t
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class BotRGCN3(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        n = self.linear_relu_num_prop(num_prop)
        x = n
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class BotRGCN4(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        c = self.linear_relu_cat_prop(cat_prop)
        x = c
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class BotRGCN12(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        x = torch.cat((d, t), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class BotRGCN34(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type.long())
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


# -----------------------------
# Non-relational baselines (unchanged except tiny type guard)
# -----------------------------
class BotGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768,
                 num_prop_size=6, cat_prop_size=11,
                 embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.gcn1 = GCNConv(embedding_dimension, embedding_dimension)
        self.gcn2 = GCNConv(embedding_dimension, embedding_dimension)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class BotGAT(nn.Module):
    def __init__(self, des_size=768, tweet_size=768,
                 num_prop_size=6, cat_prop_size=11,
                 embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.gat1 = GATConv(embedding_dimension, int(embedding_dimension / 4), heads=4)
        self.gat2 = GATConv(embedding_dimension, embedding_dimension)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


# -----------------------------
# Deeper RGCN stacks (kept, with fixed num_relations handling)
# -----------------------------
class BotRGCN_4layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        for _ in range(4):
            x = self.rgcn(x, edge_index, edge_type.long())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class BotRGCN_8layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_relations=None,
                 num_prop_size=6, cat_prop_size=11, embedding_dimension=128, dropout=0.3):
        super().__init__()
        self.num_relations = int(num_relations) if num_relations is not None else 5
        self.dropout = dropout

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(
            embedding_dimension, embedding_dimension,
            num_relations=self.num_relations
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        for _ in range(8):
            x = self.rgcn(x, edge_index, edge_type.long())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x
