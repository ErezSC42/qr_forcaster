from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from FinticaDataset import FinticaDataset


class DatasetHandler:
    def __init__(self, data_path, num_samples, hist_days, pred_horizon, batch_size, val_split_ratio=0.3, asset_ratio_val=0.3,
                 forking_total_seq_length=None):
        self.data_path = Path(data_path)
        self.num_samples = num_samples
        self.hist_days = hist_days
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.val_split_ratio = val_split_ratio
        self.asset_ratio_val = asset_ratio_val
        self.forking_total_seq_length = forking_total_seq_length
        if forking_total_seq_length is not None:
            assert (self.forking_total_seq_length > self.hist_days + pred_horizon)


    def load_df(self):
        assets = []
        dict_df_features = {}
        for p in self.data_path.iterdir():
            asset = pd.read_csv(p, parse_dates=True, index_col=0)
            if len(asset.columns) > 1: # mean that additional features where added
                target_columns = [c for c in asset.columns if 'target' in c]
                assert len(target_columns) == 1, f'The target name was not clearly indicated in the dataset {p}'
                asset_name = target_columns[0]
                features_columns = [c for c in asset.columns if 'target' not in c]
                assets_features = asset[features_columns]
                dict_df_features[asset_name] = assets_features
            else:
                asset_name = asset.columns[0]
                dict_df_features[asset_name]=None

            if 'd0.' in asset_name:
                assets.append(asset[[asset_name]])    
            else:
                print(f'skipping {asset_name}')
        
        df = pd.concat(assets, axis=1)
        df = df.reset_index()
        df = df.ffill()
        df = df.dropna()
        df = df.rename({"Date": "timestamp"}, axis=1)
        # df.pop("stoxx50_d0.05")
        return df, dict_df_features


    def load_dataset(
        self, df: pd.DataFrame = None, dict_df_features: pd.DataFrame = None,
        split_train_val: bool = True, split_assets: bool = True, num_workers=0
    ):
        if df is None:
            df, dict_df_features = self.load_df()
        if split_train_val:
            (df_target_train, dict_df_feature_train), (df_target_val, dict_df_feature_val) = self.split_df(df, dict_df_features, split_assets=split_assets)
            train_dataset = FinticaDataset(
                df=df_target_train, num_samples=self.num_samples, 
                hist_days=self.hist_days, future_days=self.pred_horizon,
                forking_total_seq_length=self.forking_total_seq_length, 
                dict_features =dict_df_feature_train
            )
            val_dataset = FinticaDataset(
                df=df_target_val, num_samples=None, hist_days=self.hist_days,
                future_days=self.pred_horizon, dict_features=dict_df_feature_val
            )
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=num_workers)
            return train_dataloader, val_dataloader
        # will raise exception
        
        dataset = FinticaDataset(
            df=df, num_samples=self.num_samples, hist_days=self.hist_days,
            future_days=self.pred_horizon, 
            forking_total_seq_length=self.forking_total_seq_length, 
            dict_features=dict_df_features
        )
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)
        val_dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)
        return train_dataloader, val_dataloader

    def split_df(self, df:pd.DataFrame, dict_df_features:dict,split_assets: bool=True):
        num_rows, num_cols = df.shape
        train_size = int(num_rows * (1-self.val_split_ratio))
        if split_assets:
            print("splitting assets!")
            assets = df.columns[1:]
            train_assets, test_assets = train_test_split(assets, test_size=self.asset_ratio_val)
            assert not set(train_assets).intersection(set(test_assets)) # make sure no leakage
            train_assets = train_assets.tolist()
            test_assets = test_assets.tolist()
            dict_features_train = {name_asset: dict_df_features[name_asset] for name_asset in train_assets}
            dict_features_val = {name_asset: dict_df_features[name_asset] for name_asset in test_assets}
            
            train_assets.insert(0, "timestamp")
            test_assets.insert(0, "timestamp")
            df_target_train = df[train_assets]
            df_target_test = df[test_assets]

        else:
            df_target_train = df.copy()
            df_target_test = df.copy()
            extra_features=list(dict_df_features.values())
            dict_features_train = {name_asset: df[:train_size] for name_asset, df in dict_df_features.items()} if all(extra_features) else dict_df_features
            dict_features_val = {name_asset: df[train_size:] for name_asset, df in dict_df_features.items()} if all(extra_features) else dict_df_features
            
        df_target_train = df_target_train[:train_size]
        df_target_val = df_target_test[train_size:]
        return (df_target_train,dict_features_train), (df_target_val,dict_features_val) # (df_train, df_feature_train) , ()
