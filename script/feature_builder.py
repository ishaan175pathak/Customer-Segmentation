from __future__ import annotations

import numpy
import pandas
import os
import matplotlib.pyplot as plt
from .data_loader import DataLoader
from pathlib import Path
from typing import TypeAlias, List, Any, Tuple, Literal
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

# defining custom DataType

CSVFile = str
ZIPFile = str


class UserFeatureBuilder:

    def __init__(self, /, fileName: CSVFile | ZIPFile) -> None:
        """
            This Module reads a CSV or ZIP file and performs Feature Engineering on the dataset
            Args:
                fileName (CSVFile or ZIPFile): It could be a Zip or CSV file, but if its a Zip file then we will extract the dataset and read it as CSV.
            
            Returns:
                None
        """

        file_name: str = str(fileName)
        cwd: Path = Path().cwd()

        filePath: Path = cwd / file_name

        self.dataset: pandas.DataFrame

        if filePath.suffix == ".zip":
            dl: DataLoader = DataLoader(indir=cwd, inputFile=filePath)
            self.dataset = dl.__extractFile__()
        elif filePath.suffix == ".csv":
            self.dataset = pandas.read_csv(filePath)
        else:
            raise FileNotFoundError("Could not the find the file with the provided Name !")
        
    
    def __normalize_boolean__(self, value) -> int | float:
        if pandas.isna(value):
            return numpy.nan

        value_str = str(value).strip().lower()

        if value_str in {"true", "1", "yes", "y"}:
            return 1
        if value_str in {"false", "0", "no", "n"}:
            return 0

        return numpy.nan

    def __clean_device_os__(self, value) -> str | float:
        if pandas.isna(value):
            return numpy.nan

        value_str = str(value).strip().lower()

        if "android" in value_str or "andro" in value_str:
            return "Android"
        if "ios" in value_str or "iphone" in value_str:
            return "iOS"

        return numpy.nan

    def __mode_or_nan__(self, series) -> Any | float:
        mode_vals: pandas.Series = series.dropna().mode()
        return mode_vals.iloc[0] if not mode_vals.empty else numpy.nan

    def __clean_data__(self) -> pandas.DataFrame:

        # Convert timestamp
        self.dataset["timestamp"] = pandas.to_datetime(self.dataset["timestamp"], errors="coerce")

        # Numeric columns
        numeric_cols: List[str] = [
            "session_duration_sec",
            "battery_level",
            "memory_usage_mb",
            "event_value",
            "user_age"
        ]

        for col in numeric_cols:
            self.dataset[col] = pandas.to_numeric(self.dataset[col], errors="coerce")

        # Boolean columns
        self.dataset["is_subscribed"] = self.dataset["is_subscribed"].apply(self.__normalize_boolean__)
        self.dataset["push_enabled"] = self.dataset["push_enabled"].apply(self.__normalize_boolean__)

        # Clean categorical
        self.dataset["device_os"] = self.dataset["device_os"].apply(self.__clean_device_os__)

        # Extract event date
        self.dataset["event_date"] = self.dataset["timestamp"].dt.date

        return self.dataset

    def __build_user_level_dataset__(self) -> pandas.DataFrame:
        self.__clean_data__()

        user_df: pandas.DataFrame = (
            self.dataset
            .groupby("user_id")
            .agg(
                total_sessions=("session_id", "nunique"),
                total_events=("user_id", "size"),
                active_days=("event_date", "nunique"),
                avg_session_duration=("session_duration_sec", "mean"),
                avg_event_value=("event_value", "mean"),
                avg_battery_level=("battery_level", "mean"),
                avg_memory_usage_mb=("memory_usage_mb", "mean"),
                user_age=("user_age", self.__mode_or_nan__),
                subscription_status=("is_subscribed", self.__mode_or_nan__)
            )
            .reset_index()
        )

        # changing the dtypes of all the columns to int except userID

        for column in user_df.columns:
            if column != "user_id":
                user_df[column] = pandas.to_numeric(user_df[column], errors="coerce")
                

        return user_df
    

if __name__ == "__main__":
    builder: UserFeatureBuilder = UserFeatureBuilder("mobile_app_interactions_expanded.csv")
    dataFrame: pandas.DataFrame = builder.__build_user_level_dataset__()

    print(f" DataFrame Head -\n {dataFrame.head()}\n\n\
        DataFrame Shape -\n {dataFrame.shape}\n\n\
        DataFrame DTypes -\n {dataFrame.dtypes}\n\n\
        DataFrame null values -\n {dataFrame.isnull().sum()}\n\n\
        Subscription Status -\n {dataFrame["subscription_status"].value_counts(dropna=False)}\n\n\
        DataFrame Duplicate Values -\n {dataFrame.duplicated().sum()}")