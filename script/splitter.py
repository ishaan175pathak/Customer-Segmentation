from .preprocessor import DataPreprocessor
from .feature_builder import UserFeatureBuilder
import pandas
import numpy
from sklearn.model_selection import train_test_split
from typing import List, Any, Tuple, TypeAlias, Union


class DataSplitter:

    def __init__(self, dataFrame: pandas.DataFrame) -> None:
        self.dataFrame: pandas.DataFrame = dataFrame
        
    def __split_dataset__(self, random_state: int = 42, stratify_y: bool = False) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]:
        
        dataset: pandas.DataFrame = self.dataFrame.dropna()
        
        y: pandas.Series = dataset["subscription_status"]
        dataset = dataset.drop(["user_id", "subscription_status"], axis=1)
                
        dp: DataPreprocessor = DataPreprocessor(dataset)
        dataFrame: pandas.DataFrame = dp.__preprocess_features__()

        
        X: pandas.DataFrame = dataFrame

        X_train: pandas.DataFrame
        X_test: pandas.DataFrame
        y_train: pandas.Series
        y_test: pandas.Series

        if stratify_y:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


        return (X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":

    
    db: UserFeatureBuilder = UserFeatureBuilder("./mobile_app_interactions_expanded.csv")
    user_df: pandas.DataFrame = db.__build_user_level_dataset__()


    ds: DataSplitter = DataSplitter(user_df)

    X_train: pandas.DataFrame
    X_test: pandas.DataFrame
    y_train: pandas.Series
    y_test: pandas.Series

    (X_train, X_test, y_train, y_test) = ds.__split_dataset__()

    assert X_train.shape[1] == X_test.shape[1]

    print(f"\n \
        \t X_train Shape - {X_train.shape}\n\
        \t X_test Shape - {X_test.shape}\n\
        \t y_train Shape - {y_train.shape}\n\
        \t y_test Shape - {y_test.shape}")