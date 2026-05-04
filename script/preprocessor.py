from typing import Any, List
from .feature_builder import UserFeatureBuilder
import numpy
import pandas
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer 
from pathlib import Path


class DataPreprocessor:

    def __init__(self, dataframe: pandas.DataFrame) -> None:
        """
            This Class would contribute in Preprocessing the Dataset, and Preparing the dataset for Machine Learning Model

            Args:
                dataframe (pandas.Dataframe):

            Returns:
                None
        """

        self.dataFrame: pandas.DataFrame = dataframe
    

    def __preprocess_features__(self) -> pandas.DataFrame:
        
        numeric_columns_df: numpy.ndarray = self.dataFrame.select_dtypes(include=[numpy.int64, numpy.float64]).columns.to_numpy()

        num_pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer())
        ])


        pipeline: ColumnTransformer = ColumnTransformer([
            ("scaled", num_pipeline, numeric_columns_df),
        ], remainder="passthrough")

        scaled_features: numpy.ndarray | Any = pipeline.fit_transform(self.dataFrame)

        return pandas.DataFrame(scaled_features, columns=list(pipeline.get_feature_names_out()))


if __name__ == "__main__":

    db: UserFeatureBuilder = UserFeatureBuilder("./mobile_app_interactions_expanded.csv")
    user_df = db.__build_user_level_dataset__()


    dp: DataPreprocessor = DataPreprocessor(user_df)

    df: pandas.DataFrame = dp.__preprocess_features__()
    print(df.head())