import pandas as pd


def wiki_sarcoma_df_merger(label_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged_feature_df = feature_df.merge(label_df,
                                         left_on="ID",
                                         right_on="Patient ID",
                                         how="left")
    merged_feature_df = merged_feature_df[merged_feature_df['Grade'].notna()]

    merged_feature_df['Grade'] = merged_feature_df['Grade'].map(
        {v: k for k, v in enumerate(merged_feature_df['Grade'].unique())})

    return merged_feature_df


def meningioma_df_merger(label_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged_feature_df = feature_df.merge(label_df,
                                         left_on='ID',
                                         right_on="Patient_ID",
                                         how='left')
    merged_feature_df = merged_feature_df[merged_feature_df['Grade'].notna()]

    return merged_feature_df
