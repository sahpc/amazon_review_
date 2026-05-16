
import pandas as pd

def clean_reviews(df):

    df = df[df['HelpfulnessDenominator'] >= 5]

    df = df.drop_duplicates(
        subset=['UserId', 'ProductId', 'Time']
    )

    df['helpfulness_ratio'] = (
        df['HelpfulnessNumerator'] /
        df['HelpfulnessDenominator']
    )

    df['is_helpful'] = (
        df['helpfulness_ratio'] >= 0.7
    ).astype(int)

    df = df.dropna(subset=['Text'])

    return df
