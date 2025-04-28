from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

penguins_df = load_penguins()
penguins_cleaned_df = penguins_df.dropna()

selected_features = ["bill_length_mm", "flipper_length_mm"]

model_params = {"n_clusters": 3,
                "max_iter": 300}

penguins_data_df = penguins_cleaned_df.copy()
features_df = penguins_data_df.loc[:, selected_features]

standard_model = StandardScaler()
features_scaled_model_1 = scaler_model_1.fit_transform(features)

kmeans_model_1 = KMeans(
    n_clusters=model_params["n_clusters"], max_iter=model_params["max_iter"])

penguin_predictions = kmeans_model_1.fit_predict(features_scaled_model_1)
penguins_data_df.loc[:, "cluster"] = penguin_predictions

cluster_species = penguins_data_df.groupby(
    "species")["cluster"].agg(lambda x: x.value_counts().index[0])

cluster_species_df = cluster_species.reset_index()

penguins_data_df['cluster'] = penguins_data_df['cluster'].astype(str)
cluster_species_df['cluster'] = cluster_species_df['cluster'].astype(str)

penguins_data_df.loc[:, "cluster"] = penguins_data_df["cluster"].map(
    cluster_species_df.set_index("cluster")["species"])

cols_compare = ["species",
                "cluster", "bill_length_mm", "flipper_length_mm"]
penguins_selected = penguins_data_df[cols_compare]

penguins_selected = penguins_selected.rename(columns={
    "species": "actual_species", "cluster": "predicted_species"})

penguins_crosstabbed = pd.crosstab(penguins_selected['predicted_species'],
                                   penguins_selected['actual_species'])

penguins_crosstabbed_mismatch = penguins_crosstabbed.copy(deep=True)

np.fill_diagonal(penguins_crosstabbed_mismatch.values, 0)
