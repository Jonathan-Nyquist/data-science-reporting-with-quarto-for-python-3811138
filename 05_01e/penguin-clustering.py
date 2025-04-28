from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

penguins = load_penguins()
penguins_cleaned = penguins.dropna()

selected_features = ["bill_length_mm", "flipper_length_mm"]

model_1_params = {"n_clusters": 3,
                  "max_iter": 300}

penguins_model_1 = penguins_cleaned

features = penguins_model_1.loc[:, selected_features]

scaler_model_1 = StandardScaler()
features_scaled_model_1 = scaler_model_1.fit_transform(features)

kmeans_model_1 = KMeans(
    n_clusters=model_1_params["n_clusters"], max_iter=model_1_params["max_iter"])

penguin_predictions = kmeans_model_1.fit_predict(features_scaled_model_1)
penguins_model_1.loc[:, "cluster"] = penguin_predictions

cluster_species = penguins_model_1.groupby(
    "species")["cluster"].agg(lambda x: x.value_counts().index[0])

cluster_species_df = cluster_species.reset_index()

penguins_model_1['cluster'] = penguins_model_1['cluster'].astype(str)
cluster_species_df['cluster'] = cluster_species_df['cluster'].astype(str)

penguins_model_1.loc[:, "cluster"] = penguins_model_1["cluster"].map(
    cluster_species_df.set_index("cluster")["species"])

cols_compare = ["species",
                "cluster", "bill_length_mm", "flipper_length_mm"]
penguins_selected = penguins_model_1[cols_compare]

penguins_selected = penguins_selected.rename(columns={
    "species": "actual_species", "cluster": "predicted_species"})

penguins_crosstabbed = pd.crosstab(penguins_selected['predicted_species'],
                                   penguins_selected['actual_species'])

penguins_crosstabbed_mismatch = penguins_crosstabbed.copy(deep=True)

np.fill_diagonal(penguins_crosstabbed_mismatch.values, 0)
