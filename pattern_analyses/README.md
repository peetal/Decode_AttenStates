# To reproduce results in Figure 5
The notebooks performed pattern analyses (using both background FC and stimulus-evoked activity) reported in Figure 5 of the manuscript. To go through the notebook, please download the data from OSF (see data folder for details). 
Both notebooks examined the clusters' sensitivity to each of the three distinctions (cognitive state, visual content and behavioral tasks). For each distinction, the code always go through the following steps: 
- Parse the timeseries into epochs of interest (i.e., perception vs retrieval epochs for cognitive state distinction; face vs. scene epochs for visual content distinction; and gender vs. naturalness for behavioral task distinction). 
- Compute within-class (e.g., face - face similarity) and between-class (e.g., face - scene) pattern similarities for each cluster. 
- Compute sensitivity index as within-class PS - between-class PS.
