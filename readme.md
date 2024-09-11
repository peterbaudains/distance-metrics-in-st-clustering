# Distance Metrics in ST-Clustering

Supporting code for:

Baudains, P., & Holliman, N. S. (2024). Comparing Distance Metrics in Space-time Clustering to Provide Visual Summaries of Traffic Congestion. In D. Hunter & A. Slingsby (Eds.), Computer Graphics and Visual Computing (CGVC). The Eurographics Association. https://doi.org/10.2312/cgvc.20241233.

The code requires a Neo4j Enterprise instance running with the following environment variables defined in a .env file:

```
NEO4J_SERVER
NEO4J_USER
NEO4J_PASSWORD
DB_NAME
```

The expected data model is outlined in the paper referenced above.

The `data_loader` module retrieves the data from the Neo4j instance, using a lat/lon-defined bounding box and a specified time window. The Cypher query in the `get_obs_for_time_period()` function details the expected property fields.

Experiments are run with the following scripts:

1. `run_eucl_clustering.py` implements a single run of the DBSCAN algorithm with a Euclidean distance metric.

2. `run_network_clustering.py` implements a single run of the DBSCAN algorithm with a Network distance metric.

3. `nrt_eucl.py` implements a series of runs of the DBSCAN algorithm with a Euclidean distance metric by chunking up the overall time period. This is used to construct the post-view evaluation metrics as explained in the paper cited above.

4. `nrt_network.py` implements a series of runs of the DBSCAN algorithm with a network-based distance metric by chunking up the overall time period. This is used to construct the post-view evaluation metrics as explain in the paper cited above.

The above scripts also require a `logs` directory and an `outputs` directory in the root of the main local repo.
