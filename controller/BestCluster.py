# Automatically determine optimal clusters with silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def best_number_of_cluster(data):
    max_clusters = min(10, len(data))  # Set an upper limit for efficiency
    best_clusters = 2
    best_score = -1

    for n in range(2, max_clusters + 1):
        model = KMeans(n_clusters=n)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_clusters = n
    return best_clusters