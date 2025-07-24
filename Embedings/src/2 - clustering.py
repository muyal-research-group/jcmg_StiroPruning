import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import os
import joblib

import hdbscan

timings = []
timings_metricas = []

def log_time(log, label, start, end):
    elapsed = end - start
    timings.append({
        "etapa": label,
        "tiempo_segundos": round(elapsed, 4)
    })
    print(f"{log} completado en {elapsed:.4f} segundos.")

def compute_metrics(name, X, labels):
    start = time.time()
    valid_labels = labels[labels != -1] if -1 in labels else labels
    valid_X = X[labels != -1] if -1 in labels else X

    if len(set(valid_labels)) > 1:
        silhouette = silhouette_score(valid_X, valid_labels)
        ch_score = calinski_harabasz_score(valid_X, valid_labels)
        db_score = davies_bouldin_score(valid_X, valid_labels)
    else:
        silhouette, ch_score, db_score = -1, -1, -1

    print(f"\nMétricas para {name}:")
    print(f"→ Silhouette Score:        {silhouette:.4f}")
    print(f"→ Calinski-Harabasz Index: {ch_score:.4f}")
    print(f"→ Davies-Bouldin Index:    {db_score:.4f}")

    timings_metricas.append({
        "etapa": f"Métricas_{name}",
        "tiempo_segundos": time.time() - start,
        "Silhouette": round(silhouette, 4),
        "Calinski-Harabasz": round(ch_score, 4),
        "Davies-Bouldin": round(db_score, 4)
    })

    return silhouette, ch_score, db_score

def run_kmeans(embeddings, n_clusters=5, model_dir="../test/Modelos"):
    print(f"\nEjecutando KMeans con {n_clusters} clusters...")
    start = time.time()

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_scaled)
    end = time.time()
    log_time(f"Clustering KMeans (k={n_clusters})", "KMeans", start, end)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"Scaler guardado en '{model_dir}/scaler.pkl'")

    joblib.dump(kmeans, os.path.join(model_dir, "modelo_kmeans.pkl"))
    print(f"Modelo KMeans guardado en '{model_dir}/modelo_kmeans.pkl'")

    compute_metrics("KMeans", embeddings_scaled, labels)
    return labels

def run_dbscan(embeddings, eps=0.5, min_samples=5, model_dir="../test/Modelos"):
    print(f"\nEjecutando DBSCAN con eps={eps}, min_samples={min_samples}...")
    start = time.time()

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings_scaled)
    end = time.time()

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Clustering con DBSCAN completado.")
    print(f"Número de clusters encontrados: {n_clusters}")
    print(f"Número de puntos marcados como ruido: {n_noise}")
    log_time("Clustering DBSCAN", "DBSCAN", start, end)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler_dbscan.pkl"))
    print(f"Scaler DBSCAN guardado en '{model_dir}/scaler_dbscan.pkl'")

    joblib.dump(dbscan, os.path.join(model_dir, "modelo_dbscan.pkl"))
    print(f"Modelo DBSCAN guardado en '{model_dir}/modelo_dbscan.pkl'")

    compute_metrics("DBSCAN", embeddings_scaled, labels)
    return labels

def run_kmeans_range(embeddings, k_range=range(2, 13), model_dir="../test/Modelos"):
    print(f"\nEjecutando KMeans para k en {list(k_range)}...")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"Scaler guardado en '{model_dir}/scaler.pkl'")

    best_silhouette = -1
    best_k = None
    best_labels = None

    for k in k_range:
        print(f"\n→ k = {k}")
        start = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, init="random", max_iter=500)
        labels = kmeans.fit_predict(embeddings_scaled)
        end = time.time()

        log_time(f"KMeans con k={k}", f"KMeans_k{k}", start, end)
        silhouette, ch_score, db_score = compute_metrics(f"KMeans_k{k}", embeddings_scaled, labels)

        # Guardar modelo
        k_model_path = os.path.join(model_dir, f"modelo_kmeans_k{k}.pkl")
        joblib.dump(kmeans, k_model_path)
        print(f"Modelo KMeans(k={k}) guardado en '{k_model_path}'")

        # Guardar el mejor k según Silhouette
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
            best_labels = labels
            best_model = kmeans

    print(f"\nMejor k según Silhouette: {best_k} (score: {best_silhouette:.4f})")
    
    # Guardar modelo seleccionado como principal
    joblib.dump(best_model, os.path.join(model_dir, "modelo_kmeans.pkl"))
    print(f"Modelo KMeans(k={best_k}) guardado como 'modelo_kmeans.pkl'")

    return best_labels, embeddings_scaled

def run_hdbscan(embeddings, min_cluster_size=5, model_dir="../test/Modelos"):
    print(f"\nEjecutando HDBSCAN con min_cluster_size={min_cluster_size}...")
    start = time.time()

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=7,
                                metric='euclidean',#'euclidean',  # o 'cosine'
                                cluster_selection_method='leaf',
                                prediction_data=True)
    labels = clusterer.fit_predict(embeddings_scaled)
    end = time.time()

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Clustering con HDBSCAN completado.")
    print(f"Número de clusters encontrados: {n_clusters}")
    print(f"Número de puntos marcados como ruido: {n_noise}")

    log_time("Clustering HDBSCAN", "HDBSCAN", start, end)
    compute_metrics("HDBSCAN", embeddings_scaled, labels)

    # Guardar modelo y scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler_hdbscan.pkl"))
    joblib.dump(clusterer, os.path.join(model_dir, "modelo_hdbscan.pkl"))
    print(f"Modelo HDBSCAN guardado en '{model_dir}/modelo_hdbscan.pkl'")
    print(f"Scaler guardado en '{model_dir}/scaler_hdbscan.pkl'")

    return labels


if __name__ == "__main__":
    embeddings_path = "../test/vectores_csv.npy"
    print(f"\nCargando vectores desde '{embeddings_path}'...")
    embeddings = np.load(embeddings_path)

    # kmeans_labels = run_kmeans(embeddings, n_clusters=5)
    # kmeans_labels, embeddings_scaled = run_kmeans_range(embeddings)
    dbscan_labels = run_dbscan(embeddings, eps=1, min_samples=3)

    df = pd.DataFrame(embeddings)
    df.columns = [f"dim_{i+1}" for i in range(df.shape[1])]
    # df["KMeans_Label"] = kmeans_labels
    df["DBSCAN_Label"] = dbscan_labels
    df.to_csv("../test/vectores_clusters.csv", index=False)
    print("\nVectores con etiquetas guardados en 'vectores_clusters.csv'")

    hdbscan_labels = run_hdbscan(embeddings, min_cluster_size=8)

    df = pd.DataFrame(embeddings)
    df.columns = [f"dim_{i+1}" for i in range(df.shape[1])]
    df["HDBSCAN_Label"] = hdbscan_labels
    df.to_csv("../test/vectores_hdbscan.csv", index=False)

    os.makedirs("../test/Logs", exist_ok=True)
    pd.DataFrame(timings).to_csv("../test/Logs/timing_hdbscan.csv", index=False)
    pd.DataFrame(timings_metricas).to_csv("../test/Logs/timing_hdbscan_metricas.csv", index=False)

    print("\nResultados y métricas de HDBSCAN guardados.")

    
    os.makedirs("../test/Logs", exist_ok=True)
    pd.DataFrame(timings).to_csv("../test/Logs/timing_clustering.csv", index=False)
    print("Tiempos guardados en 'timing_clustering.csv'")

    pd.DataFrame(timings_metricas).to_csv("../test/Logs/timing_clustering_metricas.csv", index=False)
    print("Métricas guardadas en 'timing_clustering_metricas.csv'")
