import numpy as np
import pandas as pd
import joblib
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.readers import SimpleDirectoryReader

def generar_embedding_desde_lista(entrada, modelo_path="modelo_embeddings"):
    """
    Une los elementos del vector en una cadena y genera el embedding.
    """
    texto = " ".join(entrada)
    print(f"Texto generado a partir del vector: '{texto}'")
    embedder = HuggingFaceEmbedding(model_name=modelo_path, device="cuda")
    embedding = embedder._get_text_embedding(texto)
    return embedding

def predecir_grupo_y_extraer(embedding, model_path="modelo_kmeans.pkl", scaler_path="scaler.pkl", data_path="vectores_clusters.csv"):
    print("Cargando modelo KMeans y scaler...")
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"Cargando datos desde '{data_path}'...")
    df = pd.read_csv(data_path)

    num_dims = kmeans.cluster_centers_.shape[1]
    embedding_cols = [f"dim_{i+1}" for i in range(num_dims)]

    print("Escalando vector de entrada...")
    embedding_scaled = scaler.transform([embedding])

    grupo = kmeans.predict(embedding_scaled)[0]
    print(f"\nEl vector pertenece al grupo: {grupo}")

    grupo_df = df[df["KMeans_Label"] == grupo]
    print(f"Se encontraron {len(grupo_df)} vectores en ese grupo.")
    return grupo, grupo_df

def guardar_vectores_con_texto_original(csv_file_path, df_clusters, output_path="../test/vectores_originales_con_clusters.csv"):
    """
    Guarda un archivo CSV que incluye:
    - Las dimensiones de los embeddings
    - Las etiquetas de clustering (KMeans y DBSCAN)
    - El texto original
    - Y los campos extraídos del texto original
    """
    print(f"\nCargando textos originales desde '{csv_file_path}'...")
    loader = SimpleDirectoryReader(input_files=[csv_file_path], encoding="utf-8")
    documents = loader.load_data()
    texts = [doc.text.strip() for doc in documents]

    if len(texts) != len(df_clusters):
        print("Advertencia: El número de textos originales no coincide con los embeddings.")
        min_len = min(len(texts), len(df_clusters))
        texts = texts[:min_len]
        df_clusters = df_clusters.iloc[:min_len]

    df_clusters = df_clusters.copy()
    df_clusters["Texto_Original"] = texts

    # Dividir los textos originales en 5 columnas
    columnas = ["Espacial", "Temporal", "Interes", "Referencia", "Observacion"]
    descompuestos = df_clusters["Texto_Original"].str.split(" ", expand=True)

    if descompuestos.shape[1] == 5:
        descompuestos.columns = columnas
        df_clusters = pd.concat([df_clusters, descompuestos], axis=1)
    else:
        print("Advertencia: No se pudieron descomponer los textos en 5 partes.")

    df_clusters.to_csv(output_path, index=False)
    print(f"Archivo guardado con texto original y columnas extraídas en: {output_path}")


# if __name__ == "__main__":
#     # Vector de entrada (orden: espacial, temporal, interés, referencia, observación)
#     entrada = ["Tamaulipas.Tampico", "2000", "Mujeres.45_64", "1.5", "1.3"]

#     embedding = generar_embedding_desde_lista(entrada)
#     grupo, resultados = predecir_grupo_y_extraer(embedding=embedding,
#                                                  model_path="../test/Modelos/modelo_kmeans.pkl", 
#                                                  scaler_path="../test/Modelos/scaler.pkl",
#                                                  data_path="../test/vectores_clusters.csv")

#     resultados.to_csv(f"../test/vectores_grupo_{grupo}.csv", index=False)
#     print(f"\nResultados guardados en 'vectores_grupo_{grupo}.csv'")

def extraer_datos_originales_por_grupo(grupo_id, cluster_data_path, original_csv_path, output_path):
    """
    Extrae los datos originales (no embeddings) del grupo predicho.

    Parámetros:
        grupo_id (int): Número del grupo KMeans predicho.
        cluster_data_path (str): CSV con etiquetas de clustering.
        original_csv_path (str): CSV con los datos originales.
        output_path (str): Ruta donde se guardará el nuevo CSV.

    Resultado:
        Guarda un archivo CSV con las filas originales correspondientes al grupo.
    """
    print(f"\nCargando clustering desde '{cluster_data_path}'...")
    df_clusters = pd.read_csv(cluster_data_path)

    print(f"Cargando datos originales desde '{original_csv_path}'...")
    df_original = pd.read_csv(original_csv_path)

    if len(df_clusters) != len(df_original):
        print("Advertencia: Las dimensiones no coinciden entre embeddings y datos originales.")
        min_len = min(len(df_clusters), len(df_original))
        df_clusters = df_clusters.iloc[:min_len]
        df_original = df_original.iloc[:min_len]

    # Obtener índices del grupo KMeans deseado
    indices_grupo = df_clusters[df_clusters["KMeans_Label"] == grupo_id].index

    # Extraer filas originales usando los índices
    datos_filtrados = df_original.loc[indices_grupo]

    datos_filtrados.to_csv(output_path, index=False)
    print(f"Datos originales del grupo {grupo_id} guardados en '{output_path}'")


if __name__ == "__main__":
    entrada = ["Tamaulipas.Tampico", "2000", "Mujeres.45_64", "1.5", "1.3"]

    embedding = generar_embedding_desde_lista(entrada)
    grupo, _ = predecir_grupo_y_extraer(
        embedding=embedding,
        model_path="../test/Modelos/modelo_kmeans.pkl", 
        scaler_path="../test/Modelos/scaler.pkl",
        data_path="../test/vectores_clusters.csv"
    )

    extraer_datos_originales_por_grupo(
        grupo_id=grupo,
        cluster_data_path="../test/vectores_clusters.csv",
        original_csv_path="../data/sample.csv",
        output_path=f"../test/vectores_grupo_{grupo}_originales.csv"
    )


# if __name__ == "__main__":
#     # Vector de entrada (orden: espacial, temporal, interés, referencia, observación)
#     entrada = ["Tamaulipas.Tampico", "2000", "Mujeres.45_64", "1.5", "1.3"]

#     # Paso 1: Obtener embedding de entrada
#     embedding = generar_embedding_desde_lista(entrada)

#     # Paso 2: Predecir grupo y extraer todos los vectores de ese grupo
#     grupo, resultados = predecir_grupo_y_extraer(
#         embedding=embedding,
#         model_path="../test/Modelos/modelo_kmeans.pkl", 
#         scaler_path="../test/Modelos/scaler.pkl",
#         data_path="../test/vectores_clusters.csv"
#     )

#     # Paso 3: Guardar resultados del grupo
#     resultados_path = f"../test/vectores_grupo_{grupo}.csv"
#     resultados.to_csv(resultados_path, index=False)
#     print(f"\nResultados guardados en '{resultados_path}'")

#     # Paso 4: Guardar archivo extendido con texto original + etiquetas
#     # Pendiente
#     # 
#     guardar_vectores_con_texto_original(
#         csv_file_path="../data/sample.csv",
#         df_clusters=pd.read_csv("../test/vectores_clusters.csv"),
#         output_path="../test/vectores_originales_con_clusters.csv"
#     )
