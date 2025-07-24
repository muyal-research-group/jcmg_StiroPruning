import pandas as pd
import os
import time
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import CSVReader as CustomCSVReader
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel, AutoTokenizer
import numpy as np

# --- Inicializar registro de tiempos ---
timings = []

def log_time(log, label, start, end):
    elapsed = end - start
    timings.append({"step": label, "time_seconds": round(elapsed, 4)})
    print(f"{log} completado en {elapsed:.4f} segundos.")

def main():
    # --- 1. Configurar modelo de embeddings ---
    start = time.time()
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device='cuda')
    end = time.time()
    log_time("Configuración del modelo de embeddings", "Config_model", start, end)

    csv_file_path = '../data/sample.csv'
    embeddings_output_path = '../test/vectores_csv.npy'

    # --- 2. Verificar existencia del CSV ---
    start = time.time()
    print(f"Verificando si el archivo '{csv_file_path}' existe...")
    if not os.path.exists(csv_file_path):
        print(f"El archivo '{csv_file_path}' no se encontró.")
        exit("El archivo CSV no fue encontrado. Por favor, asegúrate de que exista.")
    end = time.time()
    log_time("Verificación del archivo CSV", "Verif_CSV", start, end)

    # --- 3. Cargar datos con SimpleDirectoryReader ---
    start = time.time()
    print(f"Cargando el archivo '{csv_file_path}' con SimpleDirectoryReader...")
    loader = SimpleDirectoryReader(
        input_files=[csv_file_path],
        encoding="utf-8"
    )
    documents = loader.load_data()
    end = time.time()

    print(f"\nSe cargaron {len(documents)} documentos del CSV.")
    if documents:
        print(f"\nEjemplo del contenido del primer documento cargado:")
        print(f"Metadata: {documents[0].metadata}")

    log_time("Carga de documentos", "Load_Docs", start, end)
    # --- 4. Crear índice de vectores ---
    start = time.time()
    print("\nCreando el índice de vectores. Esto va a tardar un momento...")
    index = VectorStoreIndex.from_documents(documents)
    end = time.time()
    log_time("Creación del índice de vectores", "Index_Vectors", start, end)

    # --- 5. Generar embeddings ---
    start = time.time()
    print("\n--- Vectores (Embeddings) Generados por Documento ---")
    nodes = list(index.docstore.docs.values())
    embeddings = index._embed_model._get_text_embeddings([node.text for node in nodes])
    all_embeddings = [embedding for embedding in embeddings]
    end = time.time()
    log_time("Generación de embeddings", "Generate_embeddings", start, end)

    # --- 6. Guardar vectores en .npy ---
    if all_embeddings:
        start = time.time()
        embeddings_array = np.array(all_embeddings)
        np.save(embeddings_output_path, embeddings_array)
        end = time.time()
        print(f"\n¡Vectores guardados exitosamente en '{embeddings_output_path}'!")
        # print("Puedes cargarlos más tarde usando: `np.load('vectores_csv.npy')`")
        log_time("Guardado de vectores en archivo .npy",  "Save_vectors", start, end)
    else:
        print("\nNo se generaron vectores para los documentos.")

    # --- 7. Guardar el modelo HuggingFace ---
    start = time.time()
    print("\nGuardando el modelo de embeddings HuggingFace...")
    model_name = "BAAI/bge-m3"
    hf_model = AutoModel.from_pretrained(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_dir = "modelo_embeddings"
    hf_model.save_pretrained(model_dir)
    hf_tokenizer.save_pretrained(model_dir)
    end = time.time()
    log_time(f"Modelo y tokenizer guardados en la carpeta '{model_dir}'", "save_model", start, end)

    # --- 8. Guardar tiempos en CSV ---
    timings_df = pd.DataFrame(timings)
    timings_df.to_csv("../test/Logs/timings_vectors.csv", index=False)
    print("\nTiempos de ejecución guardados en 'timings_vectors.csv'")


if __name__ == "__main__":
    main()

