"""
Laboratório 9 — RAG com HNSW, HyDE (LLM local) e re-ranking com Cross-Encoder.

Passo 1 (este commit): corpus simulado, embeddings locais e índice FAISS HNSW.
"""

from __future__ import annotations

from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Hiperparâmetros HNSW (explícitos, conforme PDF) ---
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200

BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fragmentos fictícios de manual clínico/técnico (jargão médico em PT-BR).
MANUAL_FRAGMENTS: List[str] = [
    "Cefaléia tensional: dor bilateral em pressão, sem náusea importante. Tratamento de primeira linha: AINEs.",
    "Cefaléia em salvas: dor periorbitária unilateral intensa, lacrimejamento e rinorreia ipsilateral.",
    "Migrânea sem aura: crises pulsáteis de 4–72 h, fotofobia e fonofobia frequentes.",
    "Migrânea com aura: sintomas neurológicos reversíveis precedendo ou acompanhando a cefaléia.",
    "Fotofobia na migrânea: sensibilidade à luz ambiente; distingue-se de outras causas oculares.",
    "Cefaléia por uso excessivo de medicamento: piora com analgésicos frequentes, >15 dias/mês.",
    "Hipertensão intracraniana benigna: cefaléia difusa com papiledema; investigar com fundo de olho.",
    "Meningite bacteriana: rigidez de nuca, febre e alteração do estado mental; punção lombar urgente.",
    "Sinusite aguda bacteriana: dor facial purulenta persistente >10 dias ou piora bifásica.",
    "Neuralgia do trigêmeo: crises elétricas em território V2/V3, gatilhos leves na face.",
    "Temporal arterite: cefaléia nova em >50 anos com claudicação de mandíbula; VHS elevado.",
    "Hemorragia subaracnoidea: início abrupto em trovão; TC sem contraste e depois PL se indicado.",
    "AVC isquêmico agudo: déficit neurológico focal súbito; janela para trombólise limitada no tempo.",
    "Enxaqueca crônica: ≥15 dias com cefaléia migranosa/mês, por >3 meses.",
    "Vertigem periférica (BPPV): crises breves com mudança de posição; manobra de Epley.",
    "Vertigem central: nistagmo multidirecional ou déficits focais; imagem prioritária.",
    "Síndrome febril sem foco no lactente: conduta etária e risco de bacteremia oculta.",
    "Dor torácica de provável origem isquêmica: ECG seriado e troponinas; regra de escores.",
    "TEP agudo: dispneia súbita, taquicardia; D-dímero e angioTC conforme probabilidade pré-teste.",
    "Pneumonia adquirida na comunidade: critérios de internação (CURB-65, comorbidades).",
    "Asma exacerbada: espirômetro, beta-2 inalatório e corticoide sistêmico conforme gravidade.",
    "ICC descompensada: congestão, peso, diurético de alça e avaliação de perfusão renal.",
    "Cetoacidose diabética: ânion gap, fluidos, insulina IV e monitorização de potássio.",
    "Hipotireoidismo: TSH elevado com T4 livre baixo ou borderline; reposição com levotiroxina.",
]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return l2_normalize(np.asarray(emb, dtype=np.float32))


def build_hnsw_index(embeddings: np.ndarray) -> faiss.Index:
    """Índice HNSW com produto interno = cosseno após normalização L2 dos vetores."""
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(embeddings)
    return index


def main() -> None:
    print(f"Fragmentos no corpus: {len(MANUAL_FRAGMENTS)}")
    print("Carregando bi-encoder local...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    print("Gerando embeddings e construindo HNSW...")
    doc_embeddings = embed_texts(bi_encoder, MANUAL_FRAGMENTS)
    index = build_hnsw_index(doc_embeddings)
    print(f"Índice FAISS pronto. Vetores: {index.ntotal}, dim={index.d}, M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}")


if __name__ == "__main__":
    main()
