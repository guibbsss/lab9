from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import faiss
import numpy as np
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(embeddings)
    return index


def hyde_hypothetical_document(user_query: str, ollama_model: str = OLLAMA_MODEL) -> str:
    prompt = (
        "Voce e um medico especialista escrevendo um trecho de manual clinico interno, em portugues, "
        "com terminologia tecnica (sem citar que e ficticio). "
        "Escreva APENAS o paragrafo do manual (3 a 6 frases), direto ao ponto, sem saudacoes, "
        "sem markdown, sem lista numerada.\n\n"
        f"Pergunta coloquial do paciente: {user_query}\n\n"
        "Trecho do manual:"
    )
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    response = requests.post(
        url,
        json={"model": ollama_model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    text = (payload.get("response") or "").strip()
    if not text:
        raise RuntimeError(
            "Ollama retornou texto vazio. Verifique se o servico esta rodando "
            f"({url}) e se o modelo '{ollama_model}' existe (ex.: ollama pull {ollama_model})."
        )
    return text


def embed_query_vector(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    return l2_normalize(np.asarray(vec, dtype=np.float32))


def search_hnsw_topk(
    index: faiss.Index,
    query_vector: np.ndarray,
    k: int = 10,
    ef_search: int = HNSW_EF_SEARCH,
) -> tuple[np.ndarray, np.ndarray]:
    index.hnsw.efSearch = ef_search
    scores, ids = index.search(query_vector.astype(np.float32), k)
    return scores[0], ids[0]


def rerank_with_cross_encoder(
    cross_encoder: CrossEncoder,
    user_query: str,
    doc_ids: Sequence[int],
    corpus: List[str],
) -> List[tuple[int, float, str]]:
    pairs: List[List[str]] = [[user_query, corpus[int(i)]] for i in doc_ids]
    raw_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(doc_ids, raw_scores, [corpus[int(i)] for i in doc_ids]),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    top3: List[tuple[int, float, str]] = [
        (int(i), float(s), t) for i, s, t in ranked[:3]
    ]
    return top3


def main() -> None:
    print(f"Fragmentos no corpus: {len(MANUAL_FRAGMENTS)}")
    print("Carregando bi-encoder local...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    print("Gerando embeddings e construindo HNSW...")
    doc_embeddings = embed_texts(bi_encoder, MANUAL_FRAGMENTS)
    index = build_hnsw_index(doc_embeddings)
    print(f"Índice FAISS pronto. Vetores: {index.ntotal}, dim={index.d}, M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}")

    default_query = "dor de cabeca latejante e luz incomodando muito"
    parser = argparse.ArgumentParser(description="Lab 9 — RAG HNSW + HyDE + Cross-Encoder (local)")
    parser.add_argument(
        "--query",
        default=None,
        help="Pergunta coloquial do usuario (se omitido, usa RAG_QUERY ou o exemplo do PDF).",
    )
    args = parser.parse_args()
    user_query = args.query or os.environ.get("RAG_QUERY", default_query)
    print("\n--- HyDE (documento hipotetico) ---")
    print(f"Query coloquial: {user_query}")
    hypo = hyde_hypothetical_document(user_query)
    print("Documento hipotetico (trecho):\n", hypo[:1200])
    hyde_vec = embed_query_vector(bi_encoder, hypo)
    print(f"\nVetor HyDE pronto: shape={hyde_vec.shape}, norma L2={float(np.linalg.norm(hyde_vec)):.4f}")

    print("\n--- Top 10 recuperados (HNSW / bi-encoder, funil largo) ---")
    scores, ids = search_hnsw_topk(index, hyde_vec, k=10)
    for rank, (doc_id, score) in enumerate(zip(ids.tolist(), scores.tolist()), start=1):
        snippet = MANUAL_FRAGMENTS[int(doc_id)][:220].replace("\n", " ")
        print(f"{rank:2d}. id={int(doc_id):2d}  score={score:.4f}  |  {snippet}...")

    print("\n--- Top 3 apos Cross-Encoder (contexto final para o gerador) ---")
    print("Carregando Cross-Encoder (pode demorar na primeira vez)...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    top3 = rerank_with_cross_encoder(cross_encoder, user_query, [int(i) for i in ids.tolist()], MANUAL_FRAGMENTS)
    for rank, (doc_id, ce_score, full_text) in enumerate(top3, start=1):
        print(f"{rank}. id={doc_id}  cross_encoder_score={ce_score:.4f}")
        print(full_text)
        print()


if __name__ == "__main__":
    main()
