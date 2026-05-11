# Laboratório 9 — RAG com HNSW, HyDE e Cross-Encoder (local)

## Análise: HNSW vs KNN exato e uso de RAM

No **KNN exato** clássico (ex.: força bruta ou estruturas que guardam todos os vetores para comparar distância a cada consulta), o consumo de RAM costuma crescer sobretudo com o **armazenamento da matriz de embeddings** \(N \times d\) e, em implementações ingênuas, com estruturas auxiliares mínimas. Cada busca pode exigir muitas distâncias completas.

O **HNSW** (Hierarchical Navigable Small World) mantém um **grafo aproximado** em várias camadas: cada nó tem até **M** arestas (vizinhos) por camada em média. **Aumentar M** engrossa o grafo: mais ponteiros e metadados por vetor, o que **eleva a RAM** (e tende a melhorar recall/latência de busca). O parâmetro **ef_construction** controla o tamanho da lista dinâmica durante a **inserção** no índice: valores maiores exploram mais candidatos ao ligar cada ponto, gerando **grafos mais “bem conectados”** à custa de **mais RAM temporária na construção** e **indexação mais lenta**; em alguns casos o grafo final também fica mais denso, com impacto adicional na memória residente.

Em resumo: frente a um KNN exato que pode ser CPU-intensivo na query, o HNSW troca parte do problema por **estrutura de grafo em memória** governada principalmente por **M** e pelo processo de construção (**ef_construction**), com **RAM** que cresce com a conectividade do grafo, não apenas com \(N \times d\).

---

*(Nos próximos commits: HyDE, busca top-10, cross-encoder, declaração de IA e instruções de execução.)*
