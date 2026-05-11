# Laboratório 9 — RAG com HNSW, HyDE e Cross-Encoder (local)

Implementação em Python do roteiro do laboratório: corpus simulado (≥20 fragmentos), embeddings com `sentence-transformers`, índice **FAISS HNSW** com `M` e `ef_construction` explícitos, **HyDE** via **Ollama** (LLM local), recuperação **top-10** e re-ranking com **Cross-Encoder** `cross-encoder/ms-marco-MiniLM-L-6-v2` (**top-3**).

## Análise: HNSW vs KNN exato e uso de RAM

No **KNN exato** clássico (ex.: força bruta ou estruturas que guardam todos os vetores para comparar distância a cada consulta), o consumo de RAM costuma crescer sobretudo com o **armazenamento da matriz de embeddings** \(N \times d\) e, em implementações ingênuas, com estruturas auxiliares mínimas. Cada busca pode exigir muitas distâncias completas.

O **HNSW** (Hierarchical Navigable Small World) mantém um **grafo aproximado** em várias camadas: cada nó tem até **M** arestas (vizinhos) por camada em média. **Aumentar M** engrossa o grafo: mais ponteiros e metadados por vetor, o que **eleva a RAM** (e tende a melhorar recall/latência de busca). O parâmetro **ef_construction** controla o tamanho da lista dinâmica durante a **inserção** no índice: valores maiores exploram mais candidatos ao ligar cada ponto, gerando **grafos mais “bem conectados”** à custa de **mais RAM temporária na construção** e **indexação mais lenta**; em alguns casos o grafo final também fica mais denso, com impacto adicional na memória residente.

Em resumo: frente a um KNN exato que pode ser CPU-intensivo na query, o HNSW troca parte do problema por **estrutura de grafo em memória** governada principalmente por **M** e pelo processo de construção (**ef_construction**), com **RAM** que cresce com a conectividade do grafo, não apenas com \(N \times d\).

## Declaração obrigatória (Contrato Pedagógico)

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Guiri.

*(Substitua **Guiri** pelo seu nome completo se a disciplina exigir nome civil na entrega.)*

## Como rodar localmente

1. **Python 3.10+** recomendado.
2. Crie o ambiente e instale dependências:

```powershell
cd caminho\para\lab9
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. **Ollama** (LLM local para o HyDE): instale em [https://ollama.com](https://ollama.com), suba o serviço e baixe um modelo compatível, por exemplo:

```powershell
ollama pull llama3.2
```

O script usa por padrão `OLLAMA_MODEL=llama3.2` e `OLLAMA_HOST=http://127.0.0.1:11434` (ajuste com variáveis de ambiente se necessário).

4. Execute o pipeline (na primeira vez o Hugging Face baixa os pesos do bi-encoder e do cross-encoder):

```powershell
python rag_pipeline.py
```

Outra pergunta coloquial:

```powershell
python rag_pipeline.py --query "fiquei tonto ao virar na cama de manha"
```

Ou via variável de ambiente: `set RAG_QUERY=...` (CMD) / `$env:RAG_QUERY="..."` (PowerShell).

**Saída esperada:** construção do índice; bloco **HyDE** com o documento hipotético; **Top 10** do HNSW; **Top 3** após o cross-encoder (trechos que iriam para o contexto do gerador).

## Entrega (GitHub + versão)

- Submeta o **link do repositório** na plataforma indicada pela disciplina.
- A versão a ser corrigida deve ter **tag ou release `v1.0`**. Após validar tudo:

```powershell
git tag v1.0
git push origin v1.0
```
