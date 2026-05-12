# Lab 9 — busca em manual com RAG (HNSW + HyDE + cross-encoder)

Sou o Guilherme Ruben. Esse repo é o laboratório 9: eu montei um pipeline em Python que finge um assistente em cima de trechos fictícios de manual médico. A ideia do PDF é mostrar onde a similaridade “burra” falha (pergunta de paciente vs jargão) e como dá pra melhorar com HyDE, índice rápido e um rerank mais forte.

O que mais me deu trabalho na prática foi encaixar tudo no mesmo script (`rag_pipeline.py`) sem depender de API paga: embeddings e cross-encoder rodam com `sentence-transformers`, o HyDE chama o Ollama na máquina, e o índice é FAISS com HNSW.

## Declaração (obrigatória pelo contrato)

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Guilherme Ruben.

## HNSW: memória vs KNN exato (resposta curta do exercício)

No KNN exato você basicamente guarda os vetores e compara a query com muitos deles; RAM sobe com a matriz N×d e a busca pode ficar pesada em CPU.

No HNSW você mantém um grafo aproximado. O `M` controla quantos vizinhos cada ponto tende a ter: `M` maior costuma melhorar recall, mas aumenta estrutura em memória (mais arestas/ponteiros). O `ef_construction` mexe na qualidade do grafo na hora de construir: maior explora mais candidatos ao inserir, usa mais RAM na construção e deixa a indexação mais lenta, e o grafo pode ficar mais “cheio”.

Ou seja: você troca parte do custo de query do KNN exato por um custo fixo de grafo em RAM, regulado principalmente por `M` e pelo processo de construção.

## Como eu rodo aqui (Windows)

Precisa do Python, da venv, do Ollama aberto e do modelo baixado.

```powershell
cd C:\Users\guiri\OneDrive\Documentos\GitHub\lab9
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2
python rag_pipeline.py
```

Outra pergunta:

```powershell
python rag_pipeline.py --query "fiquei tonto ao virar na cama de manha"
```

Dicas rápidas: na primeira vez o Hugging Face baixa pesos (bi-encoder e cross-encoder), então demora. Se der erro de conexão na porta 11434, o Ollama não está rodando ou o modelo não existe — confere com `ollama list`.

## Entrega

Subo o link no que a disciplina pedir. A versão pra correção vai com tag `v1.0`:

```powershell
git tag v1.0
git push origin v1.0
```
