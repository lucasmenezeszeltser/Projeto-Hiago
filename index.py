import pdfplumber
import numpy as np
from openai import OpenAI

client = OpenAI()

caminho_pdf = "data/pdf/Artigo-cientifico-Vitoria.pdf"

# 1. EXTRAIR TEXTO
texto_completo = ""

with pdfplumber.open(caminho_pdf) as pdf:
    for pagina in pdf.pages:
        texto = pagina.extract_text()
        if texto:
            texto_completo += texto + "\n"

print("Texto extraído!")

# 2. DIVIDIR TEXTO
def dividir_texto(texto, tamanho=500):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

chunks = dividir_texto(texto_completo)

print(f"Total de chunks: {len(chunks)}")

# 3. GERAR EMBEDDINGS
def gerar_embedding(texto):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding

embeddings = []

for i, chunk in enumerate(chunks):
    print(f"Gerando embedding {i+1}/{len(chunks)}...")
    vetor = gerar_embedding(chunk)
    embeddings.append(vetor)

print("Embeddings gerados!")

# 4. SIMILARIDADE
def similaridade(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 5. PERGUNTA
pergunta = input("\nDigite sua pergunta: ")

embedding_pergunta = gerar_embedding(pergunta)

# 6. CALCULAR SCORES
scores = []

for i in range(len(embeddings)):
    score = similaridade(embedding_pergunta, embeddings[i])
    scores.append((score, chunks[i]))

# 7. PEGAR TOP 3
scores_ordenados = sorted(scores, reverse=True, key=lambda x: x[0])
top_3 = scores_ordenados[:3]

print("\n🔎 Top 3 trechos mais relevantes:\n")

contexto = ""

for i, (score, texto) in enumerate(top_3):
    print(f"\n--- Trecho {i+1} (score: {score:.4f}) ---\n")
    print(texto)
    contexto += texto + "\n\n"

# 8. GERAR RESPOSTA COM LLM (RAG)
resposta = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "system",
            "content": "Responda apenas com informações presentes no texto. Se possível, cite exemplos mencionados no contexto."
        },
        {
            "role": "user",
            "content": f"""
Pergunta: {pergunta}

Contexto:
{contexto}

Responda de forma clara e objetiva:
"""
        }
    ]
)

print("\n💬 Resposta final:\n")
print(resposta.choices[0].message.content)