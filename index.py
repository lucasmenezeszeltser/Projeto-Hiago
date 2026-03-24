from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import numpy as np
from openai import OpenAI
import os
import io

# Inicializa o servidor web
app = Flask(__name__)

# Configuração de CORS robusta para permitir conexões do seu front-end
CORS(app, resources={r"/*": {"origins": "*"}})

# A chave da API deve estar nas variáveis de ambiente do Render
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def dividir_texto(texto, tamanho=500):
    """Divide o texto em pedaços menores para processamento."""
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def gerar_embedding(texto):
    """Gera o vetor numérico (embedding) para um único trecho."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding

def similaridade(v1, v2):
    """Calcula a similaridade de cosseno entre dois vetores."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "mensagem": "Backend otimizado está online!"}), 200

@app.route('/analisar', methods=['POST'])
def analisar_pdf():
    # 1. Validação de Entrada
    if 'pdf' not in request.files or 'pergunta' not in request.form:
        return jsonify({"erro": "PDF ou pergunta não encontrados"}), 400
    
    arquivo_pdf = request.files['pdf']
    pergunta = request.form['pergunta']
    
    if arquivo_pdf.filename == '' or not pergunta:
        return jsonify({"erro": "Arquivo vazio ou pergunta em branco"}), 400

    try:
        # 2. Extração de Texto com Otimização de Memória
        texto_completo = ""
        pdf_bytes = io.BytesIO(arquivo_pdf.read())
        
        with pdfplumber.open(pdf_bytes) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_completo += texto + "\n"
                # Otimização: Limpa o cache da página após extrair o texto
                pagina.flush_cache()
        
        if not texto_completo.strip():
            return jsonify({"erro": "Não foi possível extrair texto legível do PDF"}), 422

        # 3. Processamento RAG Otimizado (Evita estourar a RAM)
        chunks = dividir_texto(texto_completo)
        
        # Geramos o embedding da pergunta primeiro
        embedding_pergunta = gerar_embedding(pergunta)

        # Em vez de gerar todos os embeddings e guardar numa lista (que ocupa muita RAM),
        # calculamos a similaridade um por um e guardamos apenas o score e o texto.
        scores = []
        for chunk in chunks:
            try:
                emb_chunk = gerar_embedding(chunk)
                score = similaridade(embedding_pergunta, emb_chunk)
                scores.append((score, chunk))
            except Exception:
                continue # Ignora chunks que falharem para não parar o processo

        # 4. Seleção do Contexto
        # Ordenar e pegar os 3 trechos mais relevantes
        scores_ordenados = sorted(scores, reverse=True, key=lambda x: x[0])
        top_3 = scores_ordenados[:3]
        
        contexto = ""
        for score, texto in top_3:
            contexto += texto + "\n\n"

        # 5. Geração da Resposta
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Responda apenas com informações presentes no texto fornecido. Se possível, cite exemplos mencionados no contexto."
                },
                {
                    "role": "user", 
                    "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}\n\nResponda de forma clara e objetiva:"
                }
            ]
        )

        return jsonify({"resposta": resposta.choices[0].message.content})

    except Exception as e:
        print(f"ERRO CRÍTICO NO BACKEND: {str(e)}")
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)