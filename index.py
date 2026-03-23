from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import numpy as np
from openai import OpenAI
import os

# Inicializa o servidor web
app = Flask(__name__)

# ✅ CORS SIMPLIFICADO (corrigido)
CORS(app)

# A chave da API deve ser configurada como variável de ambiente no Render
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def dividir_texto(texto, tamanho=500):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def gerar_embedding(texto):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding

def similaridade(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ROTA DE HEALTH CHECK
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "mensagem": "Backend está a funcionar!"}), 200

@app.route('/analisar', methods=['POST'])
def analisar_pdf():

    # Verifica se o arquivo e a pergunta foram enviados
    if 'pdf' not in request.files or 'pergunta' not in request.form:
        return jsonify({"erro": "PDF ou pergunta não encontrados"}), 400
    
    arquivo_pdf = request.files['pdf']
    pergunta = request.form['pergunta']
    
    if arquivo_pdf.filename == '' or not pergunta:
        return jsonify({"erro": "Arquivo vazio ou pergunta em branco"}), 400

    try:
        # 1. Extrair Texto do PDF
        texto_completo = ""
        with pdfplumber.open(arquivo_pdf) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_completo += texto + "\n"

        # 2. Dividir Texto
        chunks = dividir_texto(texto_completo)

        # 3. Gerar Embeddings
        embeddings = [gerar_embedding(chunk) for chunk in chunks]
        embedding_pergunta = gerar_embedding(pergunta)

        # 4. Similaridade
        scores = []
        for i in range(len(embeddings)):
            score = similaridade(embedding_pergunta, embeddings[i])
            scores.append((score, chunks[i]))

        # 5. Pegar Top 3
        scores_ordenados = sorted(scores, reverse=True, key=lambda x: x[0])
        top_3 = scores_ordenados[:3]
        
        contexto = ""
        for score, texto in top_3:
            contexto += texto + "\n\n"

        # 6. Gerar Resposta com LLM (RAG)
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Responda apenas com informações presentes no texto. Se possível, cite exemplos mencionados no contexto."
                },
                {
                    "role": "user", 
                    "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}\n\nResponda de forma clara e objetiva:"
                }
            ]
        )

        return jsonify({"resposta": resposta.choices[0].message.content})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)