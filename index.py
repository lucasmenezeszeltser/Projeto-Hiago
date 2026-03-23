from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import numpy as np
from openai import OpenAI
import os
import io

# Inicializa o servidor web
app = Flask(__name__)

# Configuração de CORS robusta: 
# Isso garante que os cabeçalhos de permissão sejam enviados mesmo em caso de erro 500.
CORS(app, resources={r"/*": {"origins": "*"}})

# A chave da API deve ser configurada como variável de ambiente no Render
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def dividir_texto(texto, tamanho=500):
    """Divide o texto em pedaços menores para processamento de embeddings."""
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def gerar_embedding(texto):
    """Gera o vetor numérico (embedding) para um texto usando o modelo da OpenAI."""
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
    """Rota para verificar se o backend está online."""
    return jsonify({"status": "online", "mensagem": "Backend está a funcionar!"}), 200

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
        # 2. Extração de Texto do PDF
        # Usamos BytesIO para ler o arquivo da memória de forma segura
        texto_completo = ""
        pdf_bytes = io.BytesIO(arquivo_pdf.read())
        
        with pdfplumber.open(pdf_bytes) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_completo += texto + "\n"
        
        if not texto_completo.strip():
            return jsonify({"erro": "Não foi possível extrair texto legível do PDF"}), 422

        # 3. Processamento RAG (Busca por Contexto)
        chunks = dividir_texto(texto_completo)
        embeddings = [gerar_embedding(chunk) for chunk in chunks]
        embedding_pergunta = gerar_embedding(pergunta)

        # 4. Cálculo de Similaridade
        scores = []
        for i in range(len(embeddings)):
            score = similaridade(embedding_pergunta, embeddings[i])
            scores.append((score, chunks[i]))

        # Ordenar e pegar os 3 trechos mais relevantes
        scores_ordenados = sorted(scores, reverse=True, key=lambda x: x[0])
        top_3 = scores_ordenados[:3]
        
        contexto = ""
        for score, texto in top_3:
            contexto += texto + "\n\n"

        # 5. Geração da Resposta com GPT-4o-mini
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
        # Importante: Imprime o erro real nos logs do Render para debug
        print(f"ERRO CRÍTICO NO BACKEND: {str(e)}")
        return jsonify({"erro": f"Erro interno no processamento: {str(e)}"}), 500

if __name__ == '__main__':
    # O Render define a porta automaticamente através da variável de ambiente PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)