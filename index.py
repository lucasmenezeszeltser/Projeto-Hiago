import pdfplumber

caminho_pdf = "data/pdf/Artigo-cientifico-Vitoria.pdf"

texto_completo = ""

with pdfplumber.open(caminho_pdf) as pdf:
    for pagina in pdf.pages:
        texto = pagina.extract_text()
        if texto:
            texto_completo += texto + "\n"

# salva em arquivo com UTF-8 (resolve problema de acento)
with open("saida.txt", "w", encoding="utf-8") as arquivo:
    arquivo.write(texto_completo)

print("Arquivo salvo com sucesso!")