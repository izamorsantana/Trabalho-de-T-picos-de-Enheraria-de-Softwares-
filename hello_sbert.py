from sentence_transformers import SentenceTransformer

# 1. Carregar um modelo de Transformador de Frases pré-treinado
model = SentenceTransformer("all-MiniLM-L6-v2")

# As frases para codificar
sentences = [
    "O que são abelhas?",
    "As abelhas são insetos pertencentes à ordem Hymenoptera, ",
    "conhecidos por seu papel na polinização de plantas e pela ",
    "produção de mel e cera.Elas desempenham um papel crucial ",
    "no ecossistema e na agricultura.",

     "Por que as abelhas são importantes?",
    "As abelhas são polinizadoras essenciais para muitas plantas,",
    "incluindo aquelas que fornecem alimentos como frutas, vegetais e nozes.",
    "Sem elas, a produção de alimentos seria significativamente reduzida.",

    "Qual é a diferença entre abelhas e vespas? ",
    "Embora as abelhas e as vespas pertençam à mesma ordem(Hymenoptera),",
    "elas têm algumas diferenças.As abelhas geralmente têm corpos mais peludos",
    "e são especialistas em polinização.As vespas, por outro lado,",
    " têm corpos mais lisos, são carnívoras ou onívoras"
    "e não polinizam tanto quanto as abelhas."
]

# 2. Calcular embeddings chamando model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calcular as similaridades de incorporação
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])