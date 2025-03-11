faq_list = [
    {"pergunta": "O que são abelhas?", "resposta": "As abelhas são insetos pertencentes à ordem Hymenoptera, conhecidos por seu papel na polinização de plantas e pela produção de mel e cera. Elas desempenham um papel crucial no ecossistema e na agricultura."},
    {"pergunta": "Por que as abelhas são importantes?", "resposta": "As abelhas são polinizadoras essenciais para muitas plantas, incluindo aquelas que fornecem alimentos como frutas, vegetais e nozes. Sem elas, a produção de alimentos seria significativamente reduzida."},
    {"pergunta": "Qual é a diferença entre abelhas e vespas?", "resposta": "Embora as abelhas e as vespas pertençam à mesma ordem (Hymenoptera), elas têm algumas diferenças. As abelhas geralmente têm corpos mais peludos e são especialistas em polinização. As vespas, por outro lado, têm corpos mais lisos, são carnívoras ou onívoras e não polinizam tanto quanto as abelhas."},
    {"pergunta": "As abelhas são agressivas?", "resposta": "As abelhas, em geral, não são agressivas. Elas picam apenas em defesa de sua colônia ou quando se sentem ameaçadas. A abelha só pica uma vez e morre depois de usar seu ferrão. Algumas espécies, como as abelhas africanas (abelhas assassinas), são mais defensivas, mas a maioria das abelhas não representa perigo se não for provocada."},
    {"pergunta": "Como as abelhas produzem mel?", "resposta": "As abelhas produzem mel a partir do néctar das flores, que elas coletam com suas línguas. O néctar é armazenado nas colmeias e, por meio de um processo de evaporação e transformação, se torna mel. As abelhas guardam o mel nas células de cera da colmeia, onde ele é usado como alimento."},
    {"pergunta": "Por que as abelhas estão desaparecendo?", "resposta": "As abelhas estão enfrentando várias ameaças, incluindo o uso excessivo de pesticidas, doenças, perda de habitat e mudanças climáticas. O desaparecimento das abelhas tem causado preocupações devido ao impacto na polinização e, consequentemente, na produção de alimentos."},
    {"pergunta": "Como podemos ajudar as abelhas?", "resposta": "Existem várias maneiras de ajudar as abelhas, como: - Plantar flores nativas e melíferas que atraem abelhas. - Evitar o uso de pesticidas em jardins e plantações. - Criar um ambiente amigável para as abelhas, como ter fontes de água e abrigo. - Apoiar a agricultura sustentável e a preservação do habitat natural das abelhas."},
    {"pergunta": "Qual é a expectativa de vida de uma abelha?", "resposta": "A expectativa de vida de uma abelha varia de acordo com a função dela na colônia. Uma abelha operária vive de algumas semanas a poucos meses, enquanto uma rainha pode viver de 3 a 5 anos."},
    {"pergunta": "O que é a colmeia?", "resposta": "A colmeia é a estrutura onde as abelhas vivem e trabalham. Ela é construída com cera produzida pelas abelhas e tem uma organização complexa, com diferentes compartimentos para armazenar mel, pólen e ovos. A colmeia também serve como o local onde a rainha coloca seus ovos."},
    {"pergunta": "O que é a polinização?", "resposta": "A polinização é o processo pelo qual as abelhas (e outros insetos ou animais) transferem pólen de uma flor para outra, permitindo a fecundação e a reprodução das plantas. Esse processo é crucial para a produção de muitas frutas e sementes."},
    {"pergunta": "O que é uma abelha rainha?", "resposta": "A abelha rainha é a fêmea reprodutora da colônia. Ela é a única abelha da colônia que põe ovos. Sua função principal é garantir a reprodução da colônia, e ela pode viver por vários anos."},
    {"pergunta": "O que é cera de abelha?", "resposta": "A cera de abelha é uma substância produzida pelas glândulas das abelhas operárias e usada para construir as células hexagonais da colmeia, onde elas armazenam mel, pólen e ovos. A cera de abelha também é amplamente utilizada em cosméticos, velas e outros produtos."},
    {"pergunta": "Como as abelhas comunicam-se entre si?", "resposta": "As abelhas comunicam-se principalmente por meio de danças e feromônios. A famosa 'dança das abelhas' é um comportamento complexo que permite que uma abelha indique a localização das flores ricas em néctar para as outras."},
    {"pergunta": "As abelhas produzem veneno?", "resposta": "Sim, as abelhas possuem um ferrão que, quando usado, libera veneno. Esse veneno é utilizado para defender a colônia contra ameaças. No entanto, a picada de abelha geralmente não é perigosa para a maioria das pessoas, exceto aquelas com alergia ao veneno."},
    {"pergunta": "As abelhas fazem mel durante todo o ano?", "resposta": "As abelhas produzem mel principalmente durante a primavera e o verão, quando as flores estão mais abundantes. Durante o outono e inverno, elas dependem do mel que armazenaram para se alimentar e sobreviver."}
]
"""faq_list.append({
    "pergunta": "Como podemos proteger as abelhas?",
    "resposta": "Para proteger as abelhas, é importante reduzir o uso de pesticidas, criar habitats seguros e plantar flores que forneçam néctar e pólen. Além disso, apoiar a apicultura sustentável é essencial para a preservação das colônias de abelhas."
})
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Inicializando o modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gerando embeddings para as perguntas e respostas
faq_embeddings = []
for item in faq_list:
    faq_embeddings.append(model.encode(item["pergunta"] + " " + item["resposta"]))

# Transformando em um array numpy
faq_embeddings = np.array(faq_embeddings)

# Mostrando a forma dos embeddings
print(faq_embeddings.shape)

# Pergunta a ser respondida
nova_pergunta = "Como podemos proteger as abelhas?"

# Gerando o embedding para a nova pergunta
nova_pergunta_embedding = model.encode(nova_pergunta)

from sklearn.metrics.pairwise import cosine_similarity

# Calculando a similaridade entre a nova pergunta e as perguntas/respostas do FAQ
similaridades = cosine_similarity([nova_pergunta_embedding], faq_embeddings)

# Encontrando o índice da maior similaridade
indice_mais_similar = np.argmax(similaridades)

# Exibindo a pergunta/resposta mais semelhante
pergunta_resposta_similar = faq_list[indice_mais_similar]
print(f"Pergunta mais similar: {pergunta_resposta_similar['pergunta']}")
print(f"Resposta: {pergunta_resposta_similar['resposta']}")
