# GVCPUC
Repositório para o material, discussões, códigos, e outros do Grupo de Visão Computacional da PUCRS focado na pesquisa de Ambientes Indoor
**Rodney, e Daniel**

# Problema
O problema é detectar em qual "quadrante" da imagem se tem uma escada, ou uma porta, e gerar um feedback de saída. Exemplo:

**Há uma escada à sua direita.**

# Objetivo
Dividir e classificar os três quadrantes da imagem: 'esquerda', 'centro', e 'direita' em:
* Escada;
* Porta;
* Background / Negativo / Ambiente Interno.

Para isso é necessário um dataset anotado da seguinte maneira:
imagem, esquerda, centro, direita.
Exemplo: 

```
    door1.jpg, indoors, door, door
    door2.jpg, door, indoors, indoors
```

# MVP 1 - Datasets e Classificação
Inicialmente terá um classificador com SVM e algumas features(HoG, e talvez BoW), e a montagem do dataset.

O dataset de treinamento e validação consiste em imagens separadas com os três quadrantes, conforme mostrado anteriormente:

![](/images/test1.jpg)

Seria algo como: **door1.jpg, indoors, door, door** tendo em vista que há apenas portas nos quadrantes do centro, e da direita.

# MVP 2 - CNN e Reforço 
Para critério de comparação, será montada a partir dos modelos disponibilizados no [Keras](https://keras.io/applications/).
A camada final de classificação será re-treinada com o nosso dataset, e serão feitos testes e melhorias a partir desse modelo.

Esse MVP pode ser feito em paralelo com o primeiro, tendo em vista que o Daniel tá ajudando a trabalhar nisso.
Depois anotar um vídeo (frame a frame), conforme previsto no PEP, andando em algum local da FACIN, onde passe por escadas/portas, para gerar o dataset de testes.

# MVP 3 - Fazer as comparações dos resultados
A partir disso deverá ser otimizado o treinamento, a validação, e a precisão para comparação será avaliada no dataset de teste.

# Organização das Pastas
Os códigos ficam na pasta **SRC**.
A pasta **log** contém os arquivos de logs dos testes.
E a pasta **res** contém os recursos mais utilizados, como os modelos/redes salvas, imagens geradas, etc...

# Organização dos modulos(pacotes)
Os arquivos principais podem ficar no pacote princial(pasta **src**), e os demais ficam separados por modulos.

## Keras
Os códigos antigos utilizados nos testes pré esse projeto.

## General
Módulo com métodos e classes gerais, leitura de arquivos, processamento de imagem, geração de saidas, etc ... 

## Classifiers
Módulo com os classificadores (SVM, BOW, CNN, etc).

# Criador de Datasets

Utilizar o script em **src/create_dataset**, os argumentos são:
--dir (diretório das imagens): "C:/Imagens/*.jpg"
--output (o csv de saída): "C:/Imagens/output.csv"

Por exemplo: python create_dataset.py --dir "C:/Imagens/*.jpg" --ouput "C:/Imagens/output.csv" 

*Limitação: O output tem que ser um arquivo que exista(pode ser um arquivo vázio)*

O pacote Application contém scripts para a aplicação de visualização das pastas para criação do arquivo **csv** dos datasets.
O script **create_dataset.py** dentro do **src** contém o launcher dessa applicação.

No momento a função **launch** que é responsável por inicializar e montar a tela, e também gravar o arquivo, para isso ela aceita dois parâmetros:

