# Classificadores
Pacote com os classificadores utilizados e os métodos auxiliares

# Model
Cria um modelo de classificador exemplo de uso:

```python

    # Utilizando um Flow e um Loader
    loader = ImageLoader(
            'C:/Imagens Treinamento/',
            size=(150,150)
    )

    train_flow = loader.crop_flow_from_csv('annotation.csv')
    test_flow = loader.crop_flow_from_csv('test.csv')

    # Cria um Model a partir de um SVC
    mySvm = Model(svm.SVC())
    # Treina com o Flow
    mySvm.train_flow(train_flow)
    # Gera uma métrica de Acurácia básica
    print(mySvm.evaluate_flow(test_flow))

```