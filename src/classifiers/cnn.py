from keras.applications.inception_v3 import InceptionV3
from model import Model

def __inception():
    inception = InceptionV3(weights='imagenet', include_top=False)

    return inception

# Só vou montar essa classe pra ser como um "Wrapper"
# Pra padronizar melhor os códigos nos arquivos de testes / main
def KerasModel(Model):
    def __init__(self, classifier):
        super(Model, self).__init__(classifier)
        # Não sou àvido usuário do Keras, então não sei quantas
        # vezes é necessário chamar o Compile
        self.compiled = False

    def train(self, x, y):
        self.__build()
        self.classifier.fit(x,y)
    
    def train_flow(self, flow):
        self.__build()
        self.classifier.fit_generator(self.__generator(flow))
    
    def predict(self, x):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def evaluate(self, x, y):
        pass
    
    def evaluate_flow(self, flow):
        pass
    
    def __build(self):
        # Método pra chamar o Compile
    
    def __generator(self, flow):
        # Método para montar um ImageGenerator
        # E chamar o Flow com X/Y do Flow

def Inception(KerasModel):
    def __init__(self):
        super(Model, self).__init__(__inception())

