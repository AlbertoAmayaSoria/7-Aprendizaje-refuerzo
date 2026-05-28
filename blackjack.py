"""
El problema del blackjack simplificado como un problema de aprendizaje por refuerzo
git@github.com:AlbertoAmayaSoria/IA_class_repository.git
"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import choice, random
import itertools

class BlackJack(MDPsim):
    """
    Clase que representa un MDP para el problema del jugador.
    
    El jugador tiene un capital inicial y el objetivo es llegar a un capital
    objetivo o quedarse sin dinero.
    
    """
    def __init__(self, gama = 0.9):
        self.gama = gama
        self.mazo = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 10, 10, 10]
        self.estados = [
            (suma, carta_c, usable)
            for suma, carta_c, usable in itertools.product(
                range(12, 22),  #suma jugador: 12-21
                range(1, 11),   #carta visible crupier: 1-10
                [False, True]   #as usable
            )
        ]
        self.ultima_accion = None
        
    def _reparte_carta(self):
        return choice(self.mazo)
    
    def _calcular_mano(self, cartas):
        suma = sum(cartas)
        usable = False
        ases = cartas.count(1)

        #intentamos contar as como 11
        if ases > 0 and suma + 10 <= 21:
            suma += 10
            usable = True

        #si se pasa lo contamos como 1
        while suma > 21 and usable:
            suma -= 10
            usable = False

        return suma, usable

    def estado_inicial(self):
        self.ultima_accion = None
        self.jugador = [self._reparte_carta(), self._reparte_carta()]
        self.crupier = [self._reparte_carta(), self._reparte_carta()]
        suma_j, usable_j = self._calcular_mano(self.jugador)
        while suma_j < 12:                 # garantiza estado válido
            self.jugador.append(self._reparte_carta())
            suma_j, usable_j = self._calcular_mano(self.jugador)
        return (suma_j, self.crupier[0], usable_j)
    
    def acciones_legales(self, s):
        return [0, 1] #0: quedarse, 1:pedir 
    
    def transicion(self, s, a):
        self.ultima_accion = a
        if a == 1:
            self.jugador.append(self._reparte_carta())
            suma_j, usable_j = self._calcular_mano(self.jugador)
            return (suma_j, s[1], usable_j)
        return (22, s[1], s[2])     #22 es siempre terminal

    def recompensa(self, s, a, s_):
        if a == 1:
            suma_j, _, _ = s_
            if suma_j > 21:
                return -1
            return 0    #carta recibida sin bust
        
        # Turno del crupier
        suma_j, _, _ = s
        crupier_mano = self.crupier.copy()
        suma_c, _ = self._calcular_mano(crupier_mano)
        while suma_c < 17:
            crupier_mano.append(choice(self.mazo))
            suma_c, _ = self._calcular_mano(crupier_mano)
        if suma_c > 21 or suma_j > suma_c:
            return 1
        elif suma_j == suma_c:
            return 0
        else:
            return -1


    def es_terminal(self, s):
        return s[0] > 21 #or s[0] == 22
        


if __name__ == "__main__":

    blackjack = BlackJack(gama=0)

    # TODO: definir los parámetros de SARSA y Q-learning, luego crear las instancias 
    # de cada algoritmo
    q_sarsa = SARSA( blackjack, alfa=0.1, epsilon=0.1, n_ep=500_000, n_iter=100)
    q_learning = Q_learning( blackjack, alfa=0.1, epsilon=0.1, n_ep=500_000, n_iter=100)

    # Encuentra las políticas óptimas para cada algoritmo
    pi_s = PoliticaGreedy(q_sarsa)
    pi_q = PoliticaGreedy(q_learning)

    # Imprime las políticas óptimas para cada estado no terminal
    print("Estado".center(20) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
    print("-"*20 + '|' + "-"*10 + '|' + "-"*10)
    for s in blackjack.estados:
        if not blackjack.es_terminal(s):
            print(str(s).center(20) + '|' 
                  + str(pi_s(s)).center(10) + '|' 
                  + str(pi_q(s)).center(10))
    print("-"*20 + '|' + "-"*10 + '|' + "-"*10)


"""
****************************************************************************************
Responde las siguientes preguntas:

1. ¿Cuáles son los estados, acciones, recompensas y transiciones en el problema del
blackjack?
    Los estados son tuplas (suma_jugador, carta_visible_crupier, as_usable).
    suma_jugador está en el rango de 12-21, ya que debajo de 12 siempre vale la pena pedir.
    carta_visible_crupier de 1-10.
    as_usable es un booleano.
    El espacio de estados es de 200 (10x10x2).

2. ¿Cómo se pueden representar los estados del blackjack de manera eficiente para el
aprendizaje por refuerzo?
    (suma, carta_crupier, as_usable), ya que en el blackjack no es necesario guardar
    un historial de cómo se llegó a un punto, sino que necesitamos encontrar la acción
    óptima para realizar en cada estado.

3. ¿Qué pasa si se modifica el valor de épsilon de la política epsilon-greedy?
    El épsilon controla qué tan probable es que se explore una acción aleatoria.

4. ¿Cómo afecta el valor de alfa en la convergencia de los algoritmos SARSA y Q-learning?
    Mientras menor sea, mejor es el aprendizaje a largo plazo; si este sube, sobreescribe
    mucho lo aprendido, pero si se baja demasiado, puede ser muy lenta la convergencia.
    Un alfa alto causa oscilaciones, mientras que uno bajo permite promediar mejor las
    recompensas.

5. ¿Cuál de los dos algoritmos, SARSA o Q-learning, consideras que es más adecuado para
el problema del blackjack y por qué?
    Q-learning, ya que al ser off-policy, aprende de la política óptima
    independientemente de las acciones que toma durante el entrenamiento, por lo cual,
    aunque explore con épsilon, actualiza Q hacia el máximo posible, lo que lleva a una
    política más agresiva y óptima.
    Q-learning toma más riesgos cuando tiene un as jugable en promedio que SARSA.

6. ¿Se puede explicar con cierta lógica del juego la política óptima encontrada por cada
algoritmo? ¿Qué acciones se toman en cada estado y por qué?
    Sí.
    Con sumas altas, ambos algoritmos eligen no arriesgarse, plantándose casi siempre ya
    que no vale la pena arriesgarse cuando solamente 2 o 3 cartas mejoran la mano.
    Con sumas bajas piden casi siempre, ya que es menos probable pasarse de 21 casi con
    cualquier carta.
    Donde es más interesante es en un punto medio de los valores de la suma, ya que
    el quedarse o tomar varía más dependiendo de si está o no jugable el as y de qué
    carta tiene a la vista el crupier.
    
****************************************************************************************
"""
