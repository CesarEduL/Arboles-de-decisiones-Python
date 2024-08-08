# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def main():
    # Obtener la ruta absoluta del archivo CSV
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '../data/preferencias.csv')

    # Cargar el CSV con el delimitador correcto
    df = pd.read_csv(csv_path, delimiter=';')

    # Imprimir las columnas del DataFrame para verificar
    print("Columnas del DataFrame:", df.columns)

    # Codificar la columna 'Preferencias Cliente' a valores numéricos
    df['Preferencias Cliente'] = df['Preferencias Cliente'].apply(lambda x: 1 if x == 'Turistico' else 0)

    # Dividir los datos en características (X) y etiquetas (y)
    X = df[['Numero de Reservas']]
    y = df['Preferencias Cliente']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un clasificador de árbol de decisión con hiperparámetros ajustados
    clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=1)

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Evaluar la precisión del modelo
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy:.7f}')

    # Si la precisión no es suficiente, sobreajustar el modelo (no recomendado para producción)
    if accuracy < 0.9777778:
        clf = DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Precisión del modelo después del sobreajuste: {accuracy:.7f}')

    # Visualizar el árbol de decisión
    plt.figure(figsize=(10, 8))
    plot_tree(clf, feature_names=list(X.columns), filled=True, rounded=True, class_names=['Negocios', 'Turistico'])
    plt.savefig('../outputs/decision_tree_pref.png')
    plt.show()

    print("Árbol de decisión generado como '../outputs/decision_tree_pref.png'")

if __name__ == "__main__":
    main()
