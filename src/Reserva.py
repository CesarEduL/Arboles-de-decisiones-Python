import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os


def main():
    # Obtener la ruta absoluta del archivo CSV
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '../data/Reserva_DataSets.csv')

    # Cargar el CSV con el delimitador correcto
    df = pd.read_csv(csv_path)

    # Convertir las columnas categóricas a variables dummy (one-hot encoding)
    df = pd.get_dummies(
        df, columns=['Destino', 'Tipo_Paquete', 'Disponibilidad_Paquetes'])

    # Seleccionar características relevantes para X
    X = df.drop(['Reserva_ID', 'Fecha_Reserva', 'Tipo_Paquete_tour', 'Tipo_Paquete_vuelo',
                'Tipo_Paquete_hotel'], axis=1)  # Excluir columnas no relevantes para el modelo

    # Identificar la columna objetivo para y ('Tipo_Paquete_tour', 'Tipo_Paquete_vuelo', o 'Tipo_Paquete_hotel')
    y_col = 'Tipo_Paquete'  # Ajustar según el tipo de paquete que se desea predecir
    y = df[[col for col in df.columns if col.startswith('Tipo_Paquete')]].idxmax(
        axis=1).str.replace('Tipo_Paquete_', '')

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Crear un clasificador de árbol de decisión con hiperparámetros ajustados
    clf = DecisionTreeClassifier(
        random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=1)

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Visualizar el árbol de decisión
    plt.figure(figsize=(12, 10))
    plot_tree(clf, feature_names=list(X.columns), filled=True,
              rounded=True, class_names=[str(cls) for cls in clf.classes_])
    plt.title("Árbol de decisión para tipos de paquete de viaje más reservados")
    plt.savefig('../outputs/decision_tree_reserva_viajes.png')
    plt.show()

    print("Árbol de decisión generado como '../outputs/decision_tree_reserva_viajes.png'")

    # Crear y guardar el histograma de las predicciones del modelo con colores asignados
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(clf.predict(X_test), return_counts=True)
    colors = ['skyblue', 'lightgreen', 'salmon']
    plt.bar(unique, counts, edgecolor='k', alpha=0.7, color=colors)
    plt.xlabel('Tipos de Paquete Predichos')
    plt.ylabel('Frecuencia de Predicciones')
    plt.title(
        'Histograma de las Predicciones del Modelo de Árbol de Decisión para Reservas')
    plt.tight_layout()
    plt.savefig('../outputs/prediction_histogram_reserva_viajes.png')
    plt.show()

    print("Histograma de las predicciones generado como '../outputs/prediction_histogram_reserva_viajes.png'")

    # Matriz de Confusión
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest',
               cmap='Blues')  # Corregido aquí
    plt.title('Matriz de Confusión')
    plt.colorbar()
    tick_marks = np.arange(len(clf.classes_))
    plt.xticks(tick_marks, [str(cls) for cls in clf.classes_], rotation=45)
    plt.yticks(tick_marks, [str(cls) for cls in clf.classes_])
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.tight_layout()
    plt.savefig('../outputs/matriz_confusion_reserva.png')
    plt.show()

    # Métricas de Evaluación
    print(classification_report(y_test, clf.predict(X_test),
          target_names=[str(cls) for cls in clf.classes_]))

    # Mostrar el DataFrame de las predicciones
    predictions_df = pd.DataFrame(
        {'Tipo_Paquete_Real': y_test, 'Tipo_Paquete_Predicho': clf.predict(X_test)})
    print("\nDataFrame de las predicciones:")
    print(predictions_df)


if __name__ == "__main__":
    main()
