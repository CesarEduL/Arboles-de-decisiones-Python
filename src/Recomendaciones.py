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
    csv_path = os.path.join(
        current_dir, '../data/Recomendaciones_DataSets.csv')

    # Cargar el CSV con el delimitador correcto
    df = pd.read_csv(csv_path, delimiter=',')

    # Codificar las columnas categóricas a valores numéricos
    df['Preferencias_destino'] = df['Preferencias_destino'].astype(
        'category').cat.codes
    df['Preferencias_actividades'] = df['Preferencias_actividades'].astype(
        'category').cat.codes

    # Guardar los nombres originales de las clases
    class_names = df['Recomendacion_paquete'].astype(
        'category').cat.categories.tolist()
    df['Recomendacion_paquete'] = df['Recomendacion_paquete'].astype(
        'category').cat.codes

    # Seleccionar características relevantes para X (puedes ajustar esto según tus necesidades)
    X = df[['Preferencias_destino', 'Preferencias_actividades',
            'Presupuesto_maximo', 'Historial_viajes', 'Calificaciones_previas']]
    y = df['Recomendacion_paquete']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Crear un clasificador de árbol de decisión con hiperparámetros ajustados
    clf = DecisionTreeClassifier(
        random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=1)

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Visualizar el árbol de decisión
    plt.figure(figsize=(10, 8))
    plot_tree(clf, feature_names=list(X.columns), filled=True,
              rounded=True, class_names=[str(cls) for cls in class_names])
    plt.title("Árbol de decisión para recomendaciones de paquetes de viaje")
    plt.savefig('../outputs/decision_tree_recomendacion.png')
    plt.show()

    print("Árbol de decisión generado como '../outputs/decision_tree_recomendacion.png'")

    # Histograma de Preferencias de Destino con colores asignados
    plt.figure(figsize=(10, 6))
    # Definir los destinos según la codificación
    destinos = ['América', 'Asia', 'Europa']
    # Definir colores para cada destino
    colors = ['skyblue', 'lightgreen', 'salmon']

    # Crear el histograma con colores personalizados
    df['Preferencias_destino'].value_counts().plot(kind='bar', color=colors)
    plt.title('Histograma de Preferencias de Destino')
    plt.xlabel('Destinos')
    plt.ylabel('Cantidad')
    plt.xticks(ticks=range(len(destinos)), labels=destinos, rotation=0)
    plt.tight_layout()
    plt.savefig('../outputs/histograma_recomendacion.png')
    plt.show()

    print("Histograma de Preferencias de Destino generado como '../outputs/histograma_recomendacion.png'")

    # Matriz de Confusión
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    plt.figure(figsize=(8, 6))
    # Usar 'Blues' como colormap válido
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.tight_layout()
    plt.savefig('../outputs/matriz_confusion_recomendacion.png')
    plt.show()

    # Métricas de Evaluación
    print(classification_report(y_test, clf.predict(
        X_test), target_names=class_names))

    # Mostrar el DataFrame de las predicciones
    predictions_df = pd.DataFrame(
        {'Recomendacion_paquete_Real': y_test, 'Recomendacion_paquete_Predicho': clf.predict(X_test)})
    print("\nDataFrame de las predicciones:")
    print(predictions_df)


if __name__ == "__main__":
    main()
