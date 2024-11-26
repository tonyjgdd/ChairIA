from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, db
import pickle
from Orange.data import Table, Domain, ContinuousVariable
import logging
import numpy as np
from enum import Enum

class Clasific(Enum):
    P1 = 0
    P2 = 1
    P3 = 2
    P4 = 3
    P5 = 4
    P6 = 5

# Inicializa la app de Firebase
cred = credentials.Certificate('t4movil-firebase-adminsdk-p7geo-be116f1db8.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://t4movil-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)

# Cargar el modelo entrenado
with open('NB.pkcls', 'rb') as model_file:
    model = pickle.load(model_file)



# Normalización de datos
def normalize_value(value, min_val=0, max_val=3.300000):
    return (value - min_val) / (max_val - min_val)

def normalize_data(data):
    return [round(normalize_value(value), 6) for value in data]

@app.route('/usuarios/<user_id>', methods=['PUT'])
def update_usuario(user_id):
    ref = db.reference('USUARIO')
    usuario_data = ref.child(user_id).get()
    
    if usuario_data is None:
        return jsonify({'error': 'Usuario no encontrado'}), 404

    # Normalizar los datos del usuario
    normalized_data = normalize_data([
        usuario_data['ASIENT_BAJO'],
        usuario_data['ASIENT_DEREC'],
        usuario_data['ASIENT_IZQ'],
        usuario_data['BRAZ_DEREC'],
        usuario_data['BRAZ_IZQ'],
        usuario_data['ESPALDA_ALTA'],
        usuario_data['ESPALDA_BAJA']
    ])

    # Construir final_data con CABECERA en la posición correcta
    final_data = normalized_data[:5] + [float(usuario_data['CABECERA'])] + normalized_data[5:]

    # Crear dominio y tabla
    domain = Domain([ContinuousVariable(name) for name in 
                     ['ASIENT_BAJO', 'ASIENT_DEREC', 'ASIENT_IZQ', 
                      'BRAZ_DEREC', 'BRAZ_IZQ', 'CABECERA', 
                      'ESPALDA_ALTA', 'ESPALDA_BAJA']])

    # Crear un objeto Table de Orange
    datos_usuario = Table.from_list(domain, [final_data])
    print("Datos1",final_data)
    print("Datos2",datos_usuario)

    # Realizar la predicción
    prediction = model(datos_usuario)
    print("Datos3",prediction)

    # Manejo de la predicción
    if isinstance(prediction, (np.ndarray)):
        new_classification = int(prediction[0])  # Obtén el primer valor
    elif isinstance(prediction, (np.integer, np.int64)):
        new_classification = int(prediction)
    else:
        new_classification = prediction  # Asumimos que es un valor serializable

    try:
        # Actualizar la clasificación en Firebase
        ref.child(user_id).update({"CLASIFICACION": Clasific(new_classification).name})
        return jsonify({'success': True, 'message': 'Clasificación actualizada exitosamente', 'Clasificacion': Clasific(new_classification).name}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
