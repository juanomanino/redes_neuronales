import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from pandas import DataFrame


def entrenar():
    print("Comezando entrenamiento...")
    historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
    print("Modelo entrenado!")
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])
    label = tk.Label(ventana, text="Graph Page!", font="LARGE_FONT")
    label.pack(pady=10,padx=10)

    f = Figure(figsize=(5,5), dpi=100)
    a = f.add_subplot(111)
    a.plot(historial.history["loss"])
    a.set_title('Resultados del Entrenamiento')
    a.set_xlabel('# Época')
    a.set_ylabel('Magnitud de Pérdida')


        
    canvas = FigureCanvasTkAgg(f, ventana)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, ventana)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    print("Hagamos una predicción!")
    resultado=modelo.predict([0.0])
    print("El resultado es"+str(resultado)+"fahrenheit!")


ventana=Tk()
ventana.geometry("800x500")
ventana.config(bg="snow")


celsius=np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit=np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# capa =tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo=tf.keras.Sequential([capa])
oculta1=tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2=tf.keras.layers.Dense(units=3)
salida=tf.keras.layers.Dense(units=1)
modelo=tf.keras.Sequential([oculta1, oculta2, salida])


modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)


bt_sumar = Button(ventana, width=2, height=2, command=entrenar)
bt_sumar.place(x=150, y=450)


ventana.mainloop()