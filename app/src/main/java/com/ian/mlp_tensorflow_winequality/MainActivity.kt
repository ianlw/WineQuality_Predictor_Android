package com.ian.mlp_tensorflow_winequality

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.ian.mlp_tensorflow_winequality.ml.ModeloMlpKeras
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType

class MainActivity : AppCompatActivity() {
    private lateinit var input1: EditText
    private lateinit var input2: EditText
    private lateinit var input3: EditText
    private lateinit var input4: EditText
    private lateinit var input5: EditText
    private lateinit var input6: EditText
    private lateinit var input7: EditText
    private lateinit var input8: EditText
    private lateinit var input9: EditText
    private lateinit var input10: EditText
    private lateinit var input11: EditText
    private lateinit var predictButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Inicializar vistas
        input1 = findViewById(R.id.input1)
        input2 = findViewById(R.id.input2)
        input3 = findViewById(R.id.input3)
        input4 = findViewById(R.id.input4)
        input5 = findViewById(R.id.input5)
        input6 = findViewById(R.id.input6)
        input7 = findViewById(R.id.input7)
        input8 = findViewById(R.id.input8)
        input9 = findViewById(R.id.input9)
        input10 = findViewById(R.id.input10)
        input11 = findViewById(R.id.input11)
        predictButton = findViewById(R.id.predictButton)

        // Configurar el botón para hacer la predicción
        predictButton.setOnClickListener {
            runModelInference()
        }
    }

    private fun runModelInference() {
        val model = ModeloMlpKeras.newInstance(this)

        // Crear el tensor de entrada
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 11), DataType.FLOAT32)

        // Cargar los valores de entrada
        val inputValues = floatArrayOf(
            input1.text.toString().toFloatOrNull() ?: 0f,
            input2.text.toString().toFloatOrNull() ?: 0f,
            input3.text.toString().toFloatOrNull() ?: 0f,
            input4.text.toString().toFloatOrNull() ?: 0f,
            input5.text.toString().toFloatOrNull() ?: 0f,
            input6.text.toString().toFloatOrNull() ?: 0f,
            input7.text.toString().toFloatOrNull() ?: 0f,
            input8.text.toString().toFloatOrNull() ?: 0f,
            input9.text.toString().toFloatOrNull() ?: 0f,
            input10.text.toString().toFloatOrNull() ?: 0f,
            input11.text.toString().toFloatOrNull() ?: 0f
        )

        inputFeature0.loadArray(inputValues)

        // Ejecutar la inferencia
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Obtener las probabilidades de las clases
        val probabilities = outputFeature0.floatArray

        // Crear el mensaje para el AlertDialog
        val probabilitiesText = StringBuilder("Probabilidades:\n")
        for (i in probabilities.indices) {
            probabilitiesText.append("Clase ${i + 1}: ${probabilities[i]}\n")
        }

        // Obtener la clase predicha (índice del valor máximo)
        val predictedClass = probabilities.indexOfMax()

        // Mostrar el resultado en un AlertDialog
        showResultDialog(probabilitiesText.toString(), predictedClass)

        // Liberar recursos del modelo
        model.close()
    }

    private fun showResultDialog(probabilities: String, predictedClass: Int) {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Resultado de Predicción")
        builder.setMessage("$probabilities\nClase predicha: ${predictedClass + 1}")
        builder.setPositiveButton("Aceptar") { dialog, _ -> dialog.dismiss() }
        builder.create().show()
    }

    // Extensión para encontrar el índice del valor máximo en un array
    private fun FloatArray.indexOfMax(): Int {
        var maxIndex = 0
        for (i in indices) {
            if (this[i] > this[maxIndex]) {
                maxIndex = i
            }
        }
        return maxIndex
    }
}
