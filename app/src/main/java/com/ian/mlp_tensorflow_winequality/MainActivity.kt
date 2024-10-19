package com.ian.mlp_tensorflow_winequality

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.ian.mlp_tensorflow_winequality.ml.ModeloMlpKeras
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType

class MainActivity : AppCompatActivity() {
    private lateinit var predictionTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        predictionTextView = findViewById(R.id.predictionTextView)

        // Aquí es donde cargamos el modelo y hacemos la predicción
        runModelInference()
    }

    private fun runModelInference() {
        val model = ModeloMlpKeras.newInstance(this)

        // Crear el tensor de entrada
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 11), DataType.FLOAT32)
        // Cargar los valores de entrada (ajusta según tus datos)
        val inputValues = floatArrayOf(6.8f, 0.36f, 0.32f, 1.8f, 0.067f, 4f, 8f, 0.9928f, 3.36f, 0.55f, 12.8f)
        inputFeature0.loadArray(inputValues)

        // Ejecutar la inferencia
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val probabilities = outputFeature0.floatArray
        val predictedClassIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
        predictionTextView.text = "Predicción: ${predictedClassIndex+4}\n" +
                "Probabilidad: ${probabilities[predictedClassIndex]}"
        // Mostrar el resultado en el TextView
        //predictionTextView.text = "Predicción: ${outputFeature0.floatArray[0]}\n" +
        //        "Predicción: ${outputFeature0.floatArray[1]}\n" +
        //        "Predicción: ${outputFeature0.floatArray[2]}\n" +
        //        "Predicción: ${outputFeature0.floatArray[3]}\n"

        // Liberar recursos del modelo
        model.close()
    }
}
