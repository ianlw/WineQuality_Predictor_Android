import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

// Función para realizar la predicción
fun predictWineQuality(context: Context, inputData: FloatArray): FloatArray {
    // Cargar el modelo
    val model = ModeloMlpKeras.newInstance(context)

    // Crear un ByteBuffer para cargar los datos de entrada
    val byteBuffer = ByteBuffer.allocateDirect(4 * inputData.size) // 4 bytes por float
    byteBuffer.order(ByteOrder.nativeOrder()) // Establecer el orden de bytes a nativo

    // Llenar el ByteBuffer con los datos de entrada
    for (value in inputData) {
        byteBuffer.putFloat(value) // Cargar cada valor en el ByteBuffer
    }
    byteBuffer.rewind() // Reiniciar el buffer para leerlo desde el inicio

    // Crear el tensor de entrada y cargar el ByteBuffer
    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 11), DataType.FLOAT32)
    inputFeature0.loadBuffer(byteBuffer)

    // Ejecutar la inferencia
    val outputs = model.process(inputFeature0)
    val outputFeature0 = outputs.outputFeature0AsTensorBuffer

    // Liberar recursos del modelo
    model.close()

    // Devolver la predicción como un arreglo de floats
    return outputFeature0.floatArray
}

// Uso de la función
val inputData = floatArrayOf(12.2f, 0.45f, 0.49f, 1.4f, 0.075f, 3f, 6f, 0.9969f, 3.13f, 0.63f, 10.4f)
val prediction = predictWineQuality(context, inputData)

// Mostrar el resultado de la predicción
println("Predicción de calidad del vino: ${prediction.joinToString()}")
