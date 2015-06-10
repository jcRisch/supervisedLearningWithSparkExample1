import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, SVMWithSGD, SVMModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Created by Jean-Charles Risch on 08/06/2015.
 */
object exemple {
  def main(args: Array[String]): Unit = {
    println("Salut")

    val conf = new SparkConf().setAppName("iris").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val dataFile = "C:/iris.csv"

    val rawData = sc.textFile(dataFile)

    // Nettoyage des données

    // On supprime le header (première ligne)
    val noHeaderData = rawData.mapPartitionsWithIndex((i, iterator) =>
        if (i == 0 && iterator.hasNext) {
          iterator.next
          iterator
        }
        else iterator
    )

    noHeaderData.foreach(println)

    // Création du RDD[Vector]
    val cleanData = noHeaderData.map(line => {
      val valeurs = line.split(',')

      val label = valeurs(5) match {
        case "setosa" => 0d
        case "versicolor" => 1d
        case "virginica" => 2d
      }

      LabeledPoint(label, Vectors.dense(valeurs(1).toDouble, valeurs(2).toDouble, valeurs(3).toDouble, valeurs(4).toDouble))
    })

    // Split des données
    val splits = cleanData.randomSplit(Array(0.75, 0.25))
    val (trainingData, testData) = (splits(0), splits(1))

    // Création du modèle
    val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(trainingData)

    // prédit VS réel
    val preditVSreel: RDD[(Double, Double)] = testData.map { point =>
      val predit = model.predict(point.features)
      (predit, point.label)
    }

    // Evaluation du modèle
    val evaluation = new MulticlassMetrics(preditVSreel)
    println(evaluation.confusionMatrix)
    println(evaluation.precision)
    println(evaluation.recall)
    println(evaluation.fMeasure)

    // Prédiction de nouveaux individus
    val ind1 = Vectors.dense(5,3.6,1.4,0.2)
    val ind2 = Vectors.dense(5.5,2.4,3.7,1)
    val ind3 = Vectors.dense(6.7,3,5.2,2.3)

    // Affichage des prédictions
    println(model.predict(ind1))
    println(model.predict(ind2))
    println(model.predict(ind3))
  }
}
