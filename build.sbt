name := "iris"

version := "1.0"

scalaVersion := "2.11.6"

libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "1.0.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.1" excludeAll(
  ExclusionRule(organization = "org.apache.hadoop")
  )

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.1" excludeAll(
  ExclusionRule(organization = "org.apache.hadoop")
  )