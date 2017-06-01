#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

testing = function() {

  devtools::load_all(pkg = ".")

  dir.create(path = "output/", showWarnings = FALSE)

  data = getExperimentsData(tag = "study_14", n.runs = 1000) 

  #-------------------------------------
  #  Initial plot - tasks overview
  #-------------------------------------

  g = getTasksInfoPlot(data = data)
  ggsave(g, file = "output/TaskInfoPlot.png", dpi = 500, width = 6, height = 3, units = "in")

  #-------------------------------------
  # Simple plots (performance)
  #-------------------------------------

  g1 = getSimplePlot(data = data, style = "boxplot", measure = "predictive.accuracy")
  ggsave(g1, file = "output/perf_boxplot.png", dpi = 500, width = 6, height = 2.5, units = "in")

  g2 = getSimplePlot(data = data, style = "violin", measure = "predictive.accuracy")
  ggsave(g2, file = "output/perf_violin.png", dpi = 500, width = 6, height = 2.5, units = "in")

  # Some other examples:
  # getSimplePlot(data = data, style = "violin",  measure = "f.measure")
  # getSimplePlot(data = data, style = "boxplot", measure = "kappa")

  #-------------------------------------
  # Simple plots (runtime)
  #-------------------------------------

  g3 = getSimplePlot(data = data, style = "boxplot", measure = "usercpu.time.millis")
  # getSimplePlot(data = data, style = "violin",  measure = "usercpu.time.millis.training")
  # getSimplePlot(data = data, style = "violin",  measure = "usercpu.time.millis.testing")

  # getSimplePlot(data = data, style = "boxplot", measure = "usercpu.time.millis", landscape = TRUE)
  # getSimplePlot(data = data, style = "violin", measure = "usercpu.time.millis", landscape = TRUE)
  # getRuntimePlot(data = data, style = "boxplot")
  # getRuntimePlot(data = data, style = "violin")




  #-------------------------------------
  #  Performance matrix structure
  #-------------------------------------

  mat.acc = getPerfMatrix(data = data, measure = "predictive.accuracy")
  # Another examples: 
  # mat.auc = getPerfMatrix(data = data, measure = "area.ander.roc.curve")
  # mat.run = getPerfMatrix(data = data, measure = "usercpu.time.millis")

  #-------------------------------------
  #  Performance matrix plots
  #-------------------------------------


  #-------------------------------------
  #  Ranking structure
  #-------------------------------------

  rk.acc  = getRanking(mat = mat.acc,  descending = TRUE)
  # Another examples: 
  # getRanking(mat = mat.auc,  descending = FALSE)
  # getRanking(mat = mat.run,  descending = TRUE)
  
  #-------------------------------------
  #  Ranking plots
  #-------------------------------------

  # getRankingHeatMap(data = rk.acc$rk)
  # getRankFrequencyPlot(rk = rk.acc, data = data, k = 5, version = "counter")  


  #-------------------------------------
  #-------------------------------------

# rk.acc   = getRanking(mat.acc, descending = TRUE)

# mat.time = getPerfMatrix(data = data, measure = "usercpu.time.millis")
# rk.time  = getRanking(mat.time, descending = FALSE)

# getRankingHeatMap(data = rk.time$rk)

# getRuntimePointPlots(data = data)

# scaled.mat.acc = scaleMatrix(mat = mat.acc)

# getMatrixBoxPlot(mat = scaled.mat.acc, prefix = "predictive accuracy", landscape = TRUE)

# getMatrixViolinPlot(mat = scaled.mat.acc, prefix = "predictive accuracy", landscape = TRUE)

# getMatrixHeatMap(mat = scaled.mat.acc, prefix = "predictive accuracy")

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
