#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

testing = function() {

  devtools::load_all(pkg = ".")

  dir.create(path = "output/", showWarnings = FALSE)

  data = getExperimentsData(tag = "study_14", n.runs = 10000) 

  # Several calls, this avoid memory errors
  # OpenML::populateOMLCache(data.ids = unique(data$data.id))
  # OpenML::populateOMLCache(task.id  = unique(data$task.id))
  # OpenML::populateOMLCache(flow.ids = unique(data$flow.id)) 
  # OpenML::populateOMLCache(run.ids  = unique(data$run.id))

  #-------------------------------------
  #  Initial plot - tasks overview
  #-------------------------------------

  g = getTasksInfoPlot(data = data)
  ggsave(g, file = "output/TaskInfoPlot.pdf", dpi = 400, width = 7, height = 4, units = "in")

  #-------------------------------------
  # Simple plots (performance)
  #-------------------------------------

  g1 = getSimplePlot(data = data, style = "boxplot", measure = "predictive.accuracy")
  ggsave(g1, file = "output/acc_boxplot.pdf", dpi = 400, width = 7, height = 3, units = "in")

  g2 = getSimplePlot(data = data, style = "violin", measure = "predictive.accuracy")
  ggsave(g2, file = "output/acc_violin.pdf", dpi = 400, width = 7, height = 3, units = "in")

  # Some other examples:
  # getSimplePlot(data = data, style = "violin",  measure = "f.measure")
  # getSimplePlot(data = data, style = "boxplot", measure = "kappa")

  #-------------------------------------
  # Simple plots (runtime) - there is no runtime yet
  #-------------------------------------

  # g3 = getSimplePlot(data = data, style = "boxplot", measure = "usercpu.time.millis")
  # getSimplePlot(data = data, style = "violin",  measure = "usercpu.time.millis)
  
  # g4 = getRuntimePlot(data = data, style = "boxplot")
  # getRuntimePlot(data = data, style = "violin")

  #-------------------------------------
  #  Performance matrix structure
  #-------------------------------------

  mat.acc = getPerfMatrix(data = data, measure = "predictive.accuracy")
  # Another examples: 
  # mat.auc = getPerfMatrix(data = data, measure = "area.ander.roc.curve")
  # mat.run = getPerfMatrix(data = data, measure = "usercpu.time.millis")

  g4 = getMatrixHeatMap(mat = mat.acc, prefix = "predictive \n accuracy")
  ggsave(g4, file = "output/acc_heatmap.pdf", dpi = 400, width = 10, height = 2.2, units = "in")


  #-------------------------------------
  #  Performance matrix plots
  #-------------------------------------

  # maximum performance scaled
  scaled.mat.acc = scaleMatrix(mat = mat.acc)

  g5 = getMatrixPlot(mat = scaled.mat.acc, style = "boxplot", prefix = "predictive accuracy")
  ggsave(g5, file = "output/max_acc_boxplot.pdf", dpi = 400, width = 7, height = 3, units = "in")

  # Another example:
  # getMatrixPlot(mat = scaled.mat.acc, style = "violin")

  g6 = getMatrixHeatMap(mat = scaled.mat.acc, prefix = "% of the maximum \n predictive accuracy")
  ggsave(g6, file = "output/max_acc_heatmap.pdf", dpi = 400, width = 10, height = 2.2, units = "in")
  
  #-------------------------------------
  #  Ranking structure
  #-------------------------------------

  rk.acc  = getRanking(mat = mat.acc)

  # Another examples: 
  # getRanking(mat = mat.auc,  descending = FALSE)
  # getRanking(mat = mat.run,  descending = TRUE)
  
  #-------------------------------------
  #  Ranking plots
  #-------------------------------------

  g7 = getRankingHeatMap(rk = rk.acc)
  ggsave(g7, file = "output/rk_heatMap.pdf", dpi = 400, width = 10, height = 2.2, units = "in")
  
   # getRankFrequencyPlot(rk = rk.acc, data = data, k = 5, version = "counter")  

  # best vc maximum
  best.algo = rk.acc$rk.avg$alg[which.min(rk.acc$rk.avg[,2])]
  
  g8 = getBestLinePlot(mat = mat, algo = best.algo)
  ggsave(g8, file = "output/best_vs_max.pdf", dpi = 400, width = 10, height = 3, units = "in")

  #-------------------------------------
  # Failed Jobs Bar plot
  #-------------------------------------

  g9 = getFailedBarPlot(mat = mat.acc)
  ggsave(g9, file = "output/failedJobs.pdf", dpi = 400, width = 7.19, height = 2.84, units = "in")

  #-------------------------------------
  #  Multicriteria plots
  #-------------------------------------

  # runtime vs performance (scatter, )

  # coverage

}


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
